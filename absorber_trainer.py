"""Absorber LoRA training pipeline for KV cache compression.

Trains LoRA adapters on the LLM's attention projections, but only active
during summary token forward passes. Doc tokens and Q+A tokens always go
through the frozen model.

Three-stage training step:
  Stage A (LoRA OFF, no grad): Teacher forward + per-doc KV cache generation
  Stage B (LoRA ON, with grad): Summary embedding + LoRA forward → extract summary KV
  Stage C (LoRA OFF, grad through cache): Student forward with compressed prefix → loss
"""

import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from absorber_eval import build_absorber_mask, unapply_rope
from block_attention import build_block_causal_mask, build_block_causal_mask_with_qa, build_prefix_causal_mask
from config import ModelConfig, TrainingConfig
from kv_cache_utils import _rotate_half, apply_rope_to_cache, build_dynamic_cache
from scheduler import CompressionScheduler

logger = logging.getLogger(__name__)


class AbsorberLoRATrainer:
    """Training loop for absorber LoRA KV cache compression.

    LoRA adapters are selectively enabled only during summary token forwards.
    Gradients flow: loss → student logits → compressed cache → LoRA weights.
    """

    def __init__(
        self,
        model,  # PEFT-wrapped model
        tokenizer: PreTrainedTokenizer,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model_config = model_config
        self.training_config = training_config
        self.device = device

        # Optimizer: only LoRA params
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            weight_decay=training_config.weight_decay,
        )

        # Compute total steps
        steps_per_epoch = math.ceil(
            len(train_loader) / training_config.gradient_accumulation_steps
        )
        self.total_steps = steps_per_epoch * training_config.num_epochs
        self.steps_per_epoch = steps_per_epoch

        # Auto-compute eval_steps: 4 times per epoch if not explicitly set
        if training_config.eval_steps is None:
            training_config.eval_steps = max(1, steps_per_epoch // 4)

        # Compression scheduler
        self.compression_scheduler = CompressionScheduler(
            training_config, self.total_steps
        )

        # Scaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=training_config.fp16)

        # WandB
        self.use_wandb = training_config.use_wandb
        if self.use_wandb:
            import wandb
            wandb.init(project="leefrag-absorber-lora", config={
                "model": model_config.__dict__,
                "training": training_config.__dict__,
            })

    def _get_rotary_emb(self):
        """Get rotary embedding through PEFT wrapper."""
        return self.model.get_base_model().model.rotary_emb

    def _get_embed_tokens(self):
        """Get embedding layer through PEFT wrapper."""
        return self.model.get_base_model().model.embed_tokens

    def _teacher_forward(self, batch: dict):
        """Stage A: LoRA OFF, no grad.

        1. Block-diagonal forward on [preamble+docs | Q+A] → teacher logits
        2. Single block-diagonal forward on [preamble+docs] with per-block
           position IDs → concatenated KV cache, sliced per doc

        Returns:
            teacher_logits: [1, qa_len, vocab] or None if ce_only
            doc_kv_caches: list of DynamicCache, one per doc block
        """
        doc_token_ids = batch["doc_token_ids"]
        doc_lengths = batch["doc_lengths"]
        preamble_ids = batch["preamble_ids"]
        stage_b_input_ids = batch["stage_b_input_ids"].to(self.device)

        preamble_len = preamble_ids.shape[0]
        block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

        teacher_logits = None

        # Teacher forward for KL (skip if CE-only)
        if not self.training_config.ce_only_loss:
            doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(self.device)
            preamble = preamble_ids.unsqueeze(0).to(self.device)
            full_input = torch.cat([preamble, doc_concat, stage_b_input_ids], dim=1)
            qa_length = stage_b_input_ids.shape[1]

            dtype = torch.float16 if self.training_config.fp16 else torch.float32
            attn_mask = build_block_causal_mask_with_qa(
                block_lengths, qa_length, dtype=dtype, device=self.device
            )

            with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
                outputs = self.model(
                    input_ids=full_input,
                    attention_mask=attn_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
            doc_total = sum(block_lengths)
            teacher_logits = outputs.logits[:, doc_total:, :]

        # Single block-diagonal forward for all doc KV caches
        # Per-block position IDs so each doc's RoPE starts at 0
        doc_concat = torch.cat(
            [torch.cat([preamble_ids, doc_token_ids[0]], dim=0)] + doc_token_ids[1:],
            dim=0,
        ).unsqueeze(0).to(self.device)

        position_ids = torch.zeros(sum(block_lengths), dtype=torch.long, device=self.device)
        offset = 0
        for bl in block_lengths:
            position_ids[offset:offset + bl] = torch.arange(bl, device=self.device)
            offset += bl
        position_ids = position_ids.unsqueeze(0)

        dtype = torch.float16 if self.training_config.fp16 else torch.float32
        attn_mask = build_block_causal_mask(
            block_lengths, dtype=dtype, device=self.device
        )

        with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
            out = self.model(
                input_ids=doc_concat,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=False,
            )

        # Slice the concatenated KV cache per doc block
        from transformers.cache_utils import DynamicCache
        full_cache = out.past_key_values
        num_layers = self.model_config.num_layers
        doc_kv_caches = []
        offset = 0
        for bl in block_lengths:
            doc_cache = DynamicCache()
            for li in range(num_layers):
                k_slice = full_cache.layers[li].keys[:, :, offset:offset + bl, :]
                v_slice = full_cache.layers[li].values[:, :, offset:offset + bl, :]
                doc_cache.update(k_slice.clone(), v_slice.clone(), li)
            if self.training_config.offload_stage_a_to_cpu:
                for layer in doc_cache.layers:
                    layer.keys = layer.keys.cpu()
                    layer.values = layer.values.cpu()
            doc_kv_caches.append(doc_cache)
            offset += bl

        if teacher_logits is not None and self.training_config.offload_stage_a_to_cpu:
            teacher_logits = teacher_logits.cpu()

        return teacher_logits, doc_kv_caches

    def _absorber_compress_single_doc(
        self,
        doc_ids: torch.Tensor,
        doc_kv,  # DynamicCache
        compression_ratio: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Stage B (per-doc): LoRA ON, with grad.

        1. Embed doc tokens (frozen) → mean-pool to K summary embeddings
        2. Forward with doc_kv as past → summary output
        3. Extract summary KV from cache at positions [N:N+K]
        4. Un-apply RoPE at positions [N, N+K)

        Returns:
            List of (k_raw, v) per layer, each [1, num_kv_heads, K, head_dim].
        """
        doc_ids = doc_ids.to(self.device)
        N = doc_ids.shape[0]
        K = max(1, N // compression_ratio)

        # Move doc KV back to GPU if offloaded
        if self.training_config.offload_stage_a_to_cpu:
            for layer in doc_kv.layers:
                layer.keys = layer.keys.to(self.device)
                layer.values = layer.values.to(self.device)

        # 1. Embed doc tokens (frozen, no grad through embeddings)
        embed_tokens = self._get_embed_tokens()
        with torch.no_grad():
            doc_embeds = embed_tokens(doc_ids.unsqueeze(0))  # [1, N, hidden]

        # 2. Mean-pool to K summary embeddings
        doc_t = doc_embeds.permute(0, 2, 1)  # [1, hidden, N]
        summary_t = F.adaptive_avg_pool1d(doc_t, K)  # [1, hidden, K]
        summary_embeds = summary_t.permute(0, 2, 1)  # [1, K, hidden]

        # 3. Forward with doc_kv as past, positions [N, N+K)
        position_ids = torch.arange(N, N + K, device=self.device).unsqueeze(0)

        # Clone doc_kv so the original isn't mutated by the forward pass
        from transformers.cache_utils import DynamicCache
        doc_kv_clone = DynamicCache()
        for layer in doc_kv.layers:
            doc_kv_clone.update(layer.keys.clone(), layer.values.clone(), len(doc_kv_clone.layers))

        with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
            outputs = self.model(
                inputs_embeds=summary_embeds,
                past_key_values=doc_kv_clone,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=False,
            )

        # 4. Extract summary KV at positions [N:N+K] from cache
        cache = outputs.past_key_values
        num_layers = self.model_config.num_layers

        # Get RoPE cos/sin at positions [N, N+K) for un-application
        rotary_emb = self._get_rotary_emb()
        summary_pos_ids = torch.arange(N, N + K, device=self.device).unsqueeze(0)
        dummy = cache.layers[0].keys[:, :, N:N + K, :]
        cos, sin = rotary_emb(dummy, position_ids=summary_pos_ids)
        cos = cos.unsqueeze(1)  # [1, 1, K, head_dim]
        sin = sin.unsqueeze(1)

        raw_kv_pairs = []
        for layer_idx in range(num_layers):
            k_full = cache.layers[layer_idx].keys
            v_full = cache.layers[layer_idx].values
            # Extract summary portion
            k_summary = k_full[:, :, N:N + K, :]
            v_summary = v_full[:, :, N:N + K, :]
            # Un-apply RoPE
            k_raw = unapply_rope(k_summary, cos, sin)
            raw_kv_pairs.append((k_raw, v_summary))

        return raw_kv_pairs

    def _absorber_compress_all(
        self,
        doc_token_ids: list[torch.Tensor],
        doc_kv_caches: list,
        preamble_ids: torch.Tensor,
        compression_ratio: int,
    ):
        """Stage B: Loop over all docs, concat raw KV, re-apply RoPE.

        Returns:
            DynamicCache with compressed KV, RoPE applied at [0, total_K).
            empirical_ratio: average compression ratio across docs.
        """
        all_raw_kv = []
        total_doc_tokens = 0
        total_summary_tokens = 0

        for doc_idx in range(len(doc_token_ids)):
            if doc_idx == 0:
                block_ids = torch.cat([preamble_ids, doc_token_ids[0]], dim=0)
            else:
                block_ids = doc_token_ids[doc_idx]

            raw_kv = self._absorber_compress_single_doc(
                block_ids, doc_kv_caches[doc_idx], compression_ratio
            )
            all_raw_kv.append(raw_kv)

            N = block_ids.shape[0]
            K = max(1, N // compression_ratio)
            total_doc_tokens += N
            total_summary_tokens += K

        # Concat raw KV across documents per layer
        num_layers = self.model_config.num_layers
        kv_pairs = []
        for layer_idx in range(num_layers):
            keys = [doc_kv[layer_idx][0] for doc_kv in all_raw_kv]
            values = [doc_kv[layer_idx][1] for doc_kv in all_raw_kv]
            concat_k = torch.cat(keys, dim=2)
            concat_v = torch.cat(values, dim=2)
            kv_pairs.append((concat_k, concat_v))

        # Build DynamicCache and apply RoPE at global [0, total_K)
        cache = build_dynamic_cache(kv_pairs)
        rotary_emb = self._get_rotary_emb()
        cache = apply_rope_to_cache(cache, num_layers, rotary_emb)

        empirical_ratio = total_doc_tokens / max(total_summary_tokens, 1)
        return cache, empirical_ratio

    def _training_step(
        self, batch: dict, compression_ratio: int, global_step: int = 0,
    ) -> torch.Tensor | None:
        """Execute one training step (Stage A → B → C).

        Returns the combined loss tensor or None if batch should be skipped.
        """
        doc_token_ids = batch["doc_token_ids"]
        doc_lengths = batch["doc_lengths"]
        preamble_ids = batch["preamble_ids"]
        stage_b_input_ids = batch["stage_b_input_ids"].to(self.device)
        stage_b_labels = batch["stage_b_labels"].to(self.device)

        if not doc_token_ids or sum(doc_lengths) == 0:
            return None

        # === Stage A: LoRA OFF, no grad ===
        with torch.no_grad():
            self.model.disable_adapter_layers()
            teacher_logits, doc_kv_caches = self._teacher_forward(batch)
            self.model.enable_adapter_layers()

        if teacher_logits is not None and self.training_config.offload_stage_a_to_cpu:
            teacher_logits = teacher_logits.to(self.device)

        # === Stage B: LoRA ON, with grad ===
        with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
            compressed_cache, empirical_ratio = self._absorber_compress_all(
                doc_token_ids, doc_kv_caches, preamble_ids, compression_ratio,
            )

        # === Stage C: LoRA OFF, grad flows through cache ===
        self.model.disable_adapter_layers()

        with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
            prefix_len = compressed_cache.get_seq_length()
            seq_len = stage_b_input_ids.shape[1]
            dtype = torch.float16 if self.training_config.fp16 else torch.float32
            prefix_mask = build_prefix_causal_mask(
                prefix_len, seq_len, dtype=dtype, device=self.device,
            )

            outputs = self.model(
                input_ids=stage_b_input_ids,
                past_key_values=compressed_cache,
                attention_mask=prefix_mask,
                use_cache=False,
            )

            student_logits = outputs.logits

            # CE loss on answer tokens
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = stage_b_labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if self.training_config.ce_only_loss:
                total_loss = ce_loss
                self._last_kl_loss = 0.0
            else:
                kl_loss = self._compute_kl_loss(
                    teacher_logits, student_logits, stage_b_labels
                )
                total_loss = ce_loss + self.training_config.kl_weight * kl_loss
                self._last_kl_loss = kl_loss.item()

        self.model.enable_adapter_layers()

        # Stash for logging
        self._last_ce_loss = ce_loss.item()
        self._last_empirical_ratio = empirical_ratio

        return total_loss

    def _compute_kl_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence on answer tokens (same as trainer.py)."""
        teacher_shift = teacher_logits[:, :-1, :].contiguous()
        student_shift = student_logits[:, :-1, :].contiguous()
        label_shift = labels[:, 1:].contiguous()

        valid_mask = label_shift != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=student_logits.device)

        valid_indices = valid_mask.view(-1).nonzero(as_tuple=True)[0]
        teacher_flat = teacher_shift.view(-1, teacher_shift.size(-1))[valid_indices]
        student_flat = student_shift.view(-1, student_shift.size(-1))[valid_indices]

        if self.training_config.kl_top_k > 0:
            k = self.training_config.kl_top_k
            top_k_vals, top_k_idx = teacher_flat.topk(k, dim=-1)
            student_gathered = student_flat.gather(-1, top_k_idx)
            teacher_log_probs = F.log_softmax(top_k_vals, dim=-1)
            student_log_probs = F.log_softmax(student_gathered, dim=-1)
        else:
            teacher_log_probs = F.log_softmax(teacher_flat, dim=-1)
            student_log_probs = F.log_softmax(student_flat, dim=-1)

        kl = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            log_target=True,
            reduction="batchmean",
        )
        return kl

    def _update_lr(self, step: int):
        """Per-phase warm restart (linear warmup + cosine decay)."""
        steps_per_phase = self.compression_scheduler.steps_per_phase
        phase = self.compression_scheduler.get_phase(step)
        phase_step = step - phase * steps_per_phase
        phase_warmup = int(steps_per_phase * self.training_config.warmup_ratio)

        if phase_step < phase_warmup:
            lr_scale = phase_step / max(1, phase_warmup)
        else:
            progress = (phase_step - phase_warmup) / max(1, steps_per_phase - phase_warmup)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.training_config.learning_rate * lr_scale

    def train(self):
        """Main training loop."""
        self.model.train()

        global_step = 0
        accumulation_count = 0

        accum_loss = 0.0
        accum_ce = 0.0
        accum_kl = 0.0
        accum_empirical_ratio = 0.0
        accum_micro_batches = 0

        for epoch in range(self.training_config.num_epochs):
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.training_config.num_epochs}",
                leave=True,
            )
            prev_phase = -1
            for batch_idx, batch in enumerate(pbar):
                compression_ratio = self.compression_scheduler.get_compression_ratio(
                    global_step
                )

                phase = self.compression_scheduler.get_phase(global_step)
                if phase != prev_phase:
                    logger.info(
                        f"Phase {phase}: compression={compression_ratio}x, "
                        f"LR warm-restart at step {global_step}"
                    )
                    prev_phase = phase

                loss = self._training_step(batch, compression_ratio, global_step)

                if loss is None:
                    continue

                loss = loss / self.training_config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                accumulation_count += 1

                accum_loss += loss.item() * self.training_config.gradient_accumulation_steps
                accum_ce += self._last_ce_loss
                accum_kl += self._last_kl_loss
                accum_empirical_ratio += self._last_empirical_ratio
                accum_micro_batches += 1

                if accumulation_count % self.training_config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    lora_params = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(
                        lora_params, self.training_config.max_grad_norm,
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    self._update_lr(global_step)
                    global_step += 1

                    avg_loss = accum_loss / accum_micro_batches
                    avg_ce = accum_ce / accum_micro_batches
                    avg_kl = accum_kl / accum_micro_batches
                    avg_empirical_ratio = accum_empirical_ratio / accum_micro_batches

                    accum_loss = 0.0
                    accum_ce = 0.0
                    accum_kl = 0.0
                    accum_empirical_ratio = 0.0
                    accum_micro_batches = 0

                    lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        ce=f"{avg_ce:.4f}",
                        kl=f"{avg_kl:.4f}",
                        lr=f"{lr:.2e}",
                        comp=f"{compression_ratio}x",
                        step=f"{global_step}/{self.total_steps}",
                    )

                    if global_step % self.training_config.logging_steps == 0:
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/ce_loss": avg_ce,
                                "train/kl_loss": avg_kl,
                                "train/lr": lr,
                                "train/compression_ratio": compression_ratio,
                                "train/empirical_compression_ratio": avg_empirical_ratio,
                                "train/phase": phase,
                                "train/global_step": global_step,
                            })

                    if global_step % self.training_config.eval_steps == 0:
                        eval_ce, eval_kl = self.evaluate(compression_ratio)
                        eval_total = eval_ce + self.training_config.kl_weight * eval_kl
                        tqdm.write(
                            f"Eval @ step {global_step}: "
                            f"CE={eval_ce:.4f}, KL={eval_kl:.4f}, "
                            f"total={eval_total:.4f}, "
                            f"ppl={math.exp(eval_ce):.2f}, "
                            f"compression={compression_ratio}x"
                        )
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "eval/ce_loss": eval_ce,
                                "eval/kl_loss": eval_kl,
                                "eval/total_loss": eval_total,
                                "eval/perplexity": math.exp(eval_ce),
                                "eval/compression_ratio": compression_ratio,
                            })
                        self.model.train()

                    if global_step % self.training_config.save_steps == 0:
                        self._save_checkpoint(global_step, compression_ratio)

        # Final save
        self._save_checkpoint(global_step, compression_ratio)
        logger.info("Training complete.")

    @torch.no_grad()
    def evaluate(self, compression_ratio: int) -> tuple[float, float]:
        """Evaluate on the eval set. Returns (avg CE, avg KL)."""
        self.model.eval()
        total_ce = 0.0
        total_kl = 0.0
        num_batches = 0

        eval_pbar = tqdm(self.eval_loader, desc="Evaluating", leave=False)
        for batch in eval_pbar:
            doc_token_ids = batch["doc_token_ids"]
            doc_lengths = batch["doc_lengths"]
            preamble_ids = batch["preamble_ids"]
            stage_b_input_ids = batch["stage_b_input_ids"].to(self.device)
            stage_b_labels = batch["stage_b_labels"].to(self.device)

            if not doc_token_ids or sum(doc_lengths) == 0:
                continue

            preamble_len = preamble_ids.shape[0]
            block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

            # Stage A: teacher + doc KV (LoRA OFF)
            self.model.disable_adapter_layers()

            teacher_logits = None
            if not self.training_config.ce_only_loss:
                doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(self.device)
                preamble_t = preamble_ids.unsqueeze(0).to(self.device)
                full_input = torch.cat([preamble_t, doc_concat, stage_b_input_ids], dim=1)
                qa_length = stage_b_input_ids.shape[1]
                dtype = torch.float16 if self.training_config.fp16 else torch.float32
                attn_mask = build_block_causal_mask_with_qa(
                    block_lengths, qa_length, dtype=dtype, device=self.device
                )
                with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
                    out = self.model(
                        input_ids=full_input, attention_mask=attn_mask,
                        use_cache=False, output_hidden_states=False,
                    )
                doc_total = sum(block_lengths)
                teacher_logits = out.logits[:, doc_total:, :]

            # Single block-diagonal forward for all doc KV caches
            doc_concat = torch.cat(
                [torch.cat([preamble_ids, doc_token_ids[0]], dim=0)] + doc_token_ids[1:],
                dim=0,
            ).unsqueeze(0).to(self.device)

            position_ids = torch.zeros(sum(block_lengths), dtype=torch.long, device=self.device)
            bl_offset = 0
            for bl in block_lengths:
                position_ids[bl_offset:bl_offset + bl] = torch.arange(bl, device=self.device)
                bl_offset += bl
            position_ids = position_ids.unsqueeze(0)

            doc_attn_mask = build_block_causal_mask(
                block_lengths, dtype=torch.float16 if self.training_config.fp16 else torch.float32,
                device=self.device,
            )
            with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
                out = self.model(
                    input_ids=doc_concat, attention_mask=doc_attn_mask,
                    position_ids=position_ids, use_cache=True, output_hidden_states=False,
                )

            from transformers.cache_utils import DynamicCache
            full_cache = out.past_key_values
            num_layers = self.model_config.num_layers
            doc_kv_caches = []
            bl_offset = 0
            for bl in block_lengths:
                doc_cache = DynamicCache()
                for li in range(num_layers):
                    k_slice = full_cache.layers[li].keys[:, :, bl_offset:bl_offset + bl, :]
                    v_slice = full_cache.layers[li].values[:, :, bl_offset:bl_offset + bl, :]
                    doc_cache.update(k_slice.clone(), v_slice.clone(), li)
                doc_kv_caches.append(doc_cache)
                bl_offset += bl

            self.model.enable_adapter_layers()

            # Stage B: absorber compress (LoRA ON — but no grad in eval)
            with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
                all_raw_kv = []
                for doc_idx in range(len(doc_token_ids)):
                    if doc_idx == 0:
                        block_ids = torch.cat([preamble_ids, doc_token_ids[0]], dim=0)
                    else:
                        block_ids = doc_token_ids[doc_idx]
                    block_ids = block_ids.to(self.device)
                    N = block_ids.shape[0]
                    K = max(1, N // compression_ratio)

                    embed_tokens = self._get_embed_tokens()
                    doc_embeds = embed_tokens(block_ids.unsqueeze(0))
                    doc_t = doc_embeds.permute(0, 2, 1)
                    summary_t = F.adaptive_avg_pool1d(doc_t, K)
                    summary_embeds = summary_t.permute(0, 2, 1)

                    position_ids = torch.arange(N, N + K, device=self.device).unsqueeze(0)

                    from transformers.cache_utils import DynamicCache
                    doc_kv_clone = DynamicCache()
                    for layer in doc_kv_caches[doc_idx].layers:
                        doc_kv_clone.update(
                            layer.keys.clone(), layer.values.clone(),
                            len(doc_kv_clone.layers),
                        )

                    out = self.model(
                        inputs_embeds=summary_embeds,
                        past_key_values=doc_kv_clone,
                        position_ids=position_ids,
                        use_cache=True,
                        output_hidden_states=False,
                    )

                    cache = out.past_key_values
                    rotary_emb = self._get_rotary_emb()
                    summary_pos = torch.arange(N, N + K, device=self.device).unsqueeze(0)
                    dummy = cache.layers[0].keys[:, :, N:N + K, :]
                    cos, sin = rotary_emb(dummy, position_ids=summary_pos)
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)

                    raw_kv = []
                    for li in range(self.model_config.num_layers):
                        k_s = cache.layers[li].keys[:, :, N:N + K, :]
                        v_s = cache.layers[li].values[:, :, N:N + K, :]
                        k_raw = unapply_rope(k_s, cos, sin)
                        raw_kv.append((k_raw, v_s))
                    all_raw_kv.append(raw_kv)

                # Concat + RoPE
                num_layers = self.model_config.num_layers
                kv_pairs = []
                for li in range(num_layers):
                    keys = [d[li][0] for d in all_raw_kv]
                    values = [d[li][1] for d in all_raw_kv]
                    kv_pairs.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
                compressed_cache = build_dynamic_cache(kv_pairs)
                rotary_emb = self._get_rotary_emb()
                compressed_cache = apply_rope_to_cache(compressed_cache, num_layers, rotary_emb)

            # Stage C: student forward (LoRA OFF)
            self.model.disable_adapter_layers()

            with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
                prefix_len = compressed_cache.get_seq_length()
                seq_len = stage_b_input_ids.shape[1]
                dtype = torch.float16 if self.training_config.fp16 else torch.float32
                prefix_mask = build_prefix_causal_mask(
                    prefix_len, seq_len, dtype=dtype, device=self.device,
                )
                out = self.model(
                    input_ids=stage_b_input_ids,
                    past_key_values=compressed_cache,
                    attention_mask=prefix_mask,
                    use_cache=False,
                )
                student_logits = out.logits
                shift_logits = student_logits[:, :-1, :].contiguous()
                shift_labels = stage_b_labels[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                if self.training_config.ce_only_loss or teacher_logits is None:
                    kl_loss = torch.tensor(0.0)
                else:
                    kl_loss = self._compute_kl_loss(
                        teacher_logits, student_logits, stage_b_labels
                    )

            self.model.enable_adapter_layers()

            total_ce += ce_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
            eval_pbar.set_postfix(
                ce=f"{total_ce / num_batches:.4f}",
                kl=f"{total_kl / num_batches:.4f}",
            )

        return (
            total_ce / max(num_batches, 1),
            total_kl / max(num_batches, 1),
        )

    def _save_checkpoint(self, step: int, compression_ratio: int):
        """Save LoRA adapter + optimizer + scaler checkpoint."""
        save_dir = os.path.join(
            self.training_config.output_dir, f"absorber-checkpoint-{step}"
        )
        os.makedirs(save_dir, exist_ok=True)

        from peft import get_peft_model_state_dict
        ckpt = {
            "step": step,
            "compression_ratio": compression_ratio,
            "lora_state_dict": get_peft_model_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(ckpt, os.path.join(save_dir, "checkpoint.pt"))
        logger.info(f"Checkpoint saved at {save_dir}")

    def verify_gradient_flow(self):
        """Verify LoRA parameters receive gradients (call after first backward)."""
        has_grad = False
        zero_grad_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
                else:
                    zero_grad_params.append(name)
        if has_grad:
            logger.info("Gradient flow verified: LoRA parameters have non-zero gradients.")
        else:
            logger.warning(
                "No gradients flowing to LoRA! Check the computation graph."
            )
        if zero_grad_params:
            logger.warning(f"LoRA params with zero gradients: {zero_grad_params[:5]}...")
