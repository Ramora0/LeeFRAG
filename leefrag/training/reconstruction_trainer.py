import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from leefrag.config import ModelConfig, QFormerConfig, TrainingConfig
from leefrag.model.block_attention import build_block_causal_mask, build_block_prefix_causal_mask
from leefrag.utils.kv_cache_utils import (
    apply_rope_to_cache_blocked,
    build_blocked_position_ids,
    concat_compressed_caches,
    extract_doc_hidden_states,
)
from leefrag.model.qformer import QFormerKVCompressor
from leefrag.training.scheduler import CompressionScheduler

logger = logging.getLogger(__name__)


class ReconstructionTrainer:
    """Reconstruction pretraining for KV cache compression.

    Stage A (no grad): Run frozen LLM on [preamble | doc0 | doc1 | ...] with
        block-diagonal causal mask. Extract per-document hidden states and
        teacher logits at document positions.
    Stage B (grad): Q-Former compresses hidden states → per-doc KV caches →
        block-prefix-causal LLM forward → CE loss on document tokens + optional
        KL divergence against teacher logits.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        qformer: QFormerKVCompressor,
        tokenizer: PreTrainedTokenizer,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model
        self.qformer = qformer
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.bypass = False

        # Optimizer (Q-Former parameters only — no LoRA for reconstruction)
        self.optimizer = torch.optim.AdamW(
            list(qformer.parameters()),
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
            wandb.init(project="leefrag-reconstruction", config={
                "model": model_config.__dict__,
                "qformer": QFormerConfig().__dict__,
                "training": training_config.__dict__,
            })

    def train(self):
        """Main training loop."""
        self.model.eval()
        self.qformer.train()

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
                    torch.nn.utils.clip_grad_norm_(
                        self.qformer.parameters(),
                        self.training_config.max_grad_norm,
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
                    phase = self.compression_scheduler.get_phase(global_step)
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
                        kl_weight = self.training_config.kl_weight
                        eval_total = eval_ce + kl_weight * eval_kl
                        tqdm.write(
                            f"Eval @ step {global_step}: "
                            f"CE={eval_ce:.4f}, KL={eval_kl:.4f}, "
                            f"total={eval_total:.4f}, "
                            f"ppl={math.exp(min(eval_ce, 20)):.2f}, "
                            f"compression={compression_ratio}x"
                        )
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "eval/ce_loss": eval_ce,
                                "eval/kl_loss": eval_kl,
                                "eval/total_loss": eval_total,
                                "eval/perplexity": math.exp(min(eval_ce, 20)),
                                "eval/compression_ratio": compression_ratio,
                            })
                        self.qformer.train()

                    if global_step % self.training_config.save_steps == 0:
                        self._save_checkpoint(global_step, compression_ratio)

        # Final save
        self._save_checkpoint(global_step, compression_ratio)
        logger.info("Training complete.")

    def _training_step(
        self, batch: dict, compression_ratio: int, global_step: int = 0,
    ) -> torch.Tensor | None:
        """Execute one training step (Stage A + Stage B)."""
        doc_token_ids = batch["doc_token_ids"]
        doc_lengths = batch["doc_lengths"]
        preamble_ids = batch["preamble_ids"]
        stage_b_input_ids = batch["stage_b_input_ids"].to(self.device)
        stage_b_labels = batch["stage_b_labels"].to(self.device)
        input_block_lengths = batch["input_block_lengths"]

        if not doc_token_ids or sum(doc_lengths) == 0:
            return None

        # Block lengths: preamble merges with first doc, rest are standalone
        preamble_len = preamble_ids.shape[0]
        block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

        # === Stage A: Run frozen LLM on [preamble+docs], extract hidden states + teacher logits ===
        with torch.no_grad():
            doc_hidden_states, teacher_logits_list = self._stage_a(
                doc_token_ids, doc_lengths, preamble_ids, block_lengths,
            )

        if self.training_config.offload_stage_a_to_cpu:
            doc_hidden_states = [
                [hs.to(self.device) for hs in doc_hs]
                for doc_hs in doc_hidden_states
            ]
            if teacher_logits_list is not None:
                teacher_logits_list = [t.to(self.device) for t in teacher_logits_list]

        # === Stage B: Compress hidden states and compute loss ===
        with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
            loss = self._stage_b(
                doc_hidden_states, compression_ratio,
                stage_b_input_ids, stage_b_labels,
                input_block_lengths, teacher_logits_list,
            )

        return loss

    def _get_rotary_emb(self):
        """Get rotary embedding from the LLM."""
        return self.model.model.rotary_emb

    def _stage_a(
        self,
        doc_token_ids: list[torch.Tensor],
        doc_lengths: list[int],
        preamble_ids: torch.Tensor,
        block_lengths: list[int],
    ) -> tuple[list[list[torch.Tensor]], list[torch.Tensor] | None]:
        """Stage A: Run frozen LLM on [preamble+docs] with block-diagonal causal mask.

        No Q+A tokens — just documents.

        Returns:
            per_doc_hidden: Per-block hidden states for Q-Former compression.
            teacher_logits_list: Per-document teacher logits at doc token positions,
                or None if ce_only_loss.
        """
        # Concatenate: [preamble | doc0 | doc1 | ...]
        doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(self.device)
        preamble = preamble_ids.unsqueeze(0).to(self.device)
        full_input = torch.cat([preamble, doc_concat], dim=1)

        dtype = torch.float16 if self.training_config.fp16 else torch.float32
        attn_mask = build_block_causal_mask(
            block_lengths, dtype=dtype, device=self.device
        )

        outputs = self.model(
            input_ids=full_input,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract per-block hidden states
        per_doc_hidden = extract_doc_hidden_states(
            outputs.hidden_states, block_lengths, self.model_config.num_layers
        )

        # Extract teacher logits at document positions only
        teacher_logits_list = None
        if not self.training_config.ce_only_loss:
            teacher_logits_list = []
            offset = 0
            preamble_len = block_lengths[0] - doc_lengths[0]
            for i, b_len in enumerate(block_lengths):
                if i == 0:
                    # Block 0 = preamble+doc0 → extract logits at doc0 positions only
                    doc_start = offset + preamble_len
                    doc_end = offset + b_len
                else:
                    doc_start = offset
                    doc_end = offset + b_len
                teacher_logits_list.append(
                    outputs.logits[:, doc_start:doc_end, :]
                )
                offset += b_len

        if self.training_config.offload_stage_a_to_cpu:
            per_doc_hidden = [
                [hs.cpu() for hs in doc_hs]
                for doc_hs in per_doc_hidden
            ]
            if teacher_logits_list is not None:
                teacher_logits_list = [t.cpu() for t in teacher_logits_list]

        return per_doc_hidden, teacher_logits_list

    def _stage_b(
        self,
        doc_hidden_states: list[list[torch.Tensor]],
        compression_ratio: int,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        input_block_lengths: list[int],
        teacher_logits_list: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Stage B: Compress hidden states, run LLM with block-prefix mask, compute loss."""
        # Compress each document's hidden states into KV caches
        per_doc_compressed = []
        empirical_ratios = []
        for doc_hs in doc_hidden_states:
            compressed = self.qformer(doc_hs, compression_ratio, bypass=self.bypass)
            per_doc_compressed.append(compressed)
            doc_len = doc_hs[0].shape[1]
            num_queries = compressed[0][0].shape[2]
            empirical_ratios.append(doc_len / num_queries)

        # Get per-doc prefix lengths (may vary due to different doc lengths)
        prefix_lengths = [
            doc_cache[0][0].shape[2] for doc_cache in per_doc_compressed
        ]

        # Concatenate compressed caches from all documents
        compressed_cache = concat_compressed_caches(
            per_doc_compressed, self.model_config.num_layers
        )

        # Apply RoPE with per-block positions (block-diagonal prefix)
        rotary_emb = self._get_rotary_emb()
        compressed_cache = apply_rope_to_cache_blocked(
            compressed_cache, self.model_config.num_layers, rotary_emb,
            prefix_lengths,
        )

        # Build block-prefix-causal mask
        dtype = torch.float16 if self.training_config.fp16 else torch.float32
        attn_mask = build_block_prefix_causal_mask(
            prefix_lengths, input_block_lengths,
            dtype=dtype, device=self.device,
        )

        # Build per-block position IDs for input tokens
        position_ids = build_blocked_position_ids(
            prefix_lengths, input_block_lengths, device=self.device,
        )

        # Forward through frozen LLM with compressed KV prefix
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=compressed_cache,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        student_logits = outputs.logits  # [batch, seq_len, vocab_size]

        # === CE loss on document tokens ===
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if self.training_config.ce_only_loss or teacher_logits_list is None:
            total_loss = ce_loss
            self._last_kl_loss = 0.0
        else:
            kl_loss = self._compute_reconstruction_kl(
                teacher_logits_list, student_logits, labels, input_block_lengths,
            )
            total_loss = ce_loss + self.training_config.kl_weight * kl_loss
            self._last_kl_loss = kl_loss.item()

        self._last_ce_loss = ce_loss.item()
        self._last_empirical_ratio = sum(empirical_ratios) / len(empirical_ratios)

        return total_loss

    def _compute_reconstruction_kl(
        self,
        teacher_logits_list: list[torch.Tensor],
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        input_block_lengths: list[int],
    ) -> torch.Tensor:
        """Compute per-document KL divergence for reconstruction.

        Teacher predicts doc[1..N] from full causal context.
        Student predicts doc tokens from compressed prefix + prompt + doc[0..N-1].
        We align at positions predicting doc[1..N-1] (the overlap).

        Args:
            teacher_logits_list: Per-doc teacher logits, each [1, doc_len, vocab].
            student_logits: Full student logits [1, total_seq, vocab].
            labels: [1, total_seq] with -100 for prompt positions.
            input_block_lengths: Per-doc input block lengths.

        Returns:
            Scalar KL loss averaged over docs.
        """
        total_kl = torch.tensor(0.0, device=student_logits.device)
        num_valid_docs = 0

        input_offset = 0
        for doc_idx, (teacher_doc, i_len) in enumerate(
            zip(teacher_logits_list, input_block_lengths)
        ):
            # Student logits for this block
            student_block = student_logits[:, input_offset : input_offset + i_len, :]
            block_labels = labels[:, input_offset : input_offset + i_len]

            # Shift for next-token prediction
            teacher_shift = teacher_doc[:, :-1, :].contiguous()
            student_shift = student_block[:, :-1, :].contiguous()
            label_shift = block_labels[:, 1:].contiguous()

            # Only compute KL at positions where labels are real tokens (doc tokens)
            valid_mask = label_shift != -100
            if not valid_mask.any():
                input_offset += i_len
                continue

            # Teacher doc logits: teacher predicts doc[1..N] from doc[0..N-1]
            # Student block logits: student predicts tokens from compressed prefix + prompt + doc
            # Alignment: both predict the same next tokens at doc positions
            # The teacher has doc_len logits, student block has i_len logits.
            # Teacher predicts doc[1..doc_len] (shifted). Student predicts at prompt+doc positions.
            # We need to align: student valid positions = doc token predictions.
            # Teacher shifted positions = doc[1..doc_len-1] predictions.
            # The number of valid student positions should equal doc_len (doc tokens + EOS).
            # After shifting, valid student positions = doc_len - 1 + 1(EOS-1) ... let's just
            # match by taking the minimum overlap.

            valid_indices = valid_mask.view(-1).nonzero(as_tuple=True)[0]
            student_flat = student_shift.view(-1, student_shift.size(-1))[valid_indices]

            # Teacher: shifted logits predict doc[1..N]. Take the first len(valid_indices) positions.
            teacher_flat = teacher_shift.view(-1, teacher_shift.size(-1))
            num_overlap = min(student_flat.shape[0], teacher_flat.shape[0])
            if num_overlap == 0:
                input_offset += i_len
                continue

            student_flat = student_flat[:num_overlap]
            teacher_flat = teacher_flat[:num_overlap]

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
            total_kl = total_kl + kl
            num_valid_docs += 1

            input_offset += i_len

        return total_kl / max(num_valid_docs, 1)

    def _update_lr(self, step: int):
        """Update learning rate with per-phase warm restart (linear warmup + cosine decay)."""
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

    @torch.no_grad()
    def evaluate(self, compression_ratio: int) -> tuple[float, float]:
        """Evaluate on the eval set. Returns (avg CE loss, avg KL loss)."""
        self.qformer.eval()
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
            input_block_lengths = batch["input_block_lengths"]

            if not doc_token_ids or sum(doc_lengths) == 0:
                continue

            preamble_len = preamble_ids.shape[0]
            block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

            doc_hidden_states, teacher_logits_list = self._stage_a(
                doc_token_ids, doc_lengths, preamble_ids, block_lengths,
            )
            if self.training_config.offload_stage_a_to_cpu:
                doc_hidden_states = [
                    [hs.to(self.device) for hs in doc_hs]
                    for doc_hs in doc_hidden_states
                ]
                if teacher_logits_list is not None:
                    teacher_logits_list = [t.to(self.device) for t in teacher_logits_list]

            with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
                per_doc_compressed = []
                for doc_hs in doc_hidden_states:
                    compressed = self.qformer(doc_hs, compression_ratio, bypass=self.bypass)
                    per_doc_compressed.append(compressed)

                prefix_lengths = [
                    doc_cache[0][0].shape[2] for doc_cache in per_doc_compressed
                ]

                compressed_cache = concat_compressed_caches(
                    per_doc_compressed, self.model_config.num_layers
                )

                rotary_emb = self._get_rotary_emb()
                compressed_cache = apply_rope_to_cache_blocked(
                    compressed_cache, self.model_config.num_layers, rotary_emb,
                    prefix_lengths,
                )

                dtype = torch.float16 if self.training_config.fp16 else torch.float32
                attn_mask = build_block_prefix_causal_mask(
                    prefix_lengths, input_block_lengths,
                    dtype=dtype, device=self.device,
                )

                position_ids = build_blocked_position_ids(
                    prefix_lengths, input_block_lengths, device=self.device,
                )

                outputs = self.model(
                    input_ids=stage_b_input_ids,
                    past_key_values=compressed_cache,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )

                student_logits = outputs.logits
                shift_logits = student_logits[:, :-1, :].contiguous()
                shift_labels = stage_b_labels[:, 1:].contiguous()

                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                if self.training_config.ce_only_loss or teacher_logits_list is None:
                    kl_loss = torch.tensor(0.0)
                else:
                    kl_loss = self._compute_reconstruction_kl(
                        teacher_logits_list, student_logits, stage_b_labels,
                        input_block_lengths,
                    )

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
        """Save Q-Former checkpoint."""
        save_dir = os.path.join(self.training_config.output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)

        ckpt = {
            "step": step,
            "compression_ratio": compression_ratio,
            "qformer_state_dict": self.qformer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }

        torch.save(ckpt, os.path.join(save_dir, "checkpoint.pt"))
        logger.info(f"Checkpoint saved at {save_dir}")

    def verify_gradient_flow(self):
        """Verify Q-Former parameters receive gradients (call after first backward)."""
        has_grad = False
        zero_grad_params = []
        for name, param in self.qformer.named_parameters():
            if param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
                else:
                    zero_grad_params.append(name)
        if has_grad:
            logger.info("Gradient flow verified: Q-Former parameters have non-zero gradients.")
        else:
            logger.warning(
                "No gradients flowing to Q-Former! Check the computation graph."
            )
        if zero_grad_params:
            logger.warning(f"Parameters with zero gradients: {zero_grad_params[:5]}...")
