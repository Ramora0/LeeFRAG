import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from config import ModelConfig, QFormerConfig, TrainingConfig
from block_attention import build_block_causal_mask_with_qa
from kv_cache_utils import apply_rope_to_cache, concat_compressed_caches, extract_doc_hidden_states
from qformer import QFormerKVCompressor
from scheduler import CompressionScheduler

logger = logging.getLogger(__name__)


class TwoStageTrainer:
    """Two-stage training loop for KV cache compression.

    Stage A (no grad): Run frozen LLM on [docs | Q+A] with block+causal mask.
        → extract per-document hidden states + teacher logits over Q+A tokens.
    Stage B (grad): Q-Former compresses hidden states → KV prefix → LLM forward
        → CE loss on answer tokens + KL divergence against teacher logits.
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

        # Optimizer (only Q-Former parameters)
        self.optimizer = torch.optim.AdamW(
            qformer.parameters(),
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

        # Auto-compute eval_steps: 3 times per epoch if not explicitly set
        if training_config.eval_steps is None:
            training_config.eval_steps = max(1, steps_per_epoch // 3)

        # LR scheduler
        self.warmup_steps = int(self.total_steps * training_config.warmup_ratio)

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
            wandb.init(project="leefrag-kv-compression", config={
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

        # Accumulators for averaging metrics over gradient accumulation steps
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
            for batch_idx, batch in enumerate(pbar):
                compression_ratio = self.compression_scheduler.get_compression_ratio(
                    global_step
                )

                loss = self._training_step(batch, compression_ratio, global_step)

                if loss is None:
                    continue

                # Scale loss for gradient accumulation
                loss = loss / self.training_config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                accumulation_count += 1

                # Accumulate per-micro-batch metrics
                accum_loss += loss.item() * self.training_config.gradient_accumulation_steps
                accum_ce += self._last_ce_loss
                accum_kl += self._last_kl_loss
                accum_empirical_ratio += self._last_empirical_ratio
                accum_micro_batches += 1

                if accumulation_count % self.training_config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.qformer.parameters(),
                        self.training_config.max_grad_norm,
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # LR schedule with warmup
                    self._update_lr(global_step)

                    global_step += 1

                    # Average metrics over gradient accumulation micro-batches
                    avg_loss = accum_loss / accum_micro_batches
                    avg_ce = accum_ce / accum_micro_batches
                    avg_kl = accum_kl / accum_micro_batches
                    avg_empirical_ratio = accum_empirical_ratio / accum_micro_batches

                    # Reset accumulators
                    accum_loss = 0.0
                    accum_ce = 0.0
                    accum_kl = 0.0
                    accum_empirical_ratio = 0.0
                    accum_micro_batches = 0

                    # Update progress bar
                    lr = self.optimizer.param_groups[0]["lr"]
                    phase = self.compression_scheduler.get_phase(global_step)
                    use_hs = self.training_config.hidden_state_loss
                    secondary_tag = "hs" if use_hs else "kl"
                    postfix = dict(
                        loss=f"{avg_loss:.4f}",
                        ce=f"{avg_ce:.4f}",
                        **{secondary_tag: f"{avg_kl:.4f}"},
                        lr=f"{lr:.2e}",
                        comp=f"{compression_ratio}x",
                        step=f"{global_step}/{self.total_steps}",
                    )
                    pbar.set_postfix(**postfix)

                    if global_step % self.training_config.logging_steps == 0:
                        if self.use_wandb:
                            import wandb
                            secondary_key = (
                                "train/hidden_state_loss" if use_hs
                                else "train/kl_loss"
                            )
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/ce_loss": avg_ce,
                                secondary_key: avg_kl,
                                "train/lr": lr,
                                "train/compression_ratio": compression_ratio,
                                "train/empirical_compression_ratio": avg_empirical_ratio,
                                "train/phase": phase,
                                "train/global_step": global_step,
                            })

                    # Evaluation
                    if global_step % self.training_config.eval_steps == 0:
                        eval_ce, eval_secondary = self.evaluate(compression_ratio)
                        use_hs = self.training_config.hidden_state_loss
                        secondary_weight = (
                            self.training_config.hidden_state_weight if use_hs
                            else self.training_config.kl_weight
                        )
                        eval_total = eval_ce + secondary_weight * eval_secondary
                        secondary_name = "HS" if use_hs else "KL"
                        tqdm.write(
                            f"Eval @ step {global_step}: "
                            f"CE={eval_ce:.4f}, {secondary_name}={eval_secondary:.4f}, "
                            f"total={eval_total:.4f}, "
                            f"ppl={math.exp(eval_ce):.2f}, "
                            f"compression={compression_ratio}x"
                        )
                        if self.use_wandb:
                            import wandb
                            log_dict = {
                                "eval/ce_loss": eval_ce,
                                "eval/total_loss": eval_total,
                                "eval/perplexity": math.exp(eval_ce),
                                "eval/compression_ratio": compression_ratio,
                            }
                            if use_hs:
                                log_dict["eval/hidden_state_loss"] = eval_secondary
                            else:
                                log_dict["eval/kl_loss"] = eval_secondary
                            wandb.log(log_dict)
                        self.qformer.train()

                    # Save checkpoint
                    if global_step % self.training_config.save_steps == 0:
                        self._save_checkpoint(global_step, compression_ratio)

        # Final save
        self._save_checkpoint(global_step, compression_ratio)
        logger.info("Training complete.")

    def _training_step(
        self, batch: dict, compression_ratio: int, global_step: int = 0,
    ) -> torch.Tensor | None:
        """Execute one training step (Stage A + Stage B).

        Returns the combined loss tensor or None if the batch should be skipped.
        """
        doc_token_ids = batch["doc_token_ids"]  # list of tensors
        doc_lengths = batch["doc_lengths"]
        preamble_ids = batch["preamble_ids"]  # tensor
        stage_b_input_ids = batch["stage_b_input_ids"].to(self.device)
        stage_b_labels = batch["stage_b_labels"].to(self.device)

        if not doc_token_ids or sum(doc_lengths) == 0:
            return None

        # Block lengths: preamble merges with first doc, rest are standalone
        preamble_len = preamble_ids.shape[0]
        block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

        # === Stage A: Run frozen LLM on [preamble+docs | Q+A], extract hidden states + teacher logits ===
        with torch.no_grad():
            doc_hidden_states, teacher_logits, teacher_qa_hidden = self._stage_a(
                doc_token_ids, doc_lengths, stage_b_input_ids,
                preamble_ids, block_lengths,
            )

        if self.training_config.offload_stage_a_to_cpu:
            doc_hidden_states = [
                [hs.to(self.device) for hs in doc_hs]
                for doc_hs in doc_hidden_states
            ]
            teacher_logits = teacher_logits.to(self.device)
            if teacher_qa_hidden is not None:
                teacher_qa_hidden = [h.to(self.device) for h in teacher_qa_hidden]

        # === Stage B: Compress hidden states and compute loss ===
        with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
            loss = self._stage_b(
                doc_hidden_states, compression_ratio,
                stage_b_input_ids, stage_b_labels, teacher_logits,
                teacher_qa_hidden,
            )

        return loss

    def _get_hidden_state_layer_indices(self) -> list[int]:
        """Return which layer indices to use for hidden state matching.

        Parses hidden_state_layers config: "all" or "last_N" (e.g. "last_8").
        Returns indices into hidden_states tuple (1-indexed, skipping embedding).
        """
        spec = self.training_config.hidden_state_layers
        n = self.model_config.num_layers
        if spec == "all":
            return list(range(1, n + 1))
        elif spec.startswith("last_"):
            k = int(spec.split("_")[1])
            return list(range(n - k + 1, n + 1))
        else:
            raise ValueError(f"Unknown hidden_state_layers spec: {spec}")

    def _stage_a(
        self,
        doc_token_ids: list[torch.Tensor],
        doc_lengths: list[int],
        qa_input_ids: torch.Tensor,
        preamble_ids: torch.Tensor,
        block_lengths: list[int],
    ) -> tuple[list[list[torch.Tensor]], torch.Tensor, list[torch.Tensor] | None]:
        """Stage A: Run frozen LLM on [preamble+docs | Q+A] in one forward pass.

        Uses block-diagonal causal mask for blocks (preamble+doc0, doc1, ...),
        with Q+A attending to all blocks.

        Returns:
            per_doc_hidden: Per-block hidden states for Q-Former compression.
            teacher_logits: Logits over Q+A tokens [batch, qa_len, vocab_size].
            teacher_qa_hidden: Q+A hidden states per layer (if hidden_state_loss),
                else None. Each shape: [batch, qa_len, hidden_size].
        """
        # Concatenate: [preamble | doc0 | doc1 | ... | Q+A]
        doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(self.device)
        preamble = preamble_ids.unsqueeze(0).to(self.device)
        full_input = torch.cat([preamble, doc_concat, qa_input_ids], dim=1)

        qa_length = qa_input_ids.shape[1]

        # Build attention mask: block-diagonal for blocks, Q+A attends to all blocks + causal
        dtype = torch.float16 if self.training_config.fp16 else torch.float32
        attn_mask = build_block_causal_mask_with_qa(
            block_lengths, qa_length, dtype=dtype, device=self.device
        )

        # Forward through frozen LLM
        outputs = self.model(
            input_ids=full_input,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract per-block hidden states (preamble+doc0 is block 0, doc1 is block 1, etc.)
        per_doc_hidden = extract_doc_hidden_states(
            outputs.hidden_states, block_lengths, self.model_config.num_layers
        )

        # Extract teacher logits/hidden states (skip if CE-only to save memory)
        teacher_logits = None
        teacher_qa_hidden = None

        if not self.training_config.ce_only_loss:
            doc_total = sum(block_lengths)
            teacher_logits = outputs.logits[:, doc_total:, :]

            if self.training_config.hidden_state_loss:
                layer_indices = self._get_hidden_state_layer_indices()
                teacher_qa_hidden = [
                    outputs.hidden_states[i][:, doc_total:, :] for i in layer_indices
                ]

        if self.training_config.offload_stage_a_to_cpu:
            per_doc_hidden = [
                [hs.cpu() for hs in doc_hs]
                for doc_hs in per_doc_hidden
            ]
            if teacher_logits is not None:
                teacher_logits = teacher_logits.cpu()
            if teacher_qa_hidden is not None:
                teacher_qa_hidden = [h.cpu() for h in teacher_qa_hidden]

        return per_doc_hidden, teacher_logits, teacher_qa_hidden

    def _stage_b(
        self,
        doc_hidden_states: list[list[torch.Tensor]],
        compression_ratio: int,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_qa_hidden: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Stage B: Compress hidden states, run LLM, compute loss.

        Loss = CE + (hidden_state_matching OR KL), controlled by config.
        Gradients flow through the compressed KV cache back to Q-Former.
        """
        # Compress each document's hidden states into KV caches
        per_doc_compressed = []
        empirical_ratios = []
        for doc_hs in doc_hidden_states:
            compressed = self.qformer(doc_hs, compression_ratio)
            per_doc_compressed.append(compressed)
            doc_len = doc_hs[0].shape[1]
            num_queries = compressed[0][0].shape[2]  # [1, num_kv_heads, num_queries, head_dim]
            empirical_ratios.append(doc_len / num_queries)

        # Concatenate compressed caches from all documents
        compressed_cache = concat_compressed_caches(
            per_doc_compressed, self.model_config.num_layers
        )

        # Apply RoPE to compressed K values at prefix positions [0, prefix_len)
        # Q-Former outputs raw K without RoPE; the LLM expects cached K to be
        # pre-rotated (as it would be during normal autoregressive generation).
        rotary_emb = self.model.model.rotary_emb
        compressed_cache = apply_rope_to_cache(
            compressed_cache, self.model_config.num_layers, rotary_emb
        )

        # Forward through frozen LLM with compressed KV prefix
        # No explicit attention_mask: HF generates the correct prefix-causal
        # pattern from past_key_values (full attn to prefix + causal among new
        # tokens). Omitting it lets SDPA use the FlashAttention backend.
        use_hs_loss = self.training_config.hidden_state_loss
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=compressed_cache,
            use_cache=False,
            output_hidden_states=use_hs_loss,
        )

        student_logits = outputs.logits  # [batch, seq_len, vocab_size]

        # === CE loss on answer tokens ===
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if self.training_config.ce_only_loss:
            total_loss = ce_loss
            self._last_kl_loss = 0.0
        elif use_hs_loss and teacher_qa_hidden is not None:
            # === Hidden state matching: teacher (full context) vs student (compressed) ===
            hs_loss = self._compute_hidden_state_loss(
                teacher_qa_hidden, outputs.hidden_states
            )
            total_loss = ce_loss + self.training_config.hidden_state_weight * hs_loss
            self._last_kl_loss = hs_loss.item()
        else:
            # === KL divergence: teacher (full context) vs student (compressed) ===
            kl_loss = self._compute_kl_loss(teacher_logits, student_logits, labels)
            total_loss = ce_loss + self.training_config.kl_weight * kl_loss
            self._last_kl_loss = kl_loss.item()

        # Stash components for logging (detached)
        self._last_ce_loss = ce_loss.item()
        self._last_empirical_ratio = sum(empirical_ratios) / len(empirical_ratios)

        return total_loss

    def _compute_hidden_state_loss(
        self,
        teacher_qa_hidden: list[torch.Tensor],
        student_hidden_states: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Compute MSE between teacher and student hidden states at Q+A positions.

        Args:
            teacher_qa_hidden: List of teacher hidden states at Q+A positions,
                one per matched layer. Each [batch, qa_len, hidden_size].
            student_hidden_states: Full tuple of student hidden states from
                model output (including embedding at index 0).

        Returns:
            Scalar MSE loss averaged over matched layers.
        """
        layer_indices = self._get_hidden_state_layer_indices()
        total_loss = torch.tensor(0.0, device=teacher_qa_hidden[0].device)

        for i, layer_idx in enumerate(layer_indices):
            teacher_hs = teacher_qa_hidden[i]
            student_hs = student_hidden_states[layer_idx]  # [batch, qa_len, hidden_size]
            total_loss = total_loss + F.mse_loss(student_hs, teacher_hs)

        return total_loss / len(layer_indices)

    def _compute_kl_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between teacher and student logits.

        Only computed over positions where labels != -100 (answer tokens).

        Args:
            teacher_logits: [batch, seq_len, vocab_size] from full-context forward.
            student_logits: [batch, seq_len, vocab_size] from compressed forward.
            labels: [batch, seq_len] with -100 for non-answer positions.

        Returns:
            Scalar KL divergence loss.
        """
        # Shift to align with next-token prediction
        teacher_shift = teacher_logits[:, :-1, :].contiguous()
        student_shift = student_logits[:, :-1, :].contiguous()
        label_shift = labels[:, 1:].contiguous()

        # Mask: only compute KL on answer tokens
        valid_mask = label_shift != -100  # [batch, seq_len-1]
        if not valid_mask.any():
            return torch.tensor(0.0, device=student_logits.device)

        # Flatten to [num_valid_tokens, vocab_size]
        valid_indices = valid_mask.view(-1).nonzero(as_tuple=True)[0]
        teacher_flat = teacher_shift.view(-1, teacher_shift.size(-1))[valid_indices]
        student_flat = student_shift.view(-1, student_shift.size(-1))[valid_indices]

        if self.training_config.kl_top_k > 0:
            # Compute KL only over top-k teacher logits to save memory
            k = self.training_config.kl_top_k
            top_k_vals, top_k_idx = teacher_flat.topk(k, dim=-1)
            student_gathered = student_flat.gather(-1, top_k_idx)

            teacher_log_probs = F.log_softmax(top_k_vals, dim=-1)
            student_log_probs = F.log_softmax(student_gathered, dim=-1)
        else:
            teacher_log_probs = F.log_softmax(teacher_flat, dim=-1)
            student_log_probs = F.log_softmax(student_flat, dim=-1)

        # KL(teacher || student) = sum(teacher * (log_teacher - log_student))
        kl = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            log_target=True,
            reduction="batchmean",
        )

        return kl

    def _update_lr(self, step: int):
        """Update learning rate with linear warmup + cosine decay."""
        if step < self.warmup_steps:
            lr_scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
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

            if not doc_token_ids or sum(doc_lengths) == 0:
                continue

            # Block lengths: preamble merges with first doc
            preamble_len = preamble_ids.shape[0]
            block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

            # Stage A
            doc_hidden_states, teacher_logits, teacher_qa_hidden = self._stage_a(
                doc_token_ids, doc_lengths, stage_b_input_ids,
                preamble_ids, block_lengths,
            )
            if self.training_config.offload_stage_a_to_cpu:
                doc_hidden_states = [
                    [hs.to(self.device) for hs in doc_hs]
                    for doc_hs in doc_hidden_states
                ]
                teacher_logits = teacher_logits.to(self.device)
                if teacher_qa_hidden is not None:
                    teacher_qa_hidden = [h.to(self.device) for h in teacher_qa_hidden]

            with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
                per_doc_compressed = []
                for doc_hs in doc_hidden_states:
                    compressed = self.qformer(doc_hs, compression_ratio)
                    per_doc_compressed.append(compressed)

                compressed_cache = concat_compressed_caches(
                    per_doc_compressed, self.model_config.num_layers
                )

                # Apply RoPE to compressed K values at prefix positions
                rotary_emb = self.model.model.rotary_emb
                compressed_cache = apply_rope_to_cache(
                    compressed_cache, self.model_config.num_layers, rotary_emb
                )

                use_hs_loss = self.training_config.hidden_state_loss
                outputs = self.model(
                    input_ids=stage_b_input_ids,
                    past_key_values=compressed_cache,
                    use_cache=False,
                    output_hidden_states=use_hs_loss,
                )

                student_logits = outputs.logits
                shift_logits = student_logits[:, :-1, :].contiguous()
                shift_labels = stage_b_labels[:, 1:].contiguous()

                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                if self.training_config.ce_only_loss:
                    secondary_loss = torch.tensor(0.0)
                elif use_hs_loss and teacher_qa_hidden is not None:
                    secondary_loss = self._compute_hidden_state_loss(
                        teacher_qa_hidden, outputs.hidden_states
                    )
                else:
                    secondary_loss = self._compute_kl_loss(
                        teacher_logits, student_logits, stage_b_labels
                    )

            total_ce += ce_loss.item()
            total_kl += secondary_loss.item()
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

        torch.save(
            {
                "step": step,
                "compression_ratio": compression_ratio,
                "qformer_state_dict": self.qformer.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            os.path.join(save_dir, "checkpoint.pt"),
        )
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
