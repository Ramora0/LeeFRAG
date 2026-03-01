import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from config import ModelConfig, QFormerConfig, TrainingConfig
from block_attention import (
    build_block_causal_mask_with_qa,
    build_prefix_causal_mask,
)
from kv_cache_utils import concat_compressed_caches, extract_doc_hidden_states
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
            wandb.init(project="leefrag", config={
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

        for epoch in range(self.training_config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(self.train_loader):
                compression_ratio = self.compression_scheduler.get_compression_ratio(
                    global_step
                )

                loss = self._training_step(batch, compression_ratio)

                if loss is None:
                    continue

                # Scale loss for gradient accumulation
                loss = loss / self.training_config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                accumulation_count += 1
                epoch_loss += loss.item() * self.training_config.gradient_accumulation_steps

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
                    epoch_steps += 1

                    # Logging
                    if global_step % self.training_config.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.optimizer.param_groups[0]["lr"]
                        phase = self.compression_scheduler.get_phase(global_step)
                        logger.info(
                            f"Step {global_step}/{self.total_steps} | "
                            f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | Compression: {compression_ratio}x | "
                            f"Phase: {phase+1}/{len(self.training_config.compression_schedule)}"
                        )
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/lr": lr,
                                "train/compression_ratio": compression_ratio,
                                "train/phase": phase,
                                "train/global_step": global_step,
                            })

                    # Evaluation
                    if global_step % self.training_config.eval_steps == 0:
                        eval_loss = self.evaluate(compression_ratio)
                        logger.info(
                            f"Eval @ step {global_step}: loss={eval_loss:.4f}, "
                            f"ppl={math.exp(eval_loss):.2f}, "
                            f"compression={compression_ratio}x"
                        )
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "eval/loss": eval_loss,
                                "eval/perplexity": math.exp(eval_loss),
                                "eval/compression_ratio": compression_ratio,
                            })
                        self.qformer.train()

                    # Save checkpoint
                    if global_step % self.training_config.save_steps == 0:
                        self._save_checkpoint(global_step, compression_ratio)

            logger.info(f"Epoch {epoch+1} completed. Avg loss: {epoch_loss / max(epoch_steps, 1):.4f}")

        # Final save
        self._save_checkpoint(global_step, compression_ratio)
        logger.info("Training complete.")

    def _training_step(
        self, batch: dict, compression_ratio: int
    ) -> torch.Tensor | None:
        """Execute one training step (Stage A + Stage B).

        Returns the combined loss tensor or None if the batch should be skipped.
        """
        doc_token_ids = batch["doc_token_ids"]  # list of tensors
        doc_lengths = batch["doc_lengths"]
        stage_b_input_ids = batch["stage_b_input_ids"].to(self.device)
        stage_b_labels = batch["stage_b_labels"].to(self.device)

        if not doc_token_ids or sum(doc_lengths) == 0:
            return None

        # === Stage A: Run frozen LLM on [docs | Q+A], extract hidden states + teacher logits ===
        with torch.no_grad():
            doc_hidden_states, teacher_logits = self._stage_a(
                doc_token_ids, doc_lengths, stage_b_input_ids
            )

        if self.training_config.offload_stage_a_to_cpu:
            doc_hidden_states = [
                [hs.to(self.device) for hs in doc_hs]
                for doc_hs in doc_hidden_states
            ]
            teacher_logits = teacher_logits.to(self.device)

        # === Stage B: Compress hidden states and compute loss ===
        with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
            loss = self._stage_b(
                doc_hidden_states, compression_ratio,
                stage_b_input_ids, stage_b_labels, teacher_logits,
            )

        return loss

    def _stage_a(
        self,
        doc_token_ids: list[torch.Tensor],
        doc_lengths: list[int],
        qa_input_ids: torch.Tensor,
    ) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
        """Stage A: Run frozen LLM on [docs | Q+A] in one forward pass.

        Uses block-diagonal causal mask for docs, with Q+A attending to all docs.

        Returns:
            per_doc_hidden: Per-document hidden states for Q-Former compression.
            teacher_logits: Logits over Q+A tokens [batch, qa_len, vocab_size].
        """
        # Concatenate: [doc0 | doc1 | ... | Q+A]
        doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(self.device)
        full_input = torch.cat([doc_concat, qa_input_ids], dim=1)

        qa_length = qa_input_ids.shape[1]

        # Build attention mask: block-diagonal for docs, Q+A attends to all docs + causal
        dtype = torch.float16 if self.training_config.fp16 else torch.float32
        attn_mask = build_block_causal_mask_with_qa(
            doc_lengths, qa_length, dtype=dtype, device=self.device
        )

        # Forward through frozen LLM
        outputs = self.model(
            input_ids=full_input,
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Extract per-document hidden states (from doc portion only)
        per_doc_hidden = extract_doc_hidden_states(
            outputs.hidden_states, doc_lengths, self.model_config.num_layers
        )

        # Extract teacher logits over Q+A portion
        doc_total = sum(doc_lengths)
        teacher_logits = outputs.logits[:, doc_total:, :]

        if self.training_config.offload_stage_a_to_cpu:
            per_doc_hidden = [
                [hs.cpu() for hs in doc_hs]
                for doc_hs in per_doc_hidden
            ]
            teacher_logits = teacher_logits.cpu()

        return per_doc_hidden, teacher_logits

    def _stage_b(
        self,
        doc_hidden_states: list[list[torch.Tensor]],
        compression_ratio: int,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Stage B: Compress hidden states, run LLM, compute CE + KL loss.

        Gradients flow through the compressed KV cache back to Q-Former.
        """
        # Compress each document's hidden states into KV caches
        per_doc_compressed = []
        for doc_hs in doc_hidden_states:
            compressed = self.qformer(doc_hs, compression_ratio)
            per_doc_compressed.append(compressed)

        # Concatenate compressed caches from all documents
        compressed_cache = concat_compressed_caches(
            per_doc_compressed, self.model_config.num_layers
        )

        # Build attention mask for prefix + sequence
        prefix_length = compressed_cache.get_seq_length()
        seq_length = input_ids.shape[1]
        dtype = torch.float16 if self.training_config.fp16 else torch.float32
        attn_mask = build_prefix_causal_mask(
            prefix_length, seq_length, dtype=dtype, device=self.device
        )

        # Forward through frozen LLM with compressed KV prefix
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=compressed_cache,
            use_cache=False,
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

        # === KL divergence: teacher (full context) vs student (compressed) ===
        kl_loss = self._compute_kl_loss(teacher_logits, student_logits, labels)

        total_loss = ce_loss + self.training_config.kl_weight * kl_loss
        return total_loss

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
    def evaluate(self, compression_ratio: int) -> float:
        """Evaluate on the eval set. Returns average CE loss."""
        self.qformer.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_loader:
            doc_token_ids = batch["doc_token_ids"]
            doc_lengths = batch["doc_lengths"]
            stage_b_input_ids = batch["stage_b_input_ids"].to(self.device)
            stage_b_labels = batch["stage_b_labels"].to(self.device)

            if not doc_token_ids or sum(doc_lengths) == 0:
                continue

            # Stage A
            doc_hidden_states, _ = self._stage_a(
                doc_token_ids, doc_lengths, stage_b_input_ids
            )
            if self.training_config.offload_stage_a_to_cpu:
                doc_hidden_states = [
                    [hs.to(self.device) for hs in doc_hs]
                    for doc_hs in doc_hidden_states
                ]

            # Stage B (no grad, CE only for eval metric)
            with torch.amp.autocast("cuda", enabled=self.training_config.fp16):
                per_doc_compressed = []
                for doc_hs in doc_hidden_states:
                    compressed = self.qformer(doc_hs, compression_ratio)
                    per_doc_compressed.append(compressed)

                compressed_cache = concat_compressed_caches(
                    per_doc_compressed, self.model_config.num_layers
                )

                prefix_length = compressed_cache.get_seq_length()
                seq_length = stage_b_input_ids.shape[1]
                dtype = torch.float16 if self.training_config.fp16 else torch.float32
                attn_mask = build_prefix_causal_mask(
                    prefix_length, seq_length, dtype=dtype, device=self.device
                )

                outputs = self.model(
                    input_ids=stage_b_input_ids,
                    attention_mask=attn_mask,
                    past_key_values=compressed_cache,
                    use_cache=False,
                )

                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = stage_b_labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

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
