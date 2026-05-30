"""V2Trainer: single-forward training with the gated two-path attention.

One forward over [preamble|docs|Q+A]; per-layer learned (or oracle) keep-gate;
loss = CE on answers + budget loss + (optional) offline-teacher KL. DoRA + norms
+ selector are trained; the base LLM is frozen. Gradient checkpointing on.
"""

from __future__ import annotations

import logging
import math
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from leefrag_v2.config import SelectorConfig, V2ModelConfig, V2TrainingConfig
from leefrag_v2.data.adapter import build_blocks
from leefrag_v2.model.patch import new_context, sample_step_noise, set_context
from leefrag_v2.model.peft_setup import collect_param_groups
from leefrag_v2.training.budget import (
    KeepRateScheduler,
    budget_binomial_loss,
    budget_onesided_global,
)
from leefrag_v2.training.losses import ce_on_answer, kl_to_teacher

logger = logging.getLogger(__name__)


class V2Trainer:
    def __init__(
        self,
        model,
        selector: nn.Module | None,
        tokenizer,
        train_loader,
        eval_loader,
        model_config: V2ModelConfig,
        selector_config: SelectorConfig,
        training_config: V2TrainingConfig,
        device,
    ):
        self.model = model
        self.selector = selector
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.mc = model_config
        self.sc = selector_config
        self.cfg = training_config
        self.device = device
        self.num_layers = model_config.num_layers
        self.mode = training_config.mode

        # KL distillation is the primary loss; with ce_weight low, a missing
        # teacher dir would silently gut training. Fail loudly at init instead.
        if training_config.use_kl_teacher:
            td = training_config.teacher_dir
            has_teacher = os.path.isdir(td) and any(
                f.endswith(".pt") for f in os.listdir(td)
            )
            if not has_teacher:
                logger.warning(
                    "use_kl_teacher=True but no teacher logits (*.pt) found in %r. "
                    "KL will be skipped and training will run on ce_weight=%g CE "
                    "alone -- run scripts/precompute_teacher.py first.",
                    td, training_config.ce_weight,
                )

        steps_per_epoch = math.ceil(
            len(train_loader) / training_config.gradient_accumulation_steps
        )
        self.total_steps = max(1, steps_per_epoch * training_config.num_epochs)
        self.steps_per_epoch = steps_per_epoch

        self.keep_rate_scheduler = KeepRateScheduler(
            training_config.keep_rate_schedule, self.total_steps,
            mode=training_config.keep_rate_mode,
            pi_min=training_config.keep_rate_min,
            pi_max=training_config.keep_rate_max,
            cr_steps_per_unit=training_config.cr_steps_per_unit,
        )
        if training_config.eval_steps is None:
            # phased: ~4 evals per CR phase; continuous ramp: ~16 evals over the run.
            if self.keep_rate_scheduler.is_phased:
                training_config.eval_steps = max(1, self.keep_rate_scheduler.steps_per_phase // 4)
            else:
                training_config.eval_steps = max(1, self.total_steps // 16)

        groups = collect_param_groups(model, selector, training_config)
        for g in groups:
            g["initial_lr"] = g["lr"]
        self.optimizer = torch.optim.AdamW(
            groups, betas=(training_config.adam_beta1, training_config.adam_beta2)
        )
        # bf16 needs no loss scaling; only fp16 does. autocast runs the frozen
        # base in low precision while the fp32 trainable params accumulate grads
        # in fp32 (loader keeps them fp32). A disabled GradScaler is a pass-through.
        self.amp_dtype = torch.bfloat16 if training_config.bf16 else torch.float16
        self.use_amp = training_config.bf16 or training_config.fp16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=training_config.fp16 and not training_config.bf16
        )

        if training_config.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": training_config.checkpoint_use_reentrant
                }
            )

        self.use_wandb = training_config.use_wandb
        if self.use_wandb:
            import wandb

            wandb.init(
                project=training_config.wandb_project,
                config={
                    "model": model_config.__dict__,
                    "selector": selector_config.__dict__,
                    "training": training_config.__dict__,
                },
            )

    # ----------------------- schedules -----------------------
    def _tau(self, step: int) -> float:
        p = min(1.0, step / max(1, self.total_steps))
        return self.sc.tau_end + 0.5 * (self.sc.tau_start - self.sc.tau_end) * (
            1.0 + math.cos(math.pi * p)
        )

    def _update_lr(self, step: int) -> float:
        # Discrete phases warmup + cosine-restart at each CR jump; a continuous CR
        # ramp (DMS cr_linear / linear / sampled_range) gets a single warmup +
        # cosine decay over the whole run (no restarts -- there are no phases).
        if self.keep_rate_scheduler.is_phased:
            period = self.keep_rate_scheduler.steps_per_phase
            phase = self.keep_rate_scheduler.get_phase(step)
            local = step - phase * period
        else:
            period = self.total_steps
            local = step
        warmup = int(period * self.cfg.warmup_ratio)
        if local < warmup:
            scale = local / max(1, warmup)
        else:
            prog = (local - warmup) / max(1, period - warmup)
            scale = 0.5 * (1.0 + math.cos(math.pi * prog))
        for g in self.optimizer.param_groups:
            g["lr"] = g["initial_lr"] * scale
        return scale

    # ----------------------- teacher / oracle -----------------------
    def _load_teacher(self, example_idx: int):
        if not self.cfg.use_kl_teacher or example_idx < 0:
            return None
        path = os.path.join(self.cfg.teacher_dir, f"{example_idx}.pt")
        if not os.path.exists(path):
            return None
        d = torch.load(path, map_location=self.device)
        return d["vals"].to(self.device), d["idx"].to(self.device)

    def _oracle_masks(self, D: int, pi: float, example_idx: int) -> list:
        k = max(1, round(pi * D))
        masks = []
        for layer_idx in range(self.num_layers):
            g = torch.Generator().manual_seed(max(0, example_idx) * 1000 + layer_idx)
            perm = torch.randperm(D, generator=g)[:k]
            m = torch.zeros(D)
            m[perm] = 1.0
            masks.append(m.to(self.device))
        return masks

    # ----------------------- one step -----------------------
    def _forward_loss(self, blocks: dict, step: int):
        D = blocks["doc_total"]
        pi = self.keep_rate_scheduler.get_pi(step)
        tau = self._tau(step)

        if self.mode == "learned":
            num_heads = self.mc.num_kv_heads if self.sc.per_head else 1
            noise = sample_step_noise(
                self.num_layers, D, self.device, num_heads=num_heads
            )
            ctx = new_context(
                blocks["block_lengths"], blocks["qa_len"], self.device, self.num_layers,
                use_flex=self.cfg.use_flex, selector=self.selector, tau=tau,
                capture_for_budget=True, noise=noise, gate_renorm=self.sc.gate_renorm,
            )
        else:  # oracle
            ctx = new_context(
                blocks["block_lengths"], blocks["qa_len"], self.device, self.num_layers,
                use_flex=self.cfg.use_flex, selector=None,
                oracle_masks=self._oracle_masks(D, pi, blocks["example_idx"]),
                gate_renorm=self.sc.gate_renorm,
            )
        set_context(self.model, ctx)

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model(input_ids=blocks["input_ids"], use_cache=False)
            logits = out.logits
            ce = ce_on_answer(logits, blocks["labels"])
            loss = self.cfg.ce_weight * ce

        metrics = {"ce": ce.item(), "pi": pi, "tau": tau, "budget": 0.0, "keep": 0.0, "kl": 0.0}

        if self.mode == "learned":
            budget_fn = (
                budget_onesided_global
                if self.cfg.budget_mode == "onesided_global"
                else budget_binomial_loss
            )
            budget, keep = budget_fn(
                ctx.captured_chunk_hidden, self.selector, pi, tau, ctx.noise,
                self.device, hard=self.cfg.budget_hard_count,
            )
            loss = loss + self.cfg.budget_weight * budget
            metrics["budget"] = budget.item()
            metrics["keep"] = keep

        teacher = self._load_teacher(blocks["example_idx"])
        if teacher is not None:
            kl = kl_to_teacher(logits, blocks["labels"], teacher[0], teacher[1])
            loss = loss + self.cfg.kl_weight * kl
            metrics["kl"] = kl.item()

        set_context(self.model, None)
        return loss, metrics

    # ----------------------- train -----------------------
    def train(self):
        self.model.train()
        if self.selector is not None:
            self.selector.train()

        global_step = 0
        accum = 0
        for epoch in range(self.cfg.num_epochs):
            pbar = tqdm(self.train_loader, desc=f"epoch {epoch+1}/{self.cfg.num_epochs}")
            for batch in pbar:
                blocks = build_blocks(batch, self.device)
                if blocks is None:
                    continue

                loss, m = self._forward_loss(blocks, global_step)
                self.scaler.scale(loss / self.cfg.gradient_accumulation_steps).backward()
                accum += 1

                if accum % self.cfg.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    clip = [p for g in self.optimizer.param_groups for p in g["params"]]
                    torch.nn.utils.clip_grad_norm_(clip, self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    lr_scale = self._update_lr(global_step)
                    global_step += 1

                    pbar.set_postfix(
                        ce=f"{m['ce']:.3f}", budget=f"{m['budget']:.3f}",
                        keep=f"{m['keep']:.3f}", pi=f"{m['pi']:.3f}", tau=f"{m['tau']:.2f}",
                    )
                    if self.use_wandb and global_step % self.cfg.logging_steps == 0:
                        import wandb

                        wandb.log({
                            "train/ce": m["ce"], "train/budget": m["budget"],
                            "train/kl": m["kl"], "train/keep_rate": m["keep"],
                            "train/target_pi": m["pi"], "train/tau": m["tau"],
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                            "train/step": global_step,
                        })
                    if global_step % self.cfg.eval_steps == 0:
                        self._run_eval(global_step)
                    if global_step % self.cfg.save_steps == 0:
                        self.save_checkpoint(global_step)

        self.save_checkpoint(global_step)
        logger.info("Training complete.")

    def _run_eval(self, step: int):
        # Sweep a family of keep-rates from the one checkpoint (DMS-style).
        pis = self.cfg.eval_pis or [self.keep_rate_scheduler.get_pi(step)]
        logs = {"train/step": step}
        for pi in pis:
            ce = self.evaluate(pi)
            ppl = math.exp(min(ce, 20))
            tqdm.write(f"eval @ {step}: pi={pi:.3f} CE={ce:.4f} ppl={ppl:.2f}")
            logs[f"eval/ce@{pi:g}"] = ce
            logs[f"eval/ppl@{pi:g}"] = ppl
        if self.use_wandb:
            import wandb

            wandb.log(logs)
        self.model.train()
        if self.selector is not None:
            self.selector.train()

    @torch.no_grad()
    def evaluate(self, pi: float) -> float:
        self.model.eval()
        if self.selector is not None:
            self.selector.eval()
        total, n = 0.0, 0
        for batch in tqdm(self.eval_loader, desc="eval", leave=False):
            blocks = build_blocks(batch, self.device)
            if blocks is None:
                continue
            D = blocks["doc_total"]
            if self.mode == "learned":
                ctx = new_context(
                    blocks["block_lengths"], blocks["qa_len"], self.device, self.num_layers,
                    use_flex=self.cfg.use_flex, selector=self.selector, eval_pi=pi,
                    gate_renorm=self.sc.gate_renorm,
                )
            else:
                ctx = new_context(
                    blocks["block_lengths"], blocks["qa_len"], self.device, self.num_layers,
                    use_flex=self.cfg.use_flex, selector=None,
                    oracle_masks=self._oracle_masks(D, pi, blocks["example_idx"]),
                    gate_renorm=self.sc.gate_renorm,
                )
            set_context(self.model, ctx)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(input_ids=blocks["input_ids"], use_cache=False)
                ce = ce_on_answer(out.logits, blocks["labels"])
            set_context(self.model, None)
            total += ce.item()
            n += 1
        return total / max(1, n)

    def save_checkpoint(self, step: int):
        from peft import get_peft_model_state_dict

        save_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{step}")
        os.makedirs(save_dir, exist_ok=True)
        norm_sd = {
            n: p.detach().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad and ("layernorm" in n.lower() or n.endswith("norm.weight"))
        }
        ckpt = {
            "step": step,
            "lora_state_dict": get_peft_model_state_dict(self.model),
            "norm_state_dict": norm_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "mode": self.mode,
        }
        if self.selector is not None:
            ckpt["selector_state_dict"] = self.selector.state_dict()
        torch.save(ckpt, os.path.join(save_dir, "checkpoint.pt"))
        logger.info(f"Checkpoint saved at {save_dir}")
