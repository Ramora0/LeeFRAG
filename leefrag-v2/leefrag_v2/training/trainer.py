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
from leefrag_v2.training.budget import KeepRateScheduler, budget_binomial_loss
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

        steps_per_epoch = math.ceil(
            len(train_loader) / training_config.gradient_accumulation_steps
        )
        self.total_steps = max(1, steps_per_epoch * training_config.num_epochs)
        self.steps_per_epoch = steps_per_epoch

        self.keep_rate_scheduler = KeepRateScheduler(
            training_config.keep_rate_schedule, self.total_steps
        )
        if training_config.eval_steps is None:
            training_config.eval_steps = max(1, self.keep_rate_scheduler.steps_per_phase // 4)

        groups = collect_param_groups(model, selector, training_config)
        for g in groups:
            g["initial_lr"] = g["lr"]
        self.optimizer = torch.optim.AdamW(
            groups, betas=(training_config.adam_beta1, training_config.adam_beta2)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=training_config.fp16)

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
        spp = self.keep_rate_scheduler.steps_per_phase
        phase = self.keep_rate_scheduler.get_phase(step)
        phase_step = step - phase * spp
        warmup = int(spp * self.cfg.warmup_ratio)
        if phase_step < warmup:
            scale = phase_step / max(1, warmup)
        else:
            prog = (phase_step - warmup) / max(1, spp - warmup)
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
            noise = sample_step_noise(self.num_layers, D, self.device)
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

        with torch.amp.autocast("cuda", enabled=self.cfg.fp16):
            out = self.model(input_ids=blocks["input_ids"], use_cache=False)
            logits = out.logits
            ce = ce_on_answer(logits, blocks["labels"])
            loss = self.cfg.ce_weight * ce

        metrics = {"ce": ce.item(), "pi": pi, "tau": tau, "budget": 0.0, "keep": 0.0, "kl": 0.0}

        if self.mode == "learned":
            budget, keep = budget_binomial_loss(
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
        pi = self.keep_rate_scheduler.get_pi(step)
        ce = self.evaluate(pi)
        tqdm.write(f"eval @ {step}: CE={ce:.4f} ppl={math.exp(min(ce,20)):.2f} pi={pi:.3f}")
        if self.use_wandb:
            import wandb

            wandb.log({"eval/ce": ce, "eval/ppl": math.exp(min(ce, 20)), "eval/pi": pi, "train/step": step})
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
            with torch.amp.autocast("cuda", enabled=self.cfg.fp16):
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
