"""DoRA setup, RMSNorm unfreezing, and optimizer parameter groups.

Adaptation = DoRA on all linears (heavy rank on the read path q/o/FFN, light rank
on the cache path k/v) + trainable RMSNorms + the selector. Base weights frozen.
"""

from __future__ import annotations

import torch.nn as nn

from leefrag_v2.config import V2TrainingConfig


def make_dora_config(cfg: V2TrainingConfig):
    from peft import LoraConfig

    targets = list(cfg.dora_targets_heavy) + list(cfg.dora_targets_light)
    rank_pattern = {m: cfg.dora_rank_light for m in cfg.dora_targets_light}
    alpha_pattern = {m: cfg.dora_alpha_light for m in cfg.dora_targets_light}
    return LoraConfig(
        r=cfg.dora_rank_heavy,
        lora_alpha=cfg.dora_alpha_heavy,
        lora_dropout=cfg.dora_dropout,
        target_modules=targets,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        use_dora=True,
        bias="none",
        task_type="CAUSAL_LM",
    )


def apply_dora(model, cfg: V2TrainingConfig):
    from peft import get_peft_model

    return get_peft_model(model, make_dora_config(cfg))


def _is_norm_param(name: str) -> bool:
    n = name.lower()
    return ("layernorm" in n) or name.endswith("norm.weight")


def unfreeze_norms(model) -> list:
    """Set requires_grad on all RMSNorm weights; return the param list."""
    params = []
    for name, p in model.named_parameters():
        if _is_norm_param(name):
            p.requires_grad_(True)
            params.append(p)
    return params


def collect_param_groups(model, selector: nn.Module, cfg: V2TrainingConfig) -> list[dict]:
    """AdamW param groups: DoRA (lr), norms (norm_lr, no decay), selector (selector_lr)."""
    lora_params, norm_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        elif _is_norm_param(name):
            norm_params.append(p)

    groups = []
    if lora_params:
        groups.append(
            {"params": lora_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay}
        )
    if cfg.train_norms and norm_params:
        groups.append({"params": norm_params, "lr": cfg.norm_lr, "weight_decay": 0.0})
    sel_params = [p for p in selector.parameters() if p.requires_grad]
    if sel_params:
        groups.append({"params": sel_params, "lr": cfg.selector_lr, "weight_decay": 0.0})
    return groups
