"""Milestone 3: learned selector + offline-teacher KL distillation + budget loss.

Main training entry point. Trains DoRA + norms + the per-head selector through
DMS's compression-ratio ramp (cr_linear: warm-start at CR=1, then ramp linearly
to 1/keep_rate_min at cr_steps_per_unit steps/CR). KL distillation (teacher =
original full-KV model) is the primary loss and on by default; pass
--no_kl_teacher to disable.

  python scripts/precompute_teacher.py             # once: build the KL teacher
  python scripts/milestone3_learned.py --epochs 4  # DMS ramp + KL, both default
  # short test run: auto-span the ramp so it still reaches the target CR
  python scripts/milestone3_learned.py --epochs 1 --cr_auto_span
"""

import _bootstrap  # noqa: F401

import argparse
import logging

import torch

from leefrag_v2.config import SelectorConfig, V2ModelConfig, V2TrainingConfig
from leefrag_v2.data.adapter import make_dataloaders
from leefrag_v2.loader import build_training_model
from leefrag_v2.training.trainer import V2Trainer

logging.basicConfig(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--dataset", default="rag_v1")
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--selector_lr", type=float, default=3e-4)
    ap.add_argument("--budget_weight", type=float, default=0.1)
    # DMS cr_linear ramp: CR goes 1/keep_rate_max -> 1/keep_rate_min.
    ap.add_argument("--keep_rate_max", type=float, default=1.0,
                    help="start keep-rate (1.0 = warm-start at CR 1, keep everything)")
    ap.add_argument("--keep_rate_min", type=float, default=0.125,
                    help="target keep-rate (0.125 = ramp to CR 8)")
    ap.add_argument("--cr_steps_per_unit", type=float, default=100.0,
                    help="DMS ramp rate: optimiser steps per unit of CR (DMS uses 100)")
    ap.add_argument("--cr_auto_span", action="store_true",
                    help="auto-stretch the ramp across the whole run (ignores "
                         "--cr_steps_per_unit); use for short test runs")
    ap.add_argument("--no_kl_teacher", dest="use_kl_teacher", action="store_false",
                    help="disable KL distillation (on by default)")
    ap.add_argument("--teacher_dir", default="outputs_v2/teacher")
    ap.add_argument("--kl_weight", type=float, default=1.0)
    ap.add_argument("--output_dir", default="outputs_v2/learned")
    ap.add_argument("--no_flex", action="store_true")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    cfg = V2TrainingConfig(
        mode="learned", num_epochs=args.epochs, dataset_name=args.dataset,
        learning_rate=args.learning_rate, selector_lr=args.selector_lr,
        budget_weight=args.budget_weight,
        keep_rate_mode="cr_linear",
        keep_rate_max=args.keep_rate_max, keep_rate_min=args.keep_rate_min,
        cr_steps_per_unit=(None if args.cr_auto_span else args.cr_steps_per_unit),
        use_kl_teacher=args.use_kl_teacher, teacher_dir=args.teacher_dir,
        kl_weight=args.kl_weight, output_dir=args.output_dir,
        use_flex=not args.no_flex, use_wandb=not args.no_wandb,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc, sc = V2ModelConfig(), SelectorConfig()
    model, selector, tokenizer = build_training_model(mc, sc, cfg, device)
    train_loader, eval_loader = make_dataloaders(tokenizer, mc, cfg)
    V2Trainer(model, selector, tokenizer, train_loader, eval_loader, mc, sc, cfg, device).train()


if __name__ == "__main__":
    main()
