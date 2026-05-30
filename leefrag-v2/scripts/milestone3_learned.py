"""Milestone 3: learned selector + offline-teacher KL distillation + budget loss.

Main training entry point. Trains DoRA + norms + the per-head selector through
the 2x->4x->8x keep-rate anneal. KL distillation (teacher = original full-KV
model) is the primary loss and on by default; pass --no_kl_teacher to disable.

  python scripts/precompute_teacher.py             # once: build the KL teacher
  python scripts/milestone3_learned.py --epochs 4  # KL on by default
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
    ap.add_argument("--keep_rate_schedule", type=float, nargs="+", default=[0.5, 0.25, 0.125])
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
        budget_weight=args.budget_weight, keep_rate_schedule=args.keep_rate_schedule,
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
