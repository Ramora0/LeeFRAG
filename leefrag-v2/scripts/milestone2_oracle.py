"""Milestone 2: oracle keep-set + DoRA/norm finetune.

Freezes selection to a fixed random keep-set at rate pi and finetunes the LLM
(DoRA + norms) to answer from it. Establishes the achievable quality ceiling at
each compression before the learned selector is introduced.

  python scripts/milestone2_oracle.py --keep_rate 0.25 --epochs 1
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
    ap.add_argument("--keep_rate", type=float, default=None,
                    help="constant pi (overrides the 0.5/0.25/0.125 schedule)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dataset", default="rag_v1")
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--output_dir", default="outputs_v2/oracle")
    ap.add_argument("--no_flex", action="store_true")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    cfg = V2TrainingConfig(
        mode="oracle", use_kl_teacher=False, num_epochs=args.epochs,
        dataset_name=args.dataset, learning_rate=args.learning_rate,
        output_dir=args.output_dir, use_flex=not args.no_flex, use_wandb=not args.no_wandb,
    )
    if args.keep_rate is not None:
        cfg.keep_rate_schedule = [args.keep_rate]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc, sc = V2ModelConfig(), SelectorConfig()
    model, selector, tokenizer = build_training_model(mc, sc, cfg, device)
    train_loader, eval_loader = make_dataloaders(tokenizer, mc, cfg)
    V2Trainer(model, selector, tokenizer, train_loader, eval_loader, mc, sc, cfg, device).train()


if __name__ == "__main__":
    main()
