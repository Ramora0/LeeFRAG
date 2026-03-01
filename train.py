"""Main training entry point for KV cache compression with Q-Former."""

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, QFormerConfig, TrainingConfig
from collator import RAGCollator
from dataset import RAGDataset
from qformer import QFormerKVCompressor
from trainer import TwoStageTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Former KV cache compressor")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=None, help="Eval every N steps (default: 3x per epoch)")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--gradient_checkpoint_llm", action="store_true")
    parser.add_argument("--offload_stage_a_to_cpu", action="store_true")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--kl_top_k", type=int, default=0, help="Top-k logits for KL (0=full vocab)")
    parser.add_argument("--cross_attn_mode", type=str, default="global", choices=["global", "windowed"])
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    model_config = ModelConfig()
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        gradient_checkpoint_llm=args.gradient_checkpoint_llm,
        offload_stage_a_to_cpu=args.offload_stage_a_to_cpu,
        kl_weight=args.kl_weight,
        kl_top_k=args.kl_top_k,
    )

    qformer_config = QFormerConfig(cross_attn_mode=args.cross_attn_mode)

    set_seed(training_config.seed)
    os.makedirs(training_config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load frozen LLM
    logger.info(f"Loading frozen LLM: {model_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if training_config.gradient_checkpoint_llm:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for frozen LLM (Stage B)")

    # Build Q-Former
    logger.info("Building Q-Former KV compressor")
    qformer = QFormerKVCompressor(qformer_config, model_config).to(device)
    qformer.trunk = torch.compile(qformer.trunk, dynamic=True)
    trainable_params = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    logger.info(f"Q-Former trainable params: {trainable_params / 1e6:.1f}M")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = RAGDataset(
        tokenizer=tokenizer,
        model_config=model_config,
        split="train",
        eval_split_ratio=training_config.eval_split_ratio,
        seed=training_config.seed,
    )
    eval_dataset = RAGDataset(
        tokenizer=tokenizer,
        model_config=model_config,
        split="eval",
        eval_split_ratio=training_config.eval_split_ratio,
        seed=training_config.seed,
    )
    logger.info(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    collator = RAGCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.dataloader_num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.dataloader_num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Build trainer
    trainer = TwoStageTrainer(
        model=model,
        qformer=qformer,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )

    # Resume from checkpoint
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        qformer.load_state_dict(ckpt["qformer_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Resumed from step {ckpt.get('step', '?')}")

    # Verify gradient flow on first step
    logger.info("Running gradient flow verification...")
    sample_batch = next(iter(train_loader))
    loss = trainer._training_step(sample_batch, compression_ratio=2)
    if loss is not None:
        loss.backward()
        trainer.verify_gradient_flow()
        trainer.optimizer.zero_grad()

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Done!")


if __name__ == "__main__":
    main()
