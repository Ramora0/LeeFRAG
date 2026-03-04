"""Main training entry point for absorber LoRA KV cache compression."""

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, TrainingConfig
from collator import RAGCollator
from dataset import create_dataset
from absorber_trainer import AbsorberLoRATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train absorber LoRA KV cache compressor")
    parser.add_argument("--dataset", type=str, default="hotpotqa", choices=["rag_v1", "hotpotqa"],
                        help="Dataset to train on (default: hotpotqa)")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=None, help="Eval every N steps (default: 4x per epoch)")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--gradient_checkpoint_llm", action="store_true")
    parser.add_argument("--offload_to_cpu", action="store_true",
                        help="Offload Phase 1 outputs to CPU between phases")
    parser.add_argument("--ce_only", action="store_true",
                        help="Train with CE loss only, no KL (saves memory, no teacher forward)")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--kl_top_k", type=int, default=0, help="Top-k logits for KL (0=full vocab)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--compression_schedule", type=int, nargs="+", default=[2, 4, 8, 16],
                        help="Compression ratio schedule (default: 2 4 8 16)")
    # LoRA-specific
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                        help="LLM modules to apply LoRA to")
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
        offload_stage_a_to_cpu=args.offload_to_cpu,
        ce_only_loss=args.ce_only,
        kl_weight=args.kl_weight,
        kl_top_k=args.kl_top_k,
        compression_schedule=args.compression_schedule,
        # LoRA config (stored but PEFT is set up here, not via trainer.py's lora flag)
        lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    set_seed(training_config.seed)
    os.makedirs(training_config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM
    logger.info(f"Loading LLM: {model_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Apply PEFT LoRA
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LoRA enabled: {lora_params / 1e6:.1f}M trainable params")

    if args.gradient_checkpoint_llm:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for LLM")

    # Load datasets
    logger.info(f"Loading datasets ({args.dataset})...")
    train_dataset = create_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        model_config=model_config,
        split="train",
        eval_split_ratio=training_config.eval_split_ratio,
        seed=training_config.seed,
    )
    eval_dataset = create_dataset(
        dataset_name=args.dataset,
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
    trainer = AbsorberLoRATrainer(
        model=model,
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
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, ckpt["lora_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Resumed from step {ckpt.get('step', '?')}")

    # Train
    logger.info("Starting absorber LoRA training...")
    trainer.train()

    logger.info("Done!")


if __name__ == "__main__":
    main()
