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
from dataset import create_dataset
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
    parser.add_argument("--offload_stage_a_to_cpu", action="store_true")
    parser.add_argument("--ce_only", action="store_true",
                        help="Train with CE loss only, no KL or hidden state loss")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--kl_top_k", type=int, default=0, help="Top-k logits for KL (0=full vocab)")
    parser.add_argument("--hidden_state_loss", action=argparse.BooleanOptionalAction, default=True,
                        help="Use hidden state matching instead of KL divergence")
    parser.add_argument("--hidden_state_weight", type=float, default=10.0,
                        help="Weight for hidden state matching loss")
    parser.add_argument("--hidden_state_layers", type=str, default="all",
                        help="Which layers to match: 'all' or 'last_N' (e.g. 'last_8')")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--compression_schedule", type=int, nargs="+", default=[2, 4, 8, 16],
                        help="Compression ratio schedule (default: 2 4 8 16)")
    parser.add_argument("--cross_attn_mode", type=str, default="global", choices=["global", "chunked"],
                        help="Cross-attention mode: 'global' (pooled queries attend all) or 'chunked' (one query per chunk)")
    parser.add_argument("--scale", type=int, default=1,
                        help="Scale factor for Q-Former attn_dim and ffn_dim (default: 1)")
    parser.add_argument("--layer_adapter_rank", type=int, default=0,
                        help="Per-layer low-rank adapter rank (0=disabled, try 8)")
    parser.add_argument("--lora", action="store_true",
                        help="Train the LLM with LoRA adapters alongside Q-Former")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj"],
                        help="LLM modules to apply LoRA to (default: q_proj v_proj)")
    parser.add_argument("--init_qformer_from", type=str, default=None,
                        help="Initialize Q-Former from checkpoint (Phase 2: weights only, fresh optimizer)")
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
        ce_only_loss=args.ce_only,
        kl_weight=args.kl_weight,
        kl_top_k=args.kl_top_k,
        hidden_state_loss=args.hidden_state_loss,
        hidden_state_weight=args.hidden_state_weight,
        hidden_state_layers=args.hidden_state_layers,
        compression_schedule=args.compression_schedule,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    qformer_config = QFormerConfig(
        attn_dim=256 * args.scale,
        num_attn_heads=8 * args.scale,
        ffn_dim=256 * args.scale,
        cross_attn_mode=args.cross_attn_mode,
        layer_adapter_rank=args.layer_adapter_rank,
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

    # Apply LoRA adapters if requested
    base_model = model  # keep reference to unwrapped LlamaForCausalLM
    if training_config.lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_config.lora_rank,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=training_config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"LoRA enabled: {lora_params / 1e6:.1f}M trainable LLM params")

    if training_config.gradient_checkpoint_llm:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for LLM (Stage B)")

    # Build Q-Former (pass unwrapped LLM so frozen KV projections are copied correctly)
    logger.info("Building Q-Former KV compressor")
    qformer = QFormerKVCompressor(qformer_config, model_config, llm=base_model).to(device)
    if qformer_config.cross_attn_mode == "chunked":
        qformer._cross_attend_chunked = torch.compile(qformer._cross_attend_chunked, dynamic=True)
    else:
        qformer._cross_attend = torch.compile(qformer._cross_attend, dynamic=True)
    trainable_params = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    logger.info(f"Q-Former trainable params: {trainable_params / 1e6:.1f}M")

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

    # Initialize Q-Former from pretrained checkpoint (Phase 2: weights only, fresh optimizer)
    if args.init_qformer_from:
        logger.info(f"Initializing Q-Former from: {args.init_qformer_from}")
        ckpt = torch.load(args.init_qformer_from, map_location=device, weights_only=True)
        qformer.load_state_dict(ckpt["qformer_state_dict"])
        logger.info(f"Loaded Q-Former weights from step {ckpt.get('step', '?')} (fresh optimizer)")

    # Resume from checkpoint
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        qformer.load_state_dict(ckpt["qformer_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if training_config.lora and "lora_state_dict" in ckpt:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model, ckpt["lora_state_dict"])
            logger.info("Restored LoRA adapter weights from checkpoint")
        logger.info(f"Resumed from step {ckpt.get('step', '?')}")

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Done!")


if __name__ == "__main__":
    main()
