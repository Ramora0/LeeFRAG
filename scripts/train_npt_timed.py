"""Time-budgeted NPT pretraining for Q-Former KV compressor.

Trains with a fixed wall-clock time budget. Evaluation runs after training
completes and is NOT counted against the time budget.

Prints a machine-readable summary at the end for automated experiment tracking.

Usage:
    python scripts/train_npt_timed.py --model_name meta-llama/Llama-3.2-1B
    python scripts/train_npt_timed.py --model_name meta-llama/Llama-3.2-1B --time_budget 600
"""

import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from leefrag.config import ModelConfig, QFormerConfig, TrainingConfig
from leefrag.data.dataset import create_dataset
from leefrag.data.npt_collator import NPTCollator
from leefrag.model.qformer import QFormerKVCompressor
from leefrag.training.npt_trainer import NPTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Time-budgeted NPT pretraining")

    # Model
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model name")

    # Time budget
    parser.add_argument("--time_budget", type=int, default=300,
                        help="Training time budget in seconds (default: 300 = 5 min)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="rag_v1",
                        choices=["rag_v1", "hotpotqa", "slimpajama"])
    parser.add_argument("--eval_samples", type=int, default=200,
                        help="Max eval samples (0 = use full eval set)")

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs_npt_timed")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--gradient_checkpoint_llm", action="store_true")
    parser.add_argument("--offload_stage_a_to_cpu", action="store_true")

    # Loss
    parser.add_argument("--ce_only", action="store_true")
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--kl_top_k", type=int, default=0)

    # Compression
    parser.add_argument("--compression_schedule", type=int, nargs="+", default=[2, 4, 8, 16])

    # Q-Former
    parser.add_argument("--cross_attn_mode", type=str, default="global",
                        choices=["global", "chunked"])
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--layer_adapter_rank", type=int, default=0)

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Auto-detect architecture from loaded model
    hf_config = model.config
    model_config = ModelConfig(
        model_name=args.model_name,
        num_layers=hf_config.num_hidden_layers,
        num_kv_heads=hf_config.num_key_value_heads,
        head_dim=getattr(
            hf_config, "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        hidden_size=hf_config.hidden_size,
    )
    logger.info(
        f"Model arch: {model_config.num_layers} layers, "
        f"{model_config.hidden_size} hidden, "
        f"{model_config.num_kv_heads} kv_heads, "
        f"{model_config.head_dim} head_dim"
    )

    if args.gradient_checkpoint_llm:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for LLM")

    # Configs
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        gradient_checkpoint_llm=args.gradient_checkpoint_llm,
        offload_stage_a_to_cpu=args.offload_stage_a_to_cpu,
        ce_only_loss=args.ce_only,
        kl_weight=args.kl_weight,
        kl_top_k=args.kl_top_k,
        compression_schedule=args.compression_schedule,
    )

    qformer_config = QFormerConfig(
        attn_dim=256 * args.scale,
        num_attn_heads=8 * args.scale,
        ffn_dim=256 * args.scale,
        cross_attn_mode=args.cross_attn_mode,
        layer_adapter_rank=args.layer_adapter_rank,
    )

    # Build Q-Former
    logger.info("Building Q-Former KV compressor")
    qformer = QFormerKVCompressor(qformer_config, model_config, llm=model).to(device)
    if qformer_config.cross_attn_mode == "chunked":
        qformer._cross_attend_chunked = torch.compile(
            qformer._cross_attend_chunked, dynamic=True,
        )
    else:
        qformer._cross_attend = torch.compile(
            qformer._cross_attend, dynamic=True,
        )
    trainable_params = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    logger.info(f"Q-Former trainable params: {trainable_params / 1e6:.1f}M")

    # Datasets
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

    if args.eval_samples > 0 and len(eval_dataset) > args.eval_samples:
        eval_dataset = Subset(eval_dataset, range(args.eval_samples))
        logger.info(f"Eval capped to {args.eval_samples} samples")

    collator = NPTCollator(tokenizer)
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
    trainer = NPTTrainer(
        model=model,
        qformer=qformer,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        model_config=model_config,
        training_config=training_config,
        device=device,
    )

    # Resume
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        qformer.load_state_dict(ckpt["qformer_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])

    # === Train (timed) ===
    os.makedirs(training_config.output_dir, exist_ok=True)
    logger.info(f"Starting timed NPT training ({args.time_budget}s budget)...")

    train_start = time.monotonic()
    global_step = trainer.train_timed(args.time_budget)
    training_seconds = time.monotonic() - train_start

    # === Eval (not counted in time budget) ===
    logger.info("Running final evaluation (outside time budget)...")
    final_ratio = training_config.compression_schedule[-1]
    eval_ce, eval_kl = trainer.evaluate(final_ratio)
    total_seconds = time.monotonic() - train_start

    # Peak VRAM
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Summary
    kl_weight = training_config.kl_weight
    eval_total = eval_ce + kl_weight * eval_kl

    print("\n---")
    print(f"eval_ce_loss:       {eval_ce:.6f}")
    print(f"eval_kl_loss:       {eval_kl:.6f}")
    print(f"eval_total_loss:    {eval_total:.6f}")
    print(f"eval_perplexity:    {math.exp(min(eval_ce, 20)):.2f}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"total_seconds:      {total_seconds:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"total_steps:        {global_step}")
    print(f"model:              {args.model_name}")
    print(f"qformer_params_M:   {trainable_params / 1e6:.1f}")


if __name__ == "__main__":
    main()
