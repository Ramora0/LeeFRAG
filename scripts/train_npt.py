"""Next-token prediction pretraining entry point for Q-Former KV compressor.

Trains the Q-Former to compress document KV caches such that a frozen LLM
can predict continuation text from the compressed prefix. All continuation
tokens are supervised (unlike RAG training which only supervises answers).

This is the recommended pretraining stage before RAG fine-tuning:
    1. python scripts/train_npt.py --gradient_checkpoint_llm   (pretrain)
    2. python scripts/train.py --init_qformer_from outputs_npt/checkpoint-*/checkpoint.pt  (finetune)
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
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


def detect_model_config(model, model_name: str) -> ModelConfig:
    """Auto-detect architecture dimensions from a loaded HuggingFace model."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", hidden_size // cfg.num_attention_heads)
    logger.info(
        f"Detected architecture: {num_layers} layers, "
        f"hidden={hidden_size}, kv_heads={num_kv_heads}, head_dim={head_dim}"
    )
    return ModelConfig(
        model_name=model_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="NPT pretraining for Q-Former KV compressor")
    parser.add_argument("--model_name", type=str, default=ModelConfig.model_name,
                        help="HuggingFace model name or path (default: Tulu3-Block-FT 8B)")
    parser.add_argument("--dataset", type=str, default="slimpajama",
                        choices=["rag_v1", "hotpotqa", "slimpajama"],
                        help="Dataset to train on (default: slimpajama)")
    parser.add_argument("--sources", type=str, nargs="+", default=["book", "arxiv"],
                        choices=["book", "arxiv", "wikipedia", "c4", "commoncrawl",
                                 "stackexchange", "github"],
                        help="SlimPajama source domains (default: book arxiv)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max dataset samples, 0=no limit (default: 0)")
    parser.add_argument("--max_continuation_tokens", type=int, default=1024,
                        help="Max continuation tokens for general text (default: 1024)")
    parser.add_argument("--min_chars", type=int, default=4000,
                        help="Min character length to include a text (default: 4000)")
    parser.add_argument("--output_dir", type=str, default="outputs_npt")
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
                        help="Train with CE loss only, no KL divergence")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--kl_top_k", type=int, default=0, help="Top-k logits for KL (0=full vocab)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--compression_schedule", type=int, nargs="+", default=[2, 4, 8, 16],
                        help="Compression ratio schedule (default: 2 4 8 16)")
    parser.add_argument("--cross_attn_mode", type=str, default="global", choices=["global", "chunked"],
                        help="Cross-attention mode: 'global' or 'chunked'")
    parser.add_argument("--scale", type=int, default=1,
                        help="Scale factor for Q-Former attn_dim and ffn_dim (default: 1)")
    parser.add_argument("--layer_adapter_rank", type=int, default=0,
                        help="Per-layer low-rank adapter rank (0=disabled)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load LLM first so we can auto-detect architecture
    logger.info(f"Loading LLM: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model_config = detect_model_config(model, args.model_name)
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
        compression_schedule=args.compression_schedule,
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

    if training_config.gradient_checkpoint_llm:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for LLM (Stage B)")

    # Build Q-Former
    logger.info("Building Q-Former KV compressor")
    qformer = QFormerKVCompressor(qformer_config, model_config, llm=model).to(device)
    if qformer_config.cross_attn_mode == "chunked":
        qformer._cross_attend_chunked = torch.compile(qformer._cross_attend_chunked, dynamic=True)
    else:
        qformer._cross_attend = torch.compile(qformer._cross_attend, dynamic=True)
    trainable_params = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    logger.info(f"Q-Former trainable params: {trainable_params / 1e6:.1f}M")

    # Load datasets
    logger.info(f"Loading datasets ({args.dataset})...")
    dataset_kwargs = {}
    if args.dataset == "slimpajama":
        dataset_kwargs = dict(
            sources=tuple(args.sources),
            max_samples=args.max_samples,
            max_continuation_tokens=args.max_continuation_tokens,
            min_chars=args.min_chars,
        )
    train_dataset = create_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        model_config=model_config,
        split="train",
        eval_split_ratio=training_config.eval_split_ratio,
        seed=training_config.seed,
        **dataset_kwargs,
    )
    eval_dataset = create_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        model_config=model_config,
        split="eval",
        eval_split_ratio=training_config.eval_split_ratio,
        seed=training_config.seed,
        **dataset_kwargs,
    )
    logger.info(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

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

    # Resume from checkpoint
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        qformer.load_state_dict(ckpt["qformer_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Resumed from step {ckpt.get('step', '?')}")

    # Train
    logger.info("Starting NPT pretraining...")
    trainer.train()

    logger.info("Done!")


if __name__ == "__main__":
    main()
