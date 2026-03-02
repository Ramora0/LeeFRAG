"""Main training entry point for KV cache compression with Q-Former."""

import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, QFormerConfig, TrainingConfig
from collator import RAGCollator
from dataset import create_dataset
from kv_cache_utils import apply_rope_to_cache, concat_compressed_caches, extract_doc_hidden_states
from block_attention import build_block_causal_mask_with_qa
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
    parser.add_argument("--dataset", type=str, default="rag_v1", choices=["rag_v1", "hotpotqa"],
                        help="Dataset to train on (default: rag_v1)")
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
    parser.add_argument("--ce_only", action="store_true",
                        help="Train with CE loss only, no KL or hidden state loss")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--kl_top_k", type=int, default=0, help="Top-k logits for KL (0=full vocab)")
    parser.add_argument("--hidden_state_loss", action="store_true",
                        help="Use hidden state matching instead of KL divergence")
    parser.add_argument("--hidden_state_weight", type=float, default=1.0,
                        help="Weight for hidden state matching loss")
    parser.add_argument("--hidden_state_layers", type=str, default="all",
                        help="Which layers to match: 'all' or 'last_N' (e.g. 'last_8')")
    parser.add_argument("--cross_attn_mode", type=str, default="global", choices=["global", "windowed"])
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--compression_schedule", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                        help="Compression ratio schedule (default: 1 2 4 8 16)")
    return parser.parse_args()


@torch.no_grad()
def verify_identity_passthrough(model, qformer, train_loader, model_config, device):
    """Test how close the Q-Former is to identity at compression ratio 1.

    Compares the KV cache produced by the Q-Former against the LLM's real KV
    cache from a normal forward pass, reporting per-layer cosine similarity
    and relative MSE. Also compares final logits.
    """
    qformer.eval()
    batch = next(iter(train_loader))

    doc_token_ids = batch["doc_token_ids"]
    doc_lengths = batch["doc_lengths"]
    preamble_ids = batch["preamble_ids"]
    stage_b_input_ids = batch["stage_b_input_ids"].to(device)
    stage_b_labels = batch["stage_b_labels"].to(device)

    if not doc_token_ids or sum(doc_lengths) == 0:
        logger.warning("Identity test: empty batch, skipping")
        return

    preamble_len = preamble_ids.shape[0]
    block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

    # --- Stage A: get hidden states ---
    doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(device)
    preamble = preamble_ids.unsqueeze(0).to(device)
    full_input = torch.cat([preamble, doc_concat, stage_b_input_ids], dim=1)
    qa_length = stage_b_input_ids.shape[1]

    attn_mask = build_block_causal_mask_with_qa(
        block_lengths, qa_length, dtype=torch.float16, device=device,
    )
    outputs_a = model(
        input_ids=full_input, attention_mask=attn_mask,
        output_hidden_states=True, use_cache=False,
    )
    per_doc_hidden = extract_doc_hidden_states(
        outputs_a.hidden_states, block_lengths, model_config.num_layers,
    )
    teacher_logits = outputs_a.logits[:, sum(block_lengths):, :]

    # --- Real KV cache: normal LLM forward on concatenated docs ---
    doc_input = torch.cat([preamble, doc_concat], dim=1)
    real_outputs = model(
        input_ids=doc_input, use_cache=True, output_hidden_states=False,
    )
    real_cache = real_outputs.past_key_values

    # --- Q-Former compressed KV at ratio 1 ---
    with torch.amp.autocast("cuda"):
        per_doc_compressed = [qformer(doc_hs, compression_ratio=1) for doc_hs in per_doc_hidden]
        compressed_cache = concat_compressed_caches(per_doc_compressed, model_config.num_layers)
        rotary_emb = model.model.rotary_emb
        compressed_cache = apply_rope_to_cache(compressed_cache, model_config.num_layers, rotary_emb)

    # --- Compare KV caches layer by layer ---
    total_prefix_len = sum(block_lengths)
    compressed_prefix_len = compressed_cache.get_seq_length()
    logger.info(f"Identity test: real prefix len={total_prefix_len}, compressed prefix len={compressed_prefix_len}")

    k_cos_sims = []
    v_cos_sims = []
    k_rel_mses = []
    v_rel_mses = []

    for layer_idx in range(model_config.num_layers):
        # Real KV: [batch, num_kv_heads, seq_len, head_dim]
        real_k = real_cache.layers[layer_idx].keys.float()
        real_v = real_cache.layers[layer_idx].values.float()
        comp_k = compressed_cache.layers[layer_idx].keys.float()
        comp_v = compressed_cache.layers[layer_idx].values.float()

        # Truncate to min length for comparison
        min_len = min(real_k.shape[2], comp_k.shape[2])
        real_k = real_k[:, :, :min_len, :]
        real_v = real_v[:, :, :min_len, :]
        comp_k = comp_k[:, :, :min_len, :]
        comp_v = comp_v[:, :, :min_len, :]

        # Cosine similarity (flatten heads and positions)
        k_cos = F.cosine_similarity(real_k.reshape(-1, real_k.shape[-1]),
                                     comp_k.reshape(-1, comp_k.shape[-1]), dim=-1).mean().item()
        v_cos = F.cosine_similarity(real_v.reshape(-1, real_v.shape[-1]),
                                     comp_v.reshape(-1, comp_v.shape[-1]), dim=-1).mean().item()
        k_cos_sims.append(k_cos)
        v_cos_sims.append(v_cos)

        # Relative MSE: MSE / norm(real)^2
        k_mse = (real_k - comp_k).pow(2).mean().item()
        v_mse = (real_v - comp_v).pow(2).mean().item()
        k_norm = real_k.pow(2).mean().item()
        v_norm = real_v.pow(2).mean().item()
        k_rel_mses.append(k_mse / max(k_norm, 1e-8))
        v_rel_mses.append(v_mse / max(v_norm, 1e-8))

    avg_k_cos = sum(k_cos_sims) / len(k_cos_sims)
    avg_v_cos = sum(v_cos_sims) / len(v_cos_sims)
    avg_k_rel_mse = sum(k_rel_mses) / len(k_rel_mses)
    avg_v_rel_mse = sum(v_rel_mses) / len(v_rel_mses)

    logger.info(f"Identity test KV comparison (avg over {model_config.num_layers} layers):")
    logger.info(f"  K cosine sim: {avg_k_cos:.4f}  (1.0 = perfect)")
    logger.info(f"  V cosine sim: {avg_v_cos:.4f}  (1.0 = perfect)")
    logger.info(f"  K relative MSE: {avg_k_rel_mse:.4f}  (0.0 = perfect)")
    logger.info(f"  V relative MSE: {avg_v_rel_mse:.4f}  (0.0 = perfect)")

    # Log a few individual layers for detail
    for i in [0, model_config.num_layers // 2, model_config.num_layers - 1]:
        logger.info(f"  Layer {i:2d}: K_cos={k_cos_sims[i]:.4f} V_cos={v_cos_sims[i]:.4f} "
                     f"K_rmse={k_rel_mses[i]:.4f} V_rmse={v_rel_mses[i]:.4f}")

    # --- Compare logits: run LLM with compressed KV vs teacher ---
    with torch.amp.autocast("cuda"):
        student_out = model(
            input_ids=stage_b_input_ids,
            past_key_values=compressed_cache,
            use_cache=False,
        )
    student_logits = student_out.logits

    # CE on answer tokens
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = stage_b_labels[:, 1:].contiguous()
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    ).item()

    # KL between teacher and student
    valid_mask = (shift_labels != -100).view(-1)
    if valid_mask.any():
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        t_flat = teacher_logits[:, :-1, :].contiguous().view(-1, teacher_logits.size(-1))[valid_idx]
        s_flat = shift_logits.view(-1, shift_logits.size(-1))[valid_idx]
        t_lp = F.log_softmax(t_flat.float(), dim=-1)
        s_lp = F.log_softmax(s_flat.float(), dim=-1)
        kl = F.kl_div(s_lp, t_lp, log_target=True, reduction="batchmean").item()
    else:
        kl = float("nan")

    logger.info(f"Identity test logit comparison:")
    logger.info(f"  CE loss (compressed):  {ce_loss:.4f}  (ppl={math.exp(ce_loss):.2f})")
    logger.info(f"  KL(teacher||student):  {kl:.4f}")
    logger.info(f"  If identity were perfect, CE should match baseline and KL ~ 0")

    qformer.train()


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

    # Resume from checkpoint
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        qformer.load_state_dict(ckpt["qformer_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Resumed from step {ckpt.get('step', '?')}")

    # Test how close Q-Former is to identity at ratio 1
    logger.info("Running identity passthrough test...")
    verify_identity_passthrough(model, qformer, train_loader, model_config, device)

    # Verify gradient flow on first step
    logger.info("Running gradient flow verification...")
    sample_batch = next(iter(train_loader))
    loss = trainer._training_step(sample_batch, compression_ratio=2, global_step=0)
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
