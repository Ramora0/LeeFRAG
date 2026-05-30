"""Precompute offline teacher logits (top-k) for the KL anchor.

Runs the ORIGINAL (un-adapted) model with full KV (gate=ones) over each example
and stores the top-k logits + indices at the answer positions, keyed by dataset
index. The trainer loads {idx}.pt for the KL loss.

Usage:
  python scripts/precompute_teacher.py --dataset rag_v1 --split train \
      --top_k 128 --out outputs_v2/teacher
"""

import _bootstrap  # noqa: F401

import argparse
import os

import torch
from tqdm import tqdm

from leefrag_v2.config import V2ModelConfig
from leefrag_v2.data.adapter import V2Collator, _to_model_config, build_blocks
from leefrag_v2.loader import build_teacher_model
from leefrag_v2.model.patch import new_context, set_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rag_v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--top_k", type=int, default=128)
    ap.add_argument("--out", default="outputs_v2/teacher")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval_split_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_flex", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mc = V2ModelConfig()
    model, tokenizer = build_teacher_model(mc, device)

    from leefrag.data.dataset import create_dataset

    ds = create_dataset(
        args.dataset, tokenizer, _to_model_config(mc), split=args.split,
        eval_split_ratio=args.eval_split_ratio, seed=args.seed,
    )
    collate = V2Collator(tokenizer)
    os.makedirs(args.out, exist_ok=True)

    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    for idx in tqdm(range(n), desc="teacher"):
        item = ds[idx]
        item["example_idx"] = idx
        blocks = build_blocks(collate([item]), device)
        if blocks is None:
            continue
        ctx = new_context(
            blocks["block_lengths"], blocks["qa_len"], device, mc.num_layers,
            use_flex=not args.no_flex, teacher_mode=True,
        )
        set_context(model, ctx)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            logits = model(input_ids=blocks["input_ids"], use_cache=False).logits
        set_context(model, None)

        lab = blocks["labels"][0, 1:]
        ans = logits[0, :-1, :][lab != -100]  # [N_ans, V]
        if ans.numel() == 0:
            continue
        k = min(args.top_k, ans.shape[-1])
        vals, idxs = ans.float().topk(k, dim=-1)
        torch.save(
            {"vals": vals.half().cpu(), "idx": idxs.cpu()},
            os.path.join(args.out, f"{idx}.pt"),
        )

    print(f"Saved teacher logits for up to {n} examples -> {args.out}")


if __name__ == "__main__":
    main()
