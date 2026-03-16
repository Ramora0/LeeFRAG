"""Uncompressed baseline: CE loss of the raw LLM on context + continuation.

No Q-Former, no compression. Runs the frozen LLM with standard causal
attention on [context | continuation] and computes CE on continuation tokens.
This is the theoretical ceiling — compressed results can only be worse.

Usage:
    python scripts/eval_baseline.py
    python scripts/eval_baseline.py --model_name meta-llama/Llama-3.2-1B --eval_samples 200
"""

import argparse
import logging
import math
import os
import random
import warnings

import sys
_real_stderr = sys.stderr
_devnull = open(os.devnull, "w")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from leefrag.config import ModelConfig
from leefrag.data.dataset import create_dataset
from leefrag.data.npt_collator import NPTCollator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Uncompressed baseline eval")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset", type=str, default="slimpajama",
                        choices=["rag_v1", "hotpotqa", "slimpajama"])
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--eval_samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.disable(logging.NOTSET)
        logging.basicConfig(level=logging.INFO)
        del os.environ["TQDM_DISABLE"]
    else:
        sys.stderr = _devnull

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Auto-detect architecture
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

    # Load eval dataset
    eval_dataset = create_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        model_config=model_config,
        split="eval",
        eval_split_ratio=0.1,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    if args.eval_samples > 0 and len(eval_dataset) > args.eval_samples:
        eval_dataset = Subset(eval_dataset, range(args.eval_samples))

    collator = NPTCollator(tokenizer)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
    )

    # Eval: standard causal forward on [preamble | context | continuation]
    total_ce = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            doc_token_ids = batch["doc_token_ids"]
            preamble_ids = batch["preamble_ids"]
            continuation_ids = batch["stage_b_input_ids"].to(device)
            labels = batch["stage_b_labels"].to(device)

            if not doc_token_ids or sum(batch["doc_lengths"]) == 0:
                continue

            # [preamble | context | continuation] — plain causal, no block mask
            doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(device)
            preamble = preamble_ids.unsqueeze(0).to(device)
            full_input = torch.cat([preamble, doc_concat, continuation_ids], dim=1)

            context_len = preamble.shape[1] + doc_concat.shape[1]

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=full_input, use_cache=False)

            # CE only on continuation tokens
            cont_logits = outputs.logits[:, context_len:-1, :]
            cont_labels = labels[:, 1:]

            ce_loss = F.cross_entropy(
                cont_logits.reshape(-1, cont_logits.size(-1)),
                cont_labels.reshape(-1),
                ignore_index=-100,
            )

            total_ce += ce_loss.item()
            num_batches += 1

    avg_ce = total_ce / max(num_batches, 1)

    # Restore stderr
    sys.stderr = _real_stderr

    print("\n---")
    print(f"baseline_ce_loss:   {avg_ce:.6f}")
    print(f"baseline_perplexity:{math.exp(min(avg_ce, 20)):.2f}")
    print(f"eval_samples:       {num_batches}")
    print(f"model:              {args.model_name}")


if __name__ == "__main__":
    main()
