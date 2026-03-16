"""Baseline evaluation: run frozen LLM with full documents (no Q-Former compression).

Baselines:
1. Full causal attention (standard RAG) — absolute ceiling.
2. Block attention for docs, full attention for Q+A — ceiling given our attention pattern.
3. Mean-pooled KV at 2x, 4x, 8x, 16x — naive compression floor.
"""

import argparse
import json
import logging
import math
import os
from functools import partial

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from leefrag.model.block_attention import (
    build_block_causal_mask,
    build_block_causal_mask_with_qa,
    build_prefix_causal_mask,
)
from leefrag.data.collator import RAGCollator
from leefrag.config import ModelConfig, TrainingConfig
from leefrag.data.dataset import create_dataset
from leefrag.utils.kv_cache_utils import build_dynamic_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = "eval_cache"
MAX_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Result caching
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")


def _load_cache(name: str) -> dict:
    path = _cache_path(name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(name: str, cache: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(name), "w") as f:
        json.dump(cache, f)


# ---------------------------------------------------------------------------
# Eval loop with caching
# ---------------------------------------------------------------------------

def eval_loop(name, dataset, model, tokenizer, device, forward_fn, cache_name=None):
    """Shared eval loop with per-sample result caching.

    forward_fn(model, item, tokenizer, device) -> (loss_sum, n_valid_tokens) or None.
    Cached results are stored in eval_cache/<cache_name>.json so re-runs skip
    already-evaluated samples.
    """
    if cache_name is None:
        cache_name = name.lower().replace(" ", "_")
    cache = _load_cache(cache_name)

    total_loss = 0.0
    total_tokens = 0
    num_samples = 0
    dirty = False

    n = min(len(dataset), MAX_SAMPLES)
    pbar = tqdm(range(n), desc=name)
    for idx in pbar:
        key = str(idx)

        if key in cache:
            entry = cache[key]
            if entry is None:
                continue
            loss_sum, n_valid = entry["loss_sum"], entry["n_valid"]
        else:
            item = dataset[idx]
            result = forward_fn(model, item, tokenizer, device)
            if result is None:
                cache[key] = None
                dirty = True
                if dirty and idx % 20 == 0:
                    _save_cache(cache_name, cache)
                    dirty = False
                continue
            loss_sum, n_valid = result
            cache[key] = {"loss_sum": loss_sum, "n_valid": n_valid}
            dirty = True
            if idx % 20 == 0:
                _save_cache(cache_name, cache)
                dirty = False

        total_loss += loss_sum
        total_tokens += n_valid
        num_samples += 1

        if total_tokens > 0:
            avg = total_loss / total_tokens
            pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{math.exp(avg):.2f}")

    if dirty:
        _save_cache(cache_name, cache)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, num_samples, total_tokens


# ---------------------------------------------------------------------------
# Baseline inputs
# ---------------------------------------------------------------------------

def build_baseline_input(
    item: dict, tokenizer, max_total_tokens: int = 4096
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build full input with documents in the system message.

    Format matches block attention layout:
    system(system_prompt + docs) + user(question) + assistant(answer)
    """
    doc_texts = item["doc_texts"]
    full_docs = "\n\n".join(doc_texts)
    system_with_docs = f"{item['system_prompt']}\n\n{full_docs}"
    user_content = item["question_suffix"]

    messages = [
        {"role": "system", "content": system_with_docs},
        {"role": "user", "content": user_content},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = item["answer_ids"].tolist()

    # Tulu 3 uses <|end_of_text|> (EOS) as the end-of-turn token
    end_token = tokenizer.eos_token_id
    full_ids = prompt_ids + answer_ids + [end_token]

    if len(full_ids) > max_total_tokens:
        excess = len(full_ids) - max_total_tokens
        prompt_ids = prompt_ids[excess:]
        full_ids = prompt_ids + answer_ids + [end_token]

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    answer_start = len(prompt_ids)
    labels[answer_start:] = input_ids[answer_start:]

    return input_ids, labels


# ---------------------------------------------------------------------------
# Forward functions
# ---------------------------------------------------------------------------

def forward_full_causal(model, item, tokenizer, device):
    """Standard causal attention — all documents in one sequence."""
    result = build_baseline_input(item, tokenizer)
    if result is None:
        return None

    input_ids, labels = result
    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)

    with torch.amp.autocast("cuda"):
        outputs = model(input_ids=input_ids, use_cache=False)

    return _compute_token_loss(outputs.logits, labels)


def forward_block_attention(model, item, tokenizer, device):
    """Block attention for docs, Q+A attends to all blocks + causal within itself."""
    doc_token_ids = item["doc_token_ids"]
    doc_lengths = [t.shape[0] for t in doc_token_ids]

    if not doc_token_ids or sum(doc_lengths) == 0:
        return None

    # Build Q+A tokens and preamble using the collator logic
    collator = _get_collator(tokenizer)
    stage_b_ids, answer_start, answer_end = collator._build_stage_b_tokens(item)
    preamble_ids = item["preamble_ids"]

    # Block lengths: preamble merges with first doc
    preamble_len = preamble_ids.shape[0]
    block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

    # Concatenate: [preamble | doc0 | doc1 | ... | Q+A]
    doc_concat = torch.cat(doc_token_ids, dim=0)
    full_input = torch.cat([preamble_ids, doc_concat, stage_b_ids], dim=0).unsqueeze(0).to(device)

    qa_length = stage_b_ids.shape[0]

    # Block-diagonal for blocks, Q+A attends to all blocks + causal
    attn_mask = build_block_causal_mask_with_qa(
        block_lengths, qa_length, dtype=torch.float16, device=device
    )

    # Labels: -100 for everything except answer tokens (offset by block total)
    block_total = sum(block_lengths)
    full_labels = torch.full((full_input.shape[1],), -100, dtype=torch.long, device=device)
    full_labels[block_total + answer_start : block_total + answer_end] = (
        full_input[0, block_total + answer_start : block_total + answer_end]
    )
    full_labels = full_labels.unsqueeze(0)

    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=full_input,
            attention_mask=attn_mask,
            use_cache=False,
        )

    return _compute_token_loss(outputs.logits, full_labels)


def _mean_pool_kv(k, v, compression_ratio):
    """Mean pool KV tensors along the sequence dimension.

    Args:
        k, v: [batch, num_kv_heads, seq_len, head_dim]
        compression_ratio: Pool groups of this many tokens into one.

    Returns:
        (k_pooled, v_pooled) each [batch, num_kv_heads, num_compressed, head_dim]
    """
    b, h, s, d = k.shape
    num_compressed = max(1, s // compression_ratio)

    if s <= compression_ratio:
        # Fewer tokens than ratio — collapse to a single token
        return k.mean(dim=2, keepdim=True), v.mean(dim=2, keepdim=True)

    # Truncate to evenly divisible length, then reshape and average
    usable = num_compressed * compression_ratio
    k_pooled = k[:, :, :usable, :].reshape(b, h, num_compressed, compression_ratio, d).mean(dim=3)
    v_pooled = v[:, :, :usable, :].reshape(b, h, num_compressed, compression_ratio, d).mean(dim=3)
    return k_pooled, v_pooled


def forward_mean_pool(model, item, tokenizer, device, compression_ratio):
    """Mean-pool KV baseline: run docs to get KV caches, pool, use as prefix.

    This is a naive compression baseline — mean-pooling mixes positional
    information (RoPE is already baked into K) and discards fine-grained
    token-level structure. Should serve as a floor that Q-Former beats.
    """
    doc_token_ids = item["doc_token_ids"]
    doc_lengths = [t.shape[0] for t in doc_token_ids]

    if not doc_token_ids or sum(doc_lengths) == 0:
        return None

    # Build Q+A tokens and preamble
    collator = _get_collator(tokenizer)
    stage_b_ids, answer_start, answer_end = collator._build_stage_b_tokens(item)
    preamble_ids = item["preamble_ids"]

    # Block lengths: preamble merges with first doc
    preamble_len = preamble_ids.shape[0]
    block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

    # --- Run preamble+docs through model with block attention to get KV caches ---
    doc_concat = torch.cat(doc_token_ids, dim=0)
    block_input = torch.cat([preamble_ids, doc_concat], dim=0).unsqueeze(0).to(device)
    doc_attn_mask = build_block_causal_mask(
        block_lengths, dtype=torch.float16, device=device
    )

    with torch.amp.autocast("cuda"):
        doc_outputs = model(
            input_ids=block_input,
            attention_mask=doc_attn_mask,
            use_cache=True,
        )

    full_kv = doc_outputs.past_key_values  # DynamicCache with all block tokens
    num_layers = len(full_kv.key_cache)

    # --- Slice per-block and mean-pool ---
    pooled_kv_pairs = []  # per layer: (k_concat, v_concat)
    for layer_idx in range(num_layers):
        full_k = full_kv.key_cache[layer_idx]  # [1, heads, total_block_len, dim]
        full_v = full_kv.value_cache[layer_idx]

        doc_k_parts = []
        doc_v_parts = []
        offset = 0
        for bl in block_lengths:
            k_slice = full_k[:, :, offset:offset + bl, :]
            v_slice = full_v[:, :, offset:offset + bl, :]
            k_pooled, v_pooled = _mean_pool_kv(k_slice, v_slice, compression_ratio)
            doc_k_parts.append(k_pooled)
            doc_v_parts.append(v_pooled)
            offset += bl

        # Concatenate all docs for this layer
        pooled_kv_pairs.append((
            torch.cat(doc_k_parts, dim=2),
            torch.cat(doc_v_parts, dim=2),
        ))

    # Build DynamicCache from pooled KV
    compressed_cache = build_dynamic_cache(pooled_kv_pairs)

    # --- Run Q+A with pooled prefix ---
    prefix_length = compressed_cache.get_seq_length()
    stage_b_ids = stage_b_ids.unsqueeze(0).to(device)
    seq_length = stage_b_ids.shape[1]

    attn_mask = build_prefix_causal_mask(
        prefix_length, seq_length, dtype=torch.float16, device=device
    )

    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=stage_b_ids,
            attention_mask=attn_mask,
            past_key_values=compressed_cache,
            use_cache=False,
        )

    # Labels: only answer tokens
    full_labels = torch.full((seq_length,), -100, dtype=torch.long, device=device)
    full_labels[answer_start:answer_end] = stage_b_ids[0, answer_start:answer_end]
    full_labels = full_labels.unsqueeze(0)

    return _compute_token_loss(outputs.logits, full_labels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_collator_cache = {}


def _get_collator(tokenizer):
    tid = id(tokenizer)
    if tid not in _collator_cache:
        _collator_cache[tid] = RAGCollator(tokenizer)
    return _collator_cache[tid]


def _compute_token_loss(logits, labels):
    """Compute per-token loss, return (loss_sum, n_valid_tokens)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )

    valid_mask = shift_labels.view(-1) != -100
    n_valid = valid_mask.sum().item()
    if n_valid == 0:
        return None

    return loss_per_token[valid_mask].sum().item(), n_valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_baseline(dataset_name: str = "rag_v1"):
    model_config = ModelConfig()
    training_config = TrainingConfig()

    logger.info(f"Loading tokenizer from {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    logger.info(f"Loading eval dataset ({dataset_name})")
    dataset = create_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        model_config=model_config,
        split="eval",
        eval_split_ratio=training_config.eval_split_ratio,
        seed=training_config.seed,
    )

    device = next(model.parameters()).device

    results = {}

    # === Baseline 1: Full causal attention ===
    causal_loss, causal_ppl, causal_n, causal_tok = eval_loop(
        "Full Causal", dataset, model, tokenizer, device,
        forward_full_causal, cache_name=f"{dataset_name}_full_causal",
    )
    results["full_causal"] = {"loss": causal_loss, "ppl": causal_ppl}

    # === Baseline 2: Block attention ===
    block_loss, block_ppl, block_n, block_tok = eval_loop(
        "Block Attention", dataset, model, tokenizer, device,
        forward_block_attention, cache_name=f"{dataset_name}_block_attention",
    )
    results["block_attention"] = {"loss": block_loss, "ppl": block_ppl}

    # === Mean-pooled KV baselines at various compression ratios ===
    compression_ratios = [2, 4, 8, 16]
    pool_results = {}
    for ratio in compression_ratios:
        name = f"Mean Pool {ratio}x"
        cache_name = f"{dataset_name}_mean_pool_{ratio}x"
        forward_fn = partial(forward_mean_pool, compression_ratio=ratio)
        loss, ppl, n_samples, n_tokens = eval_loop(
            name, dataset, model, tokenizer, device,
            forward_fn, cache_name=cache_name,
        )
        pool_results[ratio] = {"loss": loss, "ppl": ppl, "n": n_samples, "tok": n_tokens}
        results[f"mean_pool_{ratio}x"] = {"loss": loss, "ppl": ppl}

    # === Results ===
    logger.info("=" * 70)
    logger.info(f"BASELINE RESULTS [{dataset_name}]")
    logger.info("=" * 70)
    logger.info(
        f"  Full Causal:     CE={causal_loss:.4f}  PPL={causal_ppl:.2f}  "
        f"({causal_n} samples, {causal_tok} tokens)"
    )
    logger.info(
        f"  Block Attention: CE={block_loss:.4f}  PPL={block_ppl:.2f}  "
        f"({block_n} samples, {block_tok} tokens)"
    )
    delta = block_loss - causal_loss
    logger.info(f"  Block attn overhead: +{delta:.4f} CE ({delta/causal_loss*100:.1f}%)")
    logger.info("-" * 70)
    for ratio in compression_ratios:
        r = pool_results[ratio]
        delta_from_block = r["loss"] - block_loss
        logger.info(
            f"  Mean Pool {ratio:2d}x:   CE={r['loss']:.4f}  PPL={r['ppl']:.2f}  "
            f"(+{delta_from_block:.4f} vs block)  "
            f"({r['n']} samples, {r['tok']} tokens)"
        )
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline evaluation (no Q-Former)")
    parser.add_argument(
        "--dataset", type=str, default="rag_v1", choices=["rag_v1", "hotpotqa"],
        help="Dataset to evaluate on (default: rag_v1)",
    )
    args = parser.parse_args()
    evaluate_baseline(dataset_name=args.dataset)
