"""Absorber token evaluation pipeline for KV cache compression.

Uses the LLM itself as the compressor — zero trainable parameters. "Absorber"
summary tokens are initialized from mean-pooled document embeddings, appended
to the document, and passed through the LLM with a one-way attention mask:
summaries attend to documents (absorb info) but documents cannot attend to
summaries (no info leakage). The resulting KV values at summary positions become
the compressed prefix for Q+A.

Conditions:
1. Full context — all docs inline, standard causal attention (ceiling)
2. Absorber Nx — absorber compressed KV prefix at each ratio
3. No prefix — question only, no documents (floor)

Metrics: CE, PPL, F1, EM, Sub-span EM (same as eval.py)
"""

import argparse
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from config import ModelConfig
from eval import (
    SYSTEM_PROMPT,
    best_subspan_em,
    compute_ce_loss,
    compute_em,
    compute_f1,
    eval_ce_full_context,
    eval_ce_no_prefix,
    generate_full_context,
    generate_no_prefix,
    greedy_decode,
    load_hotpotqa,
    load_model,
    prep_hotpotqa_sample,
)
from kv_cache_utils import _rotate_half, apply_rope_to_cache, build_dynamic_cache

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Absorber attention mask
# ---------------------------------------------------------------------------

def build_absorber_mask(
    num_summary: int,
    num_doc: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build one-way attention mask for absorber tokens.

    Layout: [summary(K) | doc(N)]

                 summary(K)    doc(N)
        summary  [causal       FULL   ]   <- summaries absorb from ALL doc tokens
        doc      [MASKED       causal ]   <- docs don't see summaries

    Args:
        num_summary: Number of absorber summary tokens (K).
        num_doc: Number of document tokens (N).
        dtype: Mask dtype (float16/bfloat16).
        device: Target device.

    Returns:
        mask: [1, 1, K+N, K+N] additive attention mask (0.0=attend, -inf=masked).
    """
    total = num_summary + num_doc
    mask = torch.full((total, total), float("-inf"), dtype=dtype, device=device)

    # Summary-to-summary: causal
    if num_summary > 0:
        summary_causal = torch.triu(
            torch.full((num_summary, num_summary), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
        mask[:num_summary, :num_summary] = summary_causal

    # Summary-to-doc: full attention (summaries attend to all doc tokens)
    mask[:num_summary, num_summary:] = 0.0

    # Doc-to-doc: causal
    doc_causal = torch.triu(
        torch.full((num_doc, num_doc), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )
    mask[num_summary:, num_summary:] = doc_causal

    # Doc-to-summary: already -inf (masked)

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, K+N, K+N]


# ---------------------------------------------------------------------------
# RoPE un-application (inverse of apply_rope_to_cache)
# ---------------------------------------------------------------------------

def unapply_rope(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Remove RoPE from K values.

    Inverse of: k_rotated = k * cos + rotate_half(k) * sin
    Solution:   k_raw = k_rotated * cos - rotate_half(k_rotated) * (-sin)
                      = k_rotated * cos + rotate_half(k_rotated) * sin
                ... wait, let's derive properly.

    Forward RoPE:  k_rot = k * cos + rotate_half(k) * sin
    Inverse:       k = k_rot * cos + rotate_half(k_rot) * (-sin)
                 No — rotation matrix R(theta) has inverse R(-theta):
                   k = k_rot * cos + rotate_half(k_rot) * (-sin)
                 Actually for the standard rotation matrix:
                   R(theta) = [[cos, -sin], [sin, cos]]
                   R(-theta) = [[cos, sin], [-sin, cos]]
                 In rotate_half form: R(-theta)(x) = x * cos - rotate_half(x) * sin

    So: k_raw = k_rotated * cos - rotate_half(k_rotated) * sin
    """
    return k * cos - _rotate_half(k) * sin


# ---------------------------------------------------------------------------
# Per-document absorber forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def absorber_forward_single_doc(
    model,
    doc_ids: torch.Tensor,
    compression_ratio: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run absorber forward on a single document block.

    1. Embed doc tokens
    2. Mean-pool to K summary embeddings
    3. Forward with absorber mask
    4. Extract KV at summary positions
    5. Un-apply RoPE from K

    Args:
        model: Frozen LLM.
        doc_ids: [N] token ids for this document block.
        compression_ratio: How many doc tokens per summary token.
        device: CUDA device.

    Returns:
        List of (k_raw, v) per layer, each [1, num_kv_heads, K, head_dim].
        K values have RoPE removed (raw) for later re-application.
    """
    doc_ids = doc_ids.to(device)
    N = doc_ids.shape[0]
    K = max(1, N // compression_ratio)

    # 1. Embed document tokens
    with torch.amp.autocast("cuda"):
        doc_embeds = model.model.embed_tokens(doc_ids.unsqueeze(0))  # [1, N, hidden]

    # 2. Mean-pool to K summary embeddings via adaptive_avg_pool1d
    # adaptive_avg_pool1d expects [batch, channels, length]
    doc_t = doc_embeds.permute(0, 2, 1)  # [1, hidden, N]
    summary_t = F.adaptive_avg_pool1d(doc_t, K)  # [1, hidden, K]
    summary_embeds = summary_t.permute(0, 2, 1)  # [1, K, hidden]

    # 3. Concat [summary | doc] as inputs_embeds
    inputs_embeds = torch.cat([summary_embeds, doc_embeds], dim=1)  # [1, K+N, hidden]

    # 4. Build attention mask
    attn_mask = build_absorber_mask(K, N, dtype=inputs_embeds.dtype, device=device)

    # 5. Position IDs: [0, K+N)
    position_ids = torch.arange(K + N, device=device).unsqueeze(0)

    # 6. Forward
    with torch.amp.autocast("cuda"):
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=False,
        )

    # 7. Extract KV at summary positions [0, K) from output cache
    cache = outputs.past_key_values

    # 8. Get RoPE cos/sin for un-application at positions [0, K)
    summary_pos_ids = torch.arange(K, device=device).unsqueeze(0)
    rotary_emb = model.model.rotary_emb
    # Need a dummy tensor for the rotary_emb call (it uses it for dtype/device)
    dummy = cache[0][0][:, :, :K, :]  # [1, heads, K, head_dim]
    cos, sin = rotary_emb(dummy, position_ids=summary_pos_ids)
    cos = cos.unsqueeze(1)  # [1, 1, K, head_dim]
    sin = sin.unsqueeze(1)

    # 9. Extract and un-apply RoPE
    raw_kv_pairs = []
    num_layers = len(cache)
    for layer_idx in range(num_layers):
        k_full, v_full = cache[layer_idx]
        # Extract summary positions only
        k_summary = k_full[:, :, :K, :]  # [1, heads, K, head_dim]
        v_summary = v_full[:, :, :K, :]

        # Un-apply RoPE from K
        k_raw = unapply_rope(k_summary, cos, sin)

        raw_kv_pairs.append((k_raw, v_summary))

    return raw_kv_pairs


# ---------------------------------------------------------------------------
# Multi-document absorber compression
# ---------------------------------------------------------------------------

@torch.no_grad()
def absorber_compress_docs(
    model,
    doc_token_ids: list[torch.Tensor],
    block_lengths: list[int],
    preamble_ids: torch.Tensor,
    compression_ratio: int,
    model_config: ModelConfig,
    device: torch.device,
) -> DynamicCache:
    """Absorber compress all documents and return a DynamicCache with RoPE applied.

    Per-block processing (matches eval.py convention):
    - Block 0: preamble_ids + doc_token_ids[0]
    - Block 1+: doc_token_ids[i]

    Args:
        model: Frozen LLM.
        doc_token_ids: List of per-doc token id tensors.
        block_lengths: Block lengths (preamble merged with first doc).
        preamble_ids: System prompt token ids.
        compression_ratio: Compression ratio.
        model_config: Model configuration.
        device: CUDA device.

    Returns:
        DynamicCache with concatenated compressed KV, RoPE applied at
        global positions [0, total_K).
    """
    all_raw_kv = []  # list of per-doc raw KV pairs

    for doc_idx in range(len(doc_token_ids)):
        if doc_idx == 0:
            # Block 0: preamble + first doc
            block_ids = torch.cat([preamble_ids, doc_token_ids[0]], dim=0)
        else:
            block_ids = doc_token_ids[doc_idx]

        raw_kv = absorber_forward_single_doc(
            model, block_ids, compression_ratio, device,
        )
        all_raw_kv.append(raw_kv)

    # Concatenate raw KV across documents per layer
    num_layers = model_config.num_layers
    kv_pairs = []
    for layer_idx in range(num_layers):
        keys = [doc_kv[layer_idx][0] for doc_kv in all_raw_kv]
        values = [doc_kv[layer_idx][1] for doc_kv in all_raw_kv]
        concat_k = torch.cat(keys, dim=2)   # [1, heads, total_K, head_dim]
        concat_v = torch.cat(values, dim=2)
        kv_pairs.append((concat_k, concat_v))

    # Build DynamicCache and apply RoPE at global positions [0, total_K)
    cache = build_dynamic_cache(kv_pairs)
    rotary_emb = model.model.rotary_emb
    cache = apply_rope_to_cache(cache, num_layers, rotary_emb)

    return cache


# ---------------------------------------------------------------------------
# CE evaluation with absorber prefix
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ce_absorber(
    model,
    doc_token_ids: list[torch.Tensor],
    block_lengths: list[int],
    preamble_ids: torch.Tensor,
    compression_ratio: int,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    model_config: ModelConfig,
    device: torch.device,
):
    """CE eval with absorber compressed prefix."""
    compressed_cache = absorber_compress_docs(
        model, doc_token_ids, block_lengths, preamble_ids,
        compression_ratio, model_config, device,
    )
    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=input_ids.to(device),
            past_key_values=compressed_cache,
            use_cache=False,
        )
    return compute_ce_loss(outputs.logits, labels.to(device))


# ---------------------------------------------------------------------------
# Generation with absorber prefix
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_absorber(
    model,
    doc_token_ids: list[torch.Tensor],
    block_lengths: list[int],
    preamble_ids: torch.Tensor,
    compression_ratio: int,
    prompt_ids: torch.Tensor,
    model_config: ModelConfig,
    device: torch.device,
    max_new_tokens: int = 64,
):
    """Generate with absorber compressed prefix."""
    compressed_cache = absorber_compress_docs(
        model, doc_token_ids, block_lengths, preamble_ids,
        compression_ratio, model_config, device,
    )
    return greedy_decode(
        model, prompt_ids, device, max_new_tokens,
        past_key_values=compressed_cache,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    compression_ratios: list[int],
    dataset_name: str = "hotpotqa",
    max_samples: int = 500,
    ce_only: bool = False,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, model_config = load_model(device)

    # Load dataset
    if dataset_name == "hotpotqa":
        logger.info("Loading HotpotQA distractor validation set...")
        raw_dataset = load_hotpotqa(max_samples, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n_samples = min(len(raw_dataset), max_samples)
    verbose = n_samples <= 10 and not ce_only
    logger.info(f"Evaluating on {n_samples} samples from {dataset_name}")

    # Condition names
    conditions = ["full_context"]
    conditions += [f"absorber_{r}x" for r in compression_ratios]
    conditions.append("no_prefix")

    # Accumulators: {condition: [ce_sum, n_tokens, f1_sum, em_sum, subspan_em_sum, gen_count]}
    accum = {c: [0.0, 0, 0.0, 0.0, 0.0, 0] for c in conditions}

    pbar = tqdm(range(n_samples), desc=f"{dataset_name} absorber eval")
    skipped = 0

    for idx in pbar:
        s = prep_hotpotqa_sample(raw_dataset[idx], tokenizer, model_config)
        if s is None:
            skipped += 1
            continue

        doc_token_ids = s["doc_token_ids"]
        block_lengths = s["block_lengths"]
        preamble_ids = s["preamble_ids"]
        comp_ids = s["comp_ids"]
        comp_labels = s["comp_labels"]
        comp_prompt_ids = s["comp_prompt_ids"]
        full_ids = s["full_ids"]
        full_labels = s["full_labels"]
        full_prompt_ids = s["full_prompt_ids"]
        gold_answer = s["gold_answer"]

        # === Full context CE ===
        try:
            result = eval_ce_full_context(model, full_ids, full_labels, device)
            if result:
                accum["full_context"][0] += result[0]
                accum["full_context"][1] += result[1]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        # === Absorber CE (per ratio) ===
        for ratio in compression_ratios:
            cond = f"absorber_{ratio}x"
            try:
                result = eval_ce_absorber(
                    model, doc_token_ids, block_lengths, preamble_ids,
                    ratio, comp_ids, comp_labels, model_config, device,
                )
                if result:
                    accum[cond][0] += result[0]
                    accum[cond][1] += result[1]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

        # === No prefix CE ===
        try:
            result = eval_ce_no_prefix(model, comp_ids, comp_labels, device)
            if result:
                accum["no_prefix"][0] += result[0]
                accum["no_prefix"][1] += result[1]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        # === Generation (F1 / EM) ===
        if not ce_only:
            sample_gens = {}

            # Full context generation
            try:
                gen_tokens = generate_full_context(model, full_prompt_ids, device)
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                accum["full_context"][2] += compute_f1(gen_text, gold_answer)
                accum["full_context"][3] += compute_em(gen_text, gold_answer)
                accum["full_context"][4] += best_subspan_em(gen_text, [gold_answer])
                accum["full_context"][5] += 1
                if verbose:
                    sample_gens["full_context"] = gen_text
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

            # Absorber generation (per ratio)
            for ratio in compression_ratios:
                cond = f"absorber_{ratio}x"
                try:
                    gen_tokens = generate_absorber(
                        model, doc_token_ids, block_lengths, preamble_ids,
                        ratio, comp_prompt_ids, model_config, device,
                    )
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    accum[cond][2] += compute_f1(gen_text, gold_answer)
                    accum[cond][3] += compute_em(gen_text, gold_answer)
                    accum[cond][4] += best_subspan_em(gen_text, [gold_answer])
                    accum[cond][5] += 1
                    if verbose:
                        sample_gens[cond] = gen_text
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

            # No prefix generation
            try:
                gen_tokens = generate_no_prefix(model, comp_prompt_ids, device)
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                accum["no_prefix"][2] += compute_f1(gen_text, gold_answer)
                accum["no_prefix"][3] += compute_em(gen_text, gold_answer)
                accum["no_prefix"][4] += best_subspan_em(gen_text, [gold_answer])
                accum["no_prefix"][5] += 1
                if verbose:
                    sample_gens["no_prefix"] = gen_text
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

            # Print per-sample generations
            if verbose and sample_gens:
                tqdm.write(f"\n--- Sample {idx} ---")
                tqdm.write(f"  Gold: {gold_answer}")
                for cond in conditions:
                    if cond in sample_gens:
                        f1 = compute_f1(sample_gens[cond], gold_answer)
                        tqdm.write(f"  {cond:<20s}: {sample_gens[cond]!r}  (F1={f1:.2f})")

        # Progress bar
        if accum["full_context"][1] > 0:
            fc_ce = accum["full_context"][0] / accum["full_context"][1]
            np_ce = accum["no_prefix"][0] / max(accum["no_prefix"][1], 1)
            postfix = {"full": f"{fc_ce:.3f}", "none": f"{np_ce:.3f}"}
            for ratio in compression_ratios[:2]:
                cond = f"absorber_{ratio}x"
                if accum[cond][1] > 0:
                    postfix[f"{ratio}x"] = f"{accum[cond][0] / accum[cond][1]:.3f}"
            pbar.set_postfix(postfix)

    # === Print results ===
    print()
    print("=" * 80)
    print(f"ABSORBER EVALUATION [{dataset_name.upper()}]  (n={n_samples}, skipped={skipped})")
    print("=" * 80)

    header = f"{'Condition':<22s} {'CE':>7s} {'PPL':>8s}"
    if not ce_only:
        header += f" {'F1':>7s} {'EM':>7s} {'Sub-EM':>7s} {'n_gen':>6s}"
    print(header)
    print("-" * len(header))

    for cond in conditions:
        ce_sum, n_tok, f1_sum, em_sum, subspan_em_sum, gen_count = accum[cond]
        if n_tok == 0:
            print(f"  {cond:<20s}  {'--':>7s} {'--':>8s}")
            continue
        ce = ce_sum / n_tok
        ppl = math.exp(min(ce, 20))
        line = f"  {cond:<20s} {ce:>7.4f} {ppl:>8.2f}"
        if not ce_only:
            if gen_count > 0:
                f1 = f1_sum / gen_count
                em = em_sum / gen_count
                sub_em = subspan_em_sum / gen_count
                line += f" {f1:>7.4f} {em:>7.4f} {sub_em:>7.4f} {gen_count:>6d}"
            else:
                line += f" {'--':>7s} {'--':>7s} {'--':>7s} {'0':>6s}"
        print(line)

    print("=" * 80)
    print()

    return {cond: accum[cond] for cond in conditions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Absorber token evaluation for KV cache compression",
    )
    parser.add_argument(
        "--dataset", type=str, default="hotpotqa", choices=["hotpotqa"],
        help="Dataset to evaluate on (default: hotpotqa)",
    )
    parser.add_argument(
        "--compression_ratios", type=int, nargs="+", default=[2, 4, 8, 16],
        help="Compression ratios to evaluate (default: 2 4 8 16)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=500,
        help="Max samples to evaluate (default: 500)",
    )
    parser.add_argument(
        "--ce_only", action="store_true",
        help="Skip generation eval (F1/EM), only compute CE/PPL",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset shuffling (default: 42)",
    )
    args = parser.parse_args()

    evaluate(
        compression_ratios=args.compression_ratios,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        ce_only=args.ce_only,
        seed=args.seed,
    )
