"""Diagnostic tests for Q-Former KV cache compression.

Loads a trained checkpoint and runs ablations to verify the compressed
KV prefix is actually contributing to the model's predictions:

1. Normal:        Q-Former compressed prefix (should match training CE)
2. No prefix:     No KV cache at all (is the LLM answering from parametric knowledge?)
3. Zero prefix:   Zero-valued KV cache, same shape (is the LLM ignoring the prefix?)
4. Random prefix: Random KV cache, same shape (is any prefix equally good?)
5. Shuffled prefix: Layer-shuffled Q-Former output (is per-layer specialization real?)

If tests 2-5 all score similarly to test 1, the Q-Former isn't helping.
If test 1 is clearly better, the compression is working.
"""

import argparse
import logging
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from block_attention import build_block_causal_mask_with_qa
from collator import RAGCollator
from config import ModelConfig, QFormerConfig, TrainingConfig
from dataset import RAGDataset
from kv_cache_utils import (
    apply_rope_to_cache,
    concat_compressed_caches,
    extract_doc_hidden_states,
)
from qformer import QFormerKVCompressor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, device):
    """Load model, tokenizer, and trained Q-Former from a checkpoint."""
    model_config = ModelConfig()
    qformer_config = QFormerConfig()

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    qformer = QFormerKVCompressor(qformer_config, model_config, llm=model).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Strip _orig_mod. prefixes from torch.compile'd checkpoints
    state_dict = ckpt["qformer_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    qformer.load_state_dict(state_dict, strict=False)
    qformer.eval()

    compression_ratio = ckpt.get("compression_ratio", 4)
    step = ckpt.get("step", -1)
    logger.info(
        f"Loaded checkpoint from step {step}, compression_ratio={compression_ratio}"
    )

    return model, tokenizer, qformer, model_config, compression_ratio


@torch.no_grad()
def run_stage_a(model, doc_token_ids, block_lengths, qa_input_ids,
                preamble_ids, model_config, device):
    """Stage A: extract per-block hidden states and teacher logits."""
    doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(device)
    preamble = preamble_ids.unsqueeze(0).to(device)
    full_input = torch.cat([preamble, doc_concat, qa_input_ids], dim=1)

    qa_length = qa_input_ids.shape[1]
    attn_mask = build_block_causal_mask_with_qa(
        block_lengths, qa_length, dtype=torch.float16, device=device
    )

    outputs = model(
        input_ids=full_input,
        attention_mask=attn_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    per_doc_hidden = extract_doc_hidden_states(
        outputs.hidden_states, block_lengths, model_config.num_layers
    )

    doc_total = sum(block_lengths)
    teacher_logits = outputs.logits[:, doc_total:, :]

    return per_doc_hidden, teacher_logits


@torch.no_grad()
def compress_docs(qformer, doc_hidden_states, compression_ratio, model, model_config):
    """Run Q-Former compression and apply RoPE. Returns (cache, prefix_len)."""
    per_doc_compressed = []
    with torch.amp.autocast("cuda"):
        for doc_hs in doc_hidden_states:
            compressed = qformer(doc_hs, compression_ratio)
            per_doc_compressed.append(compressed)

    compressed_cache = concat_compressed_caches(
        per_doc_compressed, model_config.num_layers
    )

    rotary_emb = model.model.rotary_emb
    compressed_cache = apply_rope_to_cache(
        compressed_cache, model_config.num_layers, rotary_emb
    )

    prefix_len = compressed_cache.get_seq_length()
    return compressed_cache, prefix_len


def compute_ce(logits, labels):
    """Compute CE loss on answer tokens. Returns (loss, n_tokens) or None."""
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


def forward_with_cache(model, input_ids, labels, past_key_values):
    """Run LLM forward with a given KV cache prefix."""
    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=False,
        )
    return compute_ce(outputs.logits, labels)


def forward_no_prefix(model, input_ids, labels):
    """Run LLM forward with no prefix at all."""
    with torch.amp.autocast("cuda"):
        outputs = model(input_ids=input_ids, use_cache=False)
    return compute_ce(outputs.logits, labels)


def make_zero_cache(compressed_cache, num_layers):
    """Replace all K/V in the cache with zeros."""
    from kv_cache_utils import build_dynamic_cache

    pairs = []
    for layer_idx in range(num_layers):
        k = compressed_cache.layers[layer_idx].keys
        v = compressed_cache.layers[layer_idx].values
        pairs.append((torch.zeros_like(k), torch.zeros_like(v)))
    return build_dynamic_cache(pairs)


def make_random_cache(compressed_cache, num_layers):
    """Replace all K/V in the cache with random values matching the scale."""
    from kv_cache_utils import build_dynamic_cache

    pairs = []
    for layer_idx in range(num_layers):
        k = compressed_cache.layers[layer_idx].keys
        v = compressed_cache.layers[layer_idx].values
        pairs.append((
            torch.randn_like(k) * k.std(),
            torch.randn_like(v) * v.std(),
        ))
    return build_dynamic_cache(pairs)


def make_shuffled_cache(compressed_cache, num_layers):
    """Shuffle the layer assignment of the Q-Former output KVs."""
    from kv_cache_utils import build_dynamic_cache

    perm = torch.randperm(num_layers)
    pairs = []
    for layer_idx in range(num_layers):
        src = perm[layer_idx].item()
        k = compressed_cache.layers[src].keys
        v = compressed_cache.layers[src].values
        pairs.append((k.clone(), v.clone()))
    return build_dynamic_cache(pairs)


@torch.no_grad()
def run_tests(
    checkpoint_path: str,
    max_samples: int = 200,
    compression_ratio_override: int | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, qformer, model_config, ckpt_compression_ratio = load_checkpoint(
        checkpoint_path, device
    )
    compression_ratio = compression_ratio_override or ckpt_compression_ratio

    training_config = TrainingConfig()
    dataset = RAGDataset(
        tokenizer=tokenizer,
        model_config=model_config,
        split="eval",
        eval_split_ratio=training_config.eval_split_ratio,
        seed=training_config.seed,
    )
    collator = RAGCollator(tokenizer)

    # Accumulators: {test_name: (loss_sum, n_tokens)}
    tests = ["normal", "no_prefix", "zero_prefix", "random_prefix", "shuffled_layers"]
    accum = {t: [0.0, 0] for t in tests}

    n = min(len(dataset), max_samples)
    pbar = tqdm(range(n), desc=f"Diagnostics @ {compression_ratio}x")

    for idx in pbar:
        item = dataset[idx]
        batch = collator([item])

        doc_token_ids = batch["doc_token_ids"]
        doc_lengths = batch["doc_lengths"]
        stage_b_input_ids = batch["stage_b_input_ids"].to(device)
        stage_b_labels = batch["stage_b_labels"].to(device)

        if not doc_token_ids or sum(doc_lengths) == 0:
            continue

        # Block lengths: preamble merges with first doc
        preamble_ids = batch["preamble_ids"]
        preamble_len = preamble_ids.shape[0]
        block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

        # Stage A (teacher_logits unused here but available for future KL tests)
        doc_hidden_states, _ = run_stage_a(
            model, doc_token_ids, block_lengths, stage_b_input_ids,
            preamble_ids, model_config, device
        )

        # Compress with trained Q-Former
        compressed_cache, _ = compress_docs(
            qformer, doc_hidden_states, compression_ratio, model, model_config
        )

        # --- Test 1: Normal (trained Q-Former) ---
        result = forward_with_cache(model, stage_b_input_ids, stage_b_labels, compressed_cache)
        if result is None:
            continue
        accum["normal"][0] += result[0]
        accum["normal"][1] += result[1]

        # --- Test 2: No prefix at all ---
        result = forward_no_prefix(model, stage_b_input_ids, stage_b_labels)
        if result:
            accum["no_prefix"][0] += result[0]
            accum["no_prefix"][1] += result[1]

        # --- Test 3: Zero prefix ---
        zero_cache = make_zero_cache(compressed_cache, model_config.num_layers)
        result = forward_with_cache(model, stage_b_input_ids, stage_b_labels, zero_cache)
        if result:
            accum["zero_prefix"][0] += result[0]
            accum["zero_prefix"][1] += result[1]

        # --- Test 4: Random prefix ---
        random_cache = make_random_cache(compressed_cache, model_config.num_layers)
        result = forward_with_cache(model, stage_b_input_ids, stage_b_labels, random_cache)
        if result:
            accum["random_prefix"][0] += result[0]
            accum["random_prefix"][1] += result[1]

        # --- Test 5: Shuffled layers ---
        shuffled_cache = make_shuffled_cache(compressed_cache, model_config.num_layers)
        result = forward_with_cache(model, stage_b_input_ids, stage_b_labels, shuffled_cache)
        if result:
            accum["shuffled_layers"][0] += result[0]
            accum["shuffled_layers"][1] += result[1]

        # Update progress bar with running normal CE
        if accum["normal"][1] > 0:
            avg_normal = accum["normal"][0] / accum["normal"][1]
            avg_none = accum["no_prefix"][0] / max(accum["no_prefix"][1], 1)
            pbar.set_postfix(
                normal=f"{avg_normal:.3f}",
                no_pfx=f"{avg_none:.3f}",
            )

    # === Results ===
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"DIAGNOSTIC RESULTS  (compression={compression_ratio}x, n={n})")
    logger.info("=" * 70)

    normal_ce = accum["normal"][0] / max(accum["normal"][1], 1)

    labels = {
        "normal": "Normal (Q-Former)",
        "no_prefix": "No prefix",
        "zero_prefix": "Zero prefix",
        "random_prefix": "Random prefix",
        "shuffled_layers": "Shuffled layers",
    }

    for test in tests:
        loss_sum, n_tok = accum[test]
        if n_tok == 0:
            logger.info(f"  {labels[test]:20s}  --no valid tokens--")
            continue
        ce = loss_sum / n_tok
        ppl = math.exp(ce)
        delta = ce - normal_ce
        sign = "+" if delta >= 0 else ""
        logger.info(
            f"  {labels[test]:20s}  CE={ce:.4f}  PPL={ppl:.2f}  ({sign}{delta:.4f})"
        )

    logger.info("=" * 70)
    logger.info("")

    # Interpretation
    no_prefix_ce = accum["no_prefix"][0] / max(accum["no_prefix"][1], 1)
    gap = no_prefix_ce - normal_ce

    if gap < 0.05:
        logger.warning(
            "!! Normal and no-prefix CE are nearly identical (gap=%.4f). "
            "The LLM is likely answering from parametric knowledge or the "
            "question alone. The compressed prefix is NOT contributing.", gap
        )
    elif gap < 0.2:
        logger.warning(
            "Marginal gap (%.4f) between normal and no-prefix. "
            "The prefix is helping slightly but the task may be too easy.", gap
        )
    else:
        logger.info(
            "Good gap (%.4f) between normal and no-prefix. "
            "The compressed prefix is contributing meaningful information.", gap
        )

    random_ce = accum["random_prefix"][0] / max(accum["random_prefix"][1], 1)
    if abs(random_ce - normal_ce) < 0.05:
        logger.warning(
            "!! Random prefix scores nearly the same as trained Q-Former. "
            "The model may not be using the prefix content."
        )

    return {t: accum[t][0] / max(accum[t][1], 1) for t in tests}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic tests for Q-Former compression")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file (e.g., outputs/checkpoint-500/checkpoint.pt)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Max eval samples (default: 200)",
    )
    parser.add_argument(
        "--compression_ratio",
        type=int,
        default=None,
        help="Override compression ratio from checkpoint",
    )
    args = parser.parse_args()

    run_tests(
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        compression_ratio_override=args.compression_ratio,
    )
