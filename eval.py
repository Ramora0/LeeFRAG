"""Benchmark evaluation for Q-Former KV cache compression.

Supports two datasets:
- hotpotqa: HotpotQA distractor (10 paragraphs, multi-hop, short-answer)
- rag_v1: RAG-v1 eval split (same data pipeline as test.py, for sanity checks)

Conditions:
1. Full context — all docs inline, standard causal attention (ceiling)
2. Compressed Nx — Q-Former compressed KV prefix at each ratio
3. No prefix — question only, no documents (floor)

Metrics:
- CE / PPL — cross-entropy and perplexity on answer tokens (teacher-forced)
- F1 / EM — generation quality scored against gold answers (SQuAD-style)
"""

import argparse
import collections
import logging
import math
import re
import string

import torch
import torch.nn.functional as F
from datasets import load_dataset
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

SYSTEM_PROMPT = "Answer the question based on the provided documents. Give a short, direct answer."


# ---------------------------------------------------------------------------
# SQuAD-style answer normalization and scoring
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not gt_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hotpotqa(max_samples: int, seed: int = 42):
    """Load HotpotQA distractor validation set."""
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=seed)
    if max_samples < len(ds):
        ds = ds.select(range(max_samples))
    return ds


def format_paragraphs(sample: dict) -> tuple[list[str], str, str]:
    """Extract documents, question, and answer from a HotpotQA sample.

    Returns:
        doc_texts: List of paragraph strings (title + sentences).
        question: The question string.
        answer: The gold answer string.
    """
    titles = sample["context"]["title"]
    sentences_list = sample["context"]["sentences"]

    doc_texts = []
    for title, sentences in zip(titles, sentences_list):
        text = f"{title}: {''.join(sentences)}"
        doc_texts.append(text)

    return doc_texts, sample["question"], sample["answer"]


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_documents(doc_texts, tokenizer, model_config):
    """Tokenize documents with per-doc and total limits (same as training)."""
    doc_token_ids = []
    total_tokens = 0
    for doc in doc_texts:
        ids = tokenizer.encode(
            doc, add_special_tokens=False,
            max_length=model_config.max_doc_tokens, truncation=True,
        )
        if total_tokens + len(ids) > model_config.max_total_doc_tokens:
            remaining = model_config.max_total_doc_tokens - total_tokens
            if remaining > 0:
                ids = ids[:remaining]
            else:
                break
        doc_token_ids.append(torch.tensor(ids, dtype=torch.long))
        total_tokens += len(ids)
    return doc_token_ids


def build_chat_tokens(tokenizer, system_prompt, user_content, answer=None):
    """Build chat-templated token ids.

    Returns:
        input_ids: [1, seq_len]
        labels: [1, seq_len] with -100 on non-answer tokens (None if no answer)
        prompt_len: length of prompt portion (for generation)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    if answer is not None:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False,
                                      max_length=512, truncation=True)
        full_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        labels = torch.tensor([labels], dtype=torch.long)
        return input_ids, labels, len(prompt_ids)
    else:
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        return input_ids, None, len(prompt_ids)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device):
    """Load frozen model and tokenizer (no Q-Former)."""
    model_config = ModelConfig()

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    return model, tokenizer, model_config


def load_checkpoint(checkpoint_path, device):
    """Load model, tokenizer, and trained Q-Former from a checkpoint."""
    model, tokenizer, model_config = load_model(device)
    qformer_config = QFormerConfig()

    qformer = QFormerKVCompressor(qformer_config, model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    state_dict = ckpt["qformer_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    qformer.load_state_dict(state_dict)
    qformer.eval()

    step = ckpt.get("step", -1)
    ckpt_ratio = ckpt.get("compression_ratio", 4)
    logger.info(f"Loaded checkpoint from step {step}, compression_ratio={ckpt_ratio}")

    return model, tokenizer, qformer, model_config


# ---------------------------------------------------------------------------
# Stage A: extract per-doc hidden states
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_stage_a(model, doc_token_ids, doc_lengths, qa_input_ids, model_config, device):
    """Run frozen LLM on [docs | Q+A] with block-diagonal mask; extract hidden states."""
    doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(device)
    full_input = torch.cat([doc_concat, qa_input_ids], dim=1)

    qa_length = qa_input_ids.shape[1]
    attn_mask = build_block_causal_mask_with_qa(
        doc_lengths, qa_length, dtype=torch.float16, device=device,
    )

    outputs = model(
        input_ids=full_input, attention_mask=attn_mask,
        output_hidden_states=True, use_cache=False,
    )

    per_doc_hidden = extract_doc_hidden_states(
        outputs.hidden_states, doc_lengths, model_config.num_layers,
    )
    return per_doc_hidden


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

@torch.no_grad()
def compress_docs(qformer, doc_hidden_states, compression_ratio, model, model_config):
    """Q-Former compress + RoPE. Returns a fresh DynamicCache."""
    per_doc_compressed = []
    with torch.amp.autocast("cuda"):
        for doc_hs in doc_hidden_states:
            compressed = qformer(doc_hs, compression_ratio)
            per_doc_compressed.append(compressed)

    compressed_cache = concat_compressed_caches(
        per_doc_compressed, model_config.num_layers,
    )
    rotary_emb = model.model.rotary_emb
    compressed_cache = apply_rope_to_cache(
        compressed_cache, model_config.num_layers, rotary_emb,
    )
    return compressed_cache


# ---------------------------------------------------------------------------
# CE evaluation
# ---------------------------------------------------------------------------

def compute_ce_loss(logits, labels):
    """Compute CE on answer tokens. Returns (loss_sum, n_tokens) or None."""
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


@torch.no_grad()
def eval_ce_compressed(model, qformer, doc_hidden_states, compression_ratio,
                       input_ids, labels, model_config, device):
    """CE eval with compressed prefix."""
    compressed_cache = compress_docs(
        qformer, doc_hidden_states, compression_ratio, model, model_config,
    )
    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=input_ids.to(device),
            past_key_values=compressed_cache,
            use_cache=False,
        )
    return compute_ce_loss(outputs.logits, labels.to(device))


@torch.no_grad()
def eval_ce_full_context(model, full_input_ids, full_labels, device):
    """CE eval with full inline context (no compression)."""
    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=full_input_ids.to(device), use_cache=False,
        )
    return compute_ce_loss(outputs.logits, full_labels.to(device))


@torch.no_grad()
def eval_ce_no_prefix(model, input_ids, labels, device):
    """CE eval with no documents at all."""
    with torch.amp.autocast("cuda"):
        outputs = model(input_ids=input_ids.to(device), use_cache=False)
    return compute_ce_loss(outputs.logits, labels.to(device))


# ---------------------------------------------------------------------------
# Generation evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(model, input_ids, device, max_new_tokens=64, past_key_values=None):
    """Manual greedy decode loop using model() forward passes.

    model.generate() breaks with synthetic past_key_values (its internal
    cache_position tracking assumes it created the cache). This uses the
    same forward call that works in training and CE eval.
    """
    cur_ids = input_ids.to(device)
    cache = past_key_values
    generated = []

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=cur_ids,
                past_key_values=cache,
                use_cache=True,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if next_token.item() == model.config.eos_token_id:
            break
        cur_ids = next_token
        cache = outputs.past_key_values

    if generated:
        return torch.cat(generated, dim=-1)[0]
    return torch.tensor([], dtype=torch.long, device=device)


@torch.no_grad()
def generate_compressed(model, qformer, doc_hidden_states, compression_ratio,
                        prompt_ids, model_config, device, max_new_tokens=64):
    """Generate with compressed prefix."""
    compressed_cache = compress_docs(
        qformer, doc_hidden_states, compression_ratio, model, model_config,
    )
    return greedy_decode(model, prompt_ids, device, max_new_tokens,
                         past_key_values=compressed_cache)


@torch.no_grad()
def generate_full_context(model, full_prompt_ids, device, max_new_tokens=64):
    """Generate with full inline context."""
    return greedy_decode(model, full_prompt_ids, device, max_new_tokens)


@torch.no_grad()
def generate_no_prefix(model, prompt_ids, device, max_new_tokens=64):
    """Generate with no documents."""
    return greedy_decode(model, prompt_ids, device, max_new_tokens)


@torch.no_grad()
def eval_ce_block_context(model, doc_token_ids, doc_lengths, comp_ids, comp_labels, device):
    """CE eval with block-diagonal attention (docs isolated, Q+A attends to all)."""
    doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(device)
    full_input = torch.cat([doc_concat, comp_ids.to(device)], dim=1)

    qa_length = comp_ids.shape[1]
    attn_mask = build_block_causal_mask_with_qa(
        doc_lengths, qa_length, dtype=torch.float16, device=device,
    )

    doc_total = sum(doc_lengths)
    doc_labels = torch.full((1, doc_total), -100, dtype=torch.long, device=device)
    full_labels = torch.cat([doc_labels, comp_labels.to(device)], dim=1)

    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=full_input, attention_mask=attn_mask, use_cache=False,
        )
    return compute_ce_loss(outputs.logits, full_labels)


@torch.no_grad()
def generate_block_context(model, doc_token_ids, doc_lengths, comp_prompt_ids,
                           device, max_new_tokens=64):
    """Generate with block-diagonal attention for docs, then greedy decode."""
    doc_concat = torch.cat(doc_token_ids, dim=0).unsqueeze(0).to(device)
    full_input = torch.cat([doc_concat, comp_prompt_ids.to(device)], dim=1)

    qa_length = comp_prompt_ids.shape[1]
    attn_mask = build_block_causal_mask_with_qa(
        doc_lengths, qa_length, dtype=torch.float16, device=device,
    )

    # Encode with block mask, then greedy decode from the KV cache
    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=full_input, attention_mask=attn_mask, use_cache=True,
        )

    cache = outputs.past_key_values
    generated = []
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token)

    for _ in range(max_new_tokens - 1):
        if next_token.item() == model.config.eos_token_id:
            break
        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=next_token, past_key_values=cache, use_cache=True,
            )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        cache = outputs.past_key_values

    if generated:
        return torch.cat(generated, dim=-1)[0]
    return torch.tensor([], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Per-sample data prep (unified across datasets)
# ---------------------------------------------------------------------------

def prep_hotpotqa_sample(sample, tokenizer, model_config):
    """Prepare a HotpotQA sample. Returns dict or None on failure."""
    try:
        doc_texts, question, gold_answer = format_paragraphs(sample)
    except (KeyError, IndexError):
        return None

    doc_token_ids = tokenize_documents(doc_texts, tokenizer, model_config)
    if not doc_token_ids:
        return None

    # Compressed / no-prefix: question only in user message
    user_msg_compressed = f"Question: {question}"
    comp_ids, comp_labels, _ = build_chat_tokens(
        tokenizer, SYSTEM_PROMPT, user_msg_compressed, answer=gold_answer,
    )
    comp_prompt_ids, _, _ = build_chat_tokens(
        tokenizer, SYSTEM_PROMPT, user_msg_compressed, answer=None,
    )

    # Full context: docs + question in user message
    docs_text = "\n\n".join(doc_texts)
    user_msg_full = f"{docs_text}\n\nQuestion: {question}"
    full_ids, full_labels, _ = build_chat_tokens(
        tokenizer, SYSTEM_PROMPT, user_msg_full, answer=gold_answer,
    )
    full_prompt_ids, _, _ = build_chat_tokens(
        tokenizer, SYSTEM_PROMPT, user_msg_full, answer=None,
    )

    return {
        "doc_token_ids": doc_token_ids,
        "doc_lengths": [t.shape[0] for t in doc_token_ids],
        "comp_ids": comp_ids,
        "comp_labels": comp_labels,
        "comp_prompt_ids": comp_prompt_ids,
        "full_ids": full_ids,
        "full_labels": full_labels,
        "full_prompt_ids": full_prompt_ids,
        "gold_answer": gold_answer,
    }


def prep_rag_v1_sample(item, collator, tokenizer, model_config):
    """Prepare a RAG-v1 sample using the training collator (matches test.py exactly)."""
    batch = collator([item])

    doc_token_ids = batch["doc_token_ids"]
    if not doc_token_ids or sum(batch["doc_lengths"]) == 0:
        return None

    # comp_ids / comp_labels from collator — identical to test.py
    comp_ids = batch["stage_b_input_ids"]
    comp_labels = batch["stage_b_labels"]

    # Prompt-only for generation: strip answer tokens from stage_b
    answer_start = batch["answer_start"]
    comp_prompt_ids = comp_ids[:, :answer_start]

    # Full context: docs inline + question in user message
    doc_texts = item["doc_texts"]
    docs_text = "\n\n".join(doc_texts)
    system_prompt = item["system_prompt"]
    question_suffix = item["question_suffix"]
    user_msg_full = f"{docs_text}\n\n{question_suffix}"
    gold_answer = item["answer"]

    full_ids, full_labels, _ = build_chat_tokens(
        tokenizer, system_prompt, user_msg_full, answer=gold_answer,
    )
    full_prompt_ids, _, _ = build_chat_tokens(
        tokenizer, system_prompt, user_msg_full, answer=None,
    )

    return {
        "doc_token_ids": doc_token_ids,
        "doc_lengths": batch["doc_lengths"],
        "comp_ids": comp_ids,
        "comp_labels": comp_labels,
        "comp_prompt_ids": comp_prompt_ids,
        "full_ids": full_ids,
        "full_labels": full_labels,
        "full_prompt_ids": full_prompt_ids,
        "gold_answer": gold_answer,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    checkpoint_path: str | None,
    compression_ratios: list[int],
    dataset_name: str = "hotpotqa",
    max_samples: int = 500,
    ce_only: bool = False,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path is not None:
        model, tokenizer, qformer, model_config = load_checkpoint(checkpoint_path, device)
    else:
        model, tokenizer, model_config = load_model(device)
        qformer = None
        compression_ratios = []
        logger.info("No checkpoint — running full_context, block_context, and no_prefix")

    # Load dataset
    if dataset_name == "hotpotqa":
        logger.info("Loading HotpotQA distractor validation set...")
        raw_dataset = load_hotpotqa(max_samples, seed=seed)
        collator = None
    elif dataset_name == "rag_v1":
        logger.info("Loading RAG-v1 eval split (same as test.py)...")
        training_config = TrainingConfig()
        raw_dataset = RAGDataset(
            tokenizer=tokenizer,
            model_config=model_config,
            split="eval",
            eval_split_ratio=training_config.eval_split_ratio,
            seed=training_config.seed,
        )
        collator = RAGCollator(tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n_samples = min(len(raw_dataset), max_samples)
    verbose = n_samples <= 10 and not ce_only
    logger.info(f"Evaluating on {n_samples} samples from {dataset_name}")

    # Condition names: full_context, block_context (raw model only), compressed_Nx, no_prefix
    conditions = ["full_context"]
    if qformer is None:
        conditions.append("block_context")
    conditions += [f"compressed_{r}x" for r in compression_ratios]
    conditions.append("no_prefix")

    # Accumulators: {condition: [ce_sum, n_tokens, f1_sum, em_sum, gen_count]}
    accum = {c: [0.0, 0, 0.0, 0.0, 0] for c in conditions}

    pbar = tqdm(range(n_samples), desc=f"{dataset_name} eval")
    skipped = 0

    for idx in pbar:
        # Prep sample (dataset-specific)
        if dataset_name == "hotpotqa":
            s = prep_hotpotqa_sample(raw_dataset[idx], tokenizer, model_config)
        else:
            s = prep_rag_v1_sample(raw_dataset[idx], collator, tokenizer, model_config)

        if s is None:
            skipped += 1
            continue

        doc_token_ids = s["doc_token_ids"]
        doc_lengths = s["doc_lengths"]
        comp_ids = s["comp_ids"]
        comp_labels = s["comp_labels"]
        comp_prompt_ids = s["comp_prompt_ids"]
        full_ids = s["full_ids"]
        full_labels = s["full_labels"]
        full_prompt_ids = s["full_prompt_ids"]
        gold_answer = s["gold_answer"]

        # --- Stage A (once per sample, only if Q-Former is loaded) ---
        per_doc_hidden = None
        if qformer is not None:
            try:
                per_doc_hidden = run_stage_a(
                    model, doc_token_ids, doc_lengths,
                    comp_ids.to(device), model_config, device,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1
                continue

        # === Full context CE ===
        try:
            result = eval_ce_full_context(model, full_ids, full_labels, device)
            if result:
                accum["full_context"][0] += result[0]
                accum["full_context"][1] += result[1]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        # === Block context CE (raw model only) ===
        if qformer is None:
            try:
                result = eval_ce_block_context(
                    model, doc_token_ids, doc_lengths, comp_ids, comp_labels, device,
                )
                if result:
                    accum["block_context"][0] += result[0]
                    accum["block_context"][1] += result[1]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

        # === Compressed CE (per ratio) ===
        for ratio in compression_ratios:
            cond = f"compressed_{ratio}x"
            try:
                result = eval_ce_compressed(
                    model, qformer, per_doc_hidden, ratio,
                    comp_ids, comp_labels, model_config, device,
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
            sample_gens = {}  # {condition: gen_text} for verbose printing

            # Full context generation
            try:
                gen_tokens = generate_full_context(model, full_prompt_ids, device)
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                accum["full_context"][2] += compute_f1(gen_text, gold_answer)
                accum["full_context"][3] += compute_em(gen_text, gold_answer)
                accum["full_context"][4] += 1
                if verbose:
                    sample_gens["full_context"] = gen_text
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

            # Block context generation (raw model only)
            if qformer is None:
                try:
                    gen_tokens = generate_block_context(
                        model, doc_token_ids, doc_lengths, comp_prompt_ids, device,
                    )
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    accum["block_context"][2] += compute_f1(gen_text, gold_answer)
                    accum["block_context"][3] += compute_em(gen_text, gold_answer)
                    accum["block_context"][4] += 1
                    if verbose:
                        sample_gens["block_context"] = gen_text
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()

            # Compressed generation (per ratio, only if Q-Former loaded)
            if qformer is not None:
                for ratio in compression_ratios:
                    cond = f"compressed_{ratio}x"
                    try:
                        gen_tokens = generate_compressed(
                            model, qformer, per_doc_hidden, ratio,
                            comp_prompt_ids, model_config, device,
                        )
                        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        accum[cond][2] += compute_f1(gen_text, gold_answer)
                        accum[cond][3] += compute_em(gen_text, gold_answer)
                        accum[cond][4] += 1
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
                accum["no_prefix"][4] += 1
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
            if "block_context" in accum and accum["block_context"][1] > 0:
                postfix["block"] = f"{accum['block_context'][0] / accum['block_context'][1]:.3f}"
            for ratio in compression_ratios[:2]:
                cond = f"compressed_{ratio}x"
                if accum[cond][1] > 0:
                    postfix[f"{ratio}x"] = f"{accum[cond][0] / accum[cond][1]:.3f}"
            pbar.set_postfix(postfix)

    # === Print results ===
    print()
    print("=" * 80)
    print(f"EVALUATION [{dataset_name.upper()}]  (n={n_samples}, skipped={skipped})")
    print("=" * 80)

    header = f"{'Condition':<22s} {'CE':>7s} {'PPL':>8s}"
    if not ce_only:
        header += f" {'F1':>7s} {'EM':>7s} {'n_gen':>6s}"
    print(header)
    print("-" * len(header))

    for cond in conditions:
        ce_sum, n_tok, f1_sum, em_sum, gen_count = accum[cond]
        if n_tok == 0:
            print(f"  {cond:<20s}  {'--':>7s} {'--':>8s}")
            continue
        ce = ce_sum / n_tok
        ppl = math.exp(min(ce, 20))  # cap to avoid overflow
        line = f"  {cond:<20s} {ce:>7.4f} {ppl:>8.2f}"
        if not ce_only:
            if gen_count > 0:
                f1 = f1_sum / gen_count
                em = em_sum / gen_count
                line += f" {f1:>7.4f} {em:>7.4f} {gen_count:>6d}"
            else:
                line += f" {'--':>7s} {'--':>7s} {'0':>6s}"
        print(line)

    print("=" * 80)
    print()

    return {cond: accum[cond] for cond in conditions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for Q-Former compression",
    )
    parser.add_argument(
        "checkpoint", type=str, nargs="?", default=None,
        help="Path to checkpoint (e.g., outputs/checkpoint-500/checkpoint.pt). "
             "Omit to run full_context + no_prefix only (no compression).",
    )
    parser.add_argument(
        "--dataset", type=str, default="hotpotqa", choices=["hotpotqa", "rag_v1"],
        help="Dataset to evaluate on (default: hotpotqa). "
             "Use rag_v1 to sanity-check against test.py numbers.",
    )
    parser.add_argument(
        "--compression_ratios", type=int, nargs="+", default=[4, 8, 16, 32],
        help="Compression ratios to evaluate (default: 4 8 16 32)",
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
        checkpoint_path=args.checkpoint,
        compression_ratios=args.compression_ratios,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        ce_only=args.ce_only,
        seed=args.seed,
    )
