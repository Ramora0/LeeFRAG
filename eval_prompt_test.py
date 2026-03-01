"""Prompt-format ablation: HotpotQA data with RAG-v1 formatting.

Tests whether HotpotQA CE degradation is due to prompt mismatch by
formatting HotpotQA documents/questions to match RAG-v1 exactly:
- "Document:N\nTitle: ...\nText: ..." prefix on each document
- RAG-v1 system prompt (with citation instructions)
- "Answer Mode: Grounded\n\nQuestion: {q}" suffix format
- Same RAGCollator for Stage B token building

If compressed CE here matches eval.py --dataset rag_v1 quality,
the issue is prompt format. If it's still bad, it's domain/content.
"""

import argparse
import logging
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from collator import RAGCollator
from config import ModelConfig, QFormerConfig
from eval import (
    compute_ce_loss,
    compress_docs,
    load_checkpoint,
    run_stage_a,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RAG_V1_SYSTEM_PROMPT = (
    "You are a conversational AI assistant that is provided a list of documents "
    "and a user query to answer based on information from the documents. The user "
    "also provides an answer mode which can be 'Grounded' or 'Mixed'. For answer "
    "mode Grounded only respond with exact facts from documents, for answer mode "
    "Mixed answer using facts from documents and your own knowledge. Cite all "
    "facts from the documents using <co: doc_id></co> tags."
)


def load_hotpotqa(max_samples, seed=42):
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=seed)
    if max_samples < len(ds):
        ds = ds.select(range(max_samples))
    return ds


def format_as_rag_v1(sample, tokenizer, model_config):
    """Format a HotpotQA sample to look exactly like a RAGDataset item."""
    titles = sample["context"]["title"]
    sentences_list = sample["context"]["sentences"]

    # Format docs as "Document:N\nTitle: ...\nText: ..."
    doc_texts = []
    for i, (title, sentences) in enumerate(zip(titles, sentences_list)):
        text = "".join(sentences)
        doc_texts.append(f"Document:{i}\nTitle: {title}\nText: {text}")

    # Tokenize documents (same limits as training)
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

    if not doc_token_ids:
        return None

    # Build question_suffix in RAG-v1 format
    question = sample["question"]
    question_suffix = f"\n\nAnswer Mode: Grounded\n\nQuestion: {question}"

    # Tokenize answer
    answer = sample["answer"]
    answer_ids = tokenizer.encode(
        answer, add_special_tokens=False,
        max_length=model_config.max_answer_tokens, truncation=True,
    )

    # Build preamble: <|system|>\n{system_prompt}\n\n
    preamble_text = f"<|system|>\n{RAG_V1_SYSTEM_PROMPT}\n\n"
    preamble_ids = tokenizer.encode(preamble_text, add_special_tokens=False)

    # Build QA suffix: <|user|>\n{question_suffix}\n<|assistant|>\n
    qa_suffix_text = f"<|user|>\n{question_suffix}\n<|assistant|>\n"
    qa_suffix_ids = tokenizer.encode(qa_suffix_text, add_special_tokens=False)

    # Return dict matching RAGDataset.__getitem__ output
    return {
        "doc_texts": doc_texts,
        "doc_token_ids": doc_token_ids,
        "preamble_ids": torch.tensor(preamble_ids, dtype=torch.long),
        "qa_suffix_ids": torch.tensor(qa_suffix_ids, dtype=torch.long),
        "question_suffix": question_suffix,
        "answer": answer,
        "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
        "system_prompt": RAG_V1_SYSTEM_PROMPT,
    }


@torch.no_grad()
def run(checkpoint_path, compression_ratios, max_samples=100, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, qformer, model_config = load_checkpoint(checkpoint_path, device)
    collator = RAGCollator(tokenizer)

    dataset = load_hotpotqa(max_samples, seed=seed)
    logger.info(f"Evaluating {len(dataset)} HotpotQA samples with RAG-v1 formatting")

    conditions = [f"compressed_{r}x" for r in compression_ratios] + ["no_prefix"]
    accum = {c: [0.0, 0] for c in conditions}
    skipped = 0

    pbar = tqdm(range(len(dataset)), desc="Prompt-format ablation")

    for idx in pbar:
        item = format_as_rag_v1(dataset[idx], tokenizer, model_config)
        if item is None:
            skipped += 1
            continue

        # Use RAGCollator (identical to test.py path)
        batch = collator([item])
        doc_token_ids = batch["doc_token_ids"]
        doc_lengths = batch["doc_lengths"]
        preamble_ids = batch["preamble_ids"]
        stage_b_input_ids = batch["stage_b_input_ids"].to(device)
        stage_b_labels = batch["stage_b_labels"].to(device)

        if not doc_token_ids or sum(doc_lengths) == 0:
            skipped += 1
            continue

        # Block lengths: preamble merges with first doc
        preamble_len = preamble_ids.shape[0]
        block_lengths = [preamble_len + doc_lengths[0]] + doc_lengths[1:]

        # Stage A
        try:
            per_doc_hidden = run_stage_a(
                model, doc_token_ids, block_lengths,
                stage_b_input_ids, preamble_ids, model_config, device,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            skipped += 1
            continue

        # Compressed CE (per ratio)
        for ratio in compression_ratios:
            cond = f"compressed_{ratio}x"
            try:
                compressed_cache = compress_docs(
                    qformer, per_doc_hidden, ratio, model, model_config,
                )
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=stage_b_input_ids,
                        past_key_values=compressed_cache,
                        use_cache=False,
                    )
                result = compute_ce_loss(outputs.logits, stage_b_labels)
                if result:
                    accum[cond][0] += result[0]
                    accum[cond][1] += result[1]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

        # No prefix CE
        try:
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=stage_b_input_ids, use_cache=False,
                )
            result = compute_ce_loss(outputs.logits, stage_b_labels)
            if result:
                accum["no_prefix"][0] += result[0]
                accum["no_prefix"][1] += result[1]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        # Progress
        postfix = {}
        for ratio in compression_ratios[:2]:
            cond = f"compressed_{ratio}x"
            if accum[cond][1] > 0:
                postfix[f"{ratio}x"] = f"{accum[cond][0] / accum[cond][1]:.3f}"
        np_n = accum["no_prefix"][1]
        if np_n > 0:
            postfix["none"] = f"{accum['no_prefix'][0] / np_n:.3f}"
        pbar.set_postfix(postfix)

    # Results
    print()
    print("=" * 70)
    print(f"PROMPT-FORMAT ABLATION  (HotpotQA + RAG-v1 format, n={len(dataset)}, skipped={skipped})")
    print("=" * 70)
    print(f"  {'Condition':<22s} {'CE':>7s} {'PPL':>8s}")
    print("-" * 42)

    for cond in conditions:
        ce_sum, n_tok = accum[cond]
        if n_tok == 0:
            print(f"  {cond:<22s} {'--':>7s} {'--':>8s}")
            continue
        ce = ce_sum / n_tok
        ppl = math.exp(min(ce, 20))
        print(f"  {cond:<22s} {ce:>7.4f} {ppl:>8.2f}")

    print("=" * 70)
    print()
    print("Compare against eval.py --dataset hotpotqa (original prompt format).")
    print("If CE is similar → domain shift. If CE is much better → prompt mismatch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt-format ablation for HotpotQA")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--compression_ratios", type=int, nargs="+", default=[32])
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        checkpoint_path=args.checkpoint,
        compression_ratios=args.compression_ratios,
        max_samples=args.max_samples,
        seed=args.seed,
    )
