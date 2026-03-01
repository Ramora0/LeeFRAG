"""Baseline evaluation: run frozen LLM with full documents (no Q-Former compression).

Two baselines:
1. Full causal attention (standard RAG) — absolute ceiling.
2. Block attention for docs, full attention for Q+A — ceiling given our attention pattern.
"""

import logging
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from block_attention import build_block_causal_mask_with_qa
from collator import RAGCollator
from config import ModelConfig, TrainingConfig
from dataset import RAGDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_baseline_input(
    item: dict, tokenizer, max_total_tokens: int = 4096
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build full input with documents inlined in the user message.

    This is the standard RAG inference format:
    system + user(docs + answer_mode + question) + assistant(answer)
    """
    doc_texts = item["doc_texts"]
    full_docs = "\n\n".join(doc_texts)
    user_content = full_docs + item["question_suffix"]

    messages = [
        {"role": "system", "content": item["system_prompt"]},
        {"role": "user", "content": user_content},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    answer_ids = item["answer_ids"].tolist()

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    end_token = eot_id if eot_id is not None and eot_id != tokenizer.unk_token_id else tokenizer.eos_token_id
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


MAX_SAMPLES = 1000


def eval_loop(name, dataset, model, tokenizer, device, forward_fn):
    """Shared eval loop. forward_fn(model, item) -> (loss_sum, n_valid_tokens)."""
    total_loss = 0.0
    total_tokens = 0
    num_samples = 0

    n = min(len(dataset), MAX_SAMPLES)
    pbar = tqdm(range(n), desc=name)
    for idx in pbar:
        item = dataset[idx]
        result = forward_fn(model, item, tokenizer, device)
        if result is None:
            continue

        loss_sum, n_valid = result
        total_loss += loss_sum
        total_tokens += n_valid
        num_samples += 1

        if total_tokens > 0:
            avg = total_loss / total_tokens
            pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{math.exp(avg):.2f}")

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, num_samples, total_tokens


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
    """Block attention for docs, Q+A attends to all docs + causal within itself."""
    doc_token_ids = item["doc_token_ids"]
    doc_lengths = [t.shape[0] for t in doc_token_ids]

    if not doc_token_ids or sum(doc_lengths) == 0:
        return None

    # Build Q+A tokens using the collator logic
    collator = _get_collator(tokenizer)
    stage_b_ids, answer_start, answer_end = collator._build_stage_b_tokens(item)

    # Concatenate: [doc0 | doc1 | ... | Q+A]
    doc_concat = torch.cat(doc_token_ids, dim=0)
    full_input = torch.cat([doc_concat, stage_b_ids], dim=0).unsqueeze(0).to(device)

    qa_length = stage_b_ids.shape[0]

    # Block-diagonal for docs, Q+A attends to all docs + causal
    attn_mask = build_block_causal_mask_with_qa(
        doc_lengths, qa_length, dtype=torch.float16, device=device
    )

    # Labels: -100 for everything except answer tokens (offset by doc length)
    doc_total = sum(doc_lengths)
    full_labels = torch.full((full_input.shape[1],), -100, dtype=torch.long, device=device)
    full_labels[doc_total + answer_start : doc_total + answer_end] = (
        full_input[0, doc_total + answer_start : doc_total + answer_end]
    )
    full_labels = full_labels.unsqueeze(0)

    with torch.amp.autocast("cuda"):
        outputs = model(
            input_ids=full_input,
            attention_mask=attn_mask,
            use_cache=False,
        )

    return _compute_token_loss(outputs.logits, full_labels)


# Cache the collator instance
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


@torch.no_grad()
def evaluate_baseline():
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

    logger.info("Loading eval dataset")
    dataset = RAGDataset(
        tokenizer=tokenizer,
        model_config=model_config,
        split="eval",
        eval_split_ratio=training_config.eval_split_ratio,
        seed=training_config.seed,
    )

    device = next(model.parameters()).device

    # === Baseline 1: Full causal attention ===
    causal_loss, causal_ppl, causal_n, causal_tok = eval_loop(
        "Full Causal", dataset, model, tokenizer, device, forward_full_causal
    )

    # === Baseline 2: Block attention ===
    block_loss, block_ppl, block_n, block_tok = eval_loop(
        "Block Attention", dataset, model, tokenizer, device, forward_block_attention
    )

    # === Results ===
    logger.info("=" * 70)
    logger.info("BASELINE RESULTS")
    logger.info("=" * 70)
    logger.info(
        f"  Full Causal:    CE={causal_loss:.4f}  PPL={causal_ppl:.2f}  "
        f"({causal_n} samples, {causal_tok} tokens)"
    )
    logger.info(
        f"  Block Attention: CE={block_loss:.4f}  PPL={block_ppl:.2f}  "
        f"({block_n} samples, {block_tok} tokens)"
    )
    delta = block_loss - causal_loss
    logger.info(f"  Block attn overhead: +{delta:.4f} CE ({delta/causal_loss*100:.1f}%)")
    logger.info("=" * 70)

    return {
        "full_causal": {"loss": causal_loss, "ppl": causal_ppl},
        "block_attention": {"loss": block_loss, "ppl": block_ppl},
    }


if __name__ == "__main__":
    evaluate_baseline()
