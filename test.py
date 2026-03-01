"""Baseline evaluation: run frozen LLM with full documents (no Q-Former compression).

Computes average cross-entropy loss and perplexity on the eval split.
This establishes the ceiling that Q-Former compression must approach.
"""

import logging
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    # Reconstruct full user message with documents
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
        # Truncate from the prompt side (keep answer intact)
        excess = len(full_ids) - max_total_tokens
        prompt_ids = prompt_ids[excess:]
        full_ids = prompt_ids + answer_ids + [end_token]

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    answer_start = len(prompt_ids)
    labels[answer_start:] = input_ids[answer_start:]

    return input_ids, labels


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

    total_loss = 0.0
    total_tokens = 0
    num_samples = 0
    device = next(model.parameters()).device

    logger.info(f"Evaluating {len(dataset)} samples...")
    for idx in range(len(dataset)):
        item = dataset[idx]
        result = build_baseline_input(item, tokenizer)
        if result is None:
            continue

        input_ids, labels = result
        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, use_cache=False)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Per-token loss (reduction=none)
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )

        # Count non-ignored tokens
        valid_mask = shift_labels.view(-1) != -100
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            continue

        sample_loss = loss_per_token[valid_mask].sum().item()
        total_loss += sample_loss
        total_tokens += n_valid
        num_samples += 1

        if (idx + 1) % 100 == 0:
            avg = total_loss / total_tokens
            logger.info(
                f"  [{idx+1}/{len(dataset)}] running avg loss: {avg:.4f}, ppl: {math.exp(avg):.2f}"
            )

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)

    logger.info("=" * 60)
    logger.info(f"Baseline Results ({num_samples} samples, {total_tokens} tokens)")
    logger.info(f"  Average CE Loss: {avg_loss:.4f}")
    logger.info(f"  Perplexity:      {ppl:.2f}")
    logger.info("=" * 60)

    return avg_loss, ppl


if __name__ == "__main__":
    evaluate_baseline()
