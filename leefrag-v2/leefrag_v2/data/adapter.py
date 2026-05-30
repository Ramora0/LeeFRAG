"""Reuse the v1 data pipeline and assemble the single [preamble|docs|Q+A] sequence.

We reuse leefrag.data.dataset (RAGDataset/HotPotQADataset) and the RAGCollator
unchanged, wrapping the dataset to expose a stable example index so the offline
teacher logits can be keyed per example.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader

from leefrag.config import ModelConfig
from leefrag.data.collator import RAGCollator

from leefrag_v2.config import V2ModelConfig, V2TrainingConfig


class IndexedDataset(Dataset):
    """Wrap a base dataset so each item carries its stable index."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        item["example_idx"] = idx
        return item


class V2Collator(RAGCollator):
    """RAGCollator + pass-through of the example index."""

    def __call__(self, batch: list[dict]) -> dict:
        out = super().__call__(batch)
        out["example_idx"] = int(batch[0].get("example_idx", -1))
        return out


def _to_model_config(mc: V2ModelConfig) -> ModelConfig:
    """The reused dataset expects a v1 ModelConfig (tokenization limits only)."""
    return ModelConfig(
        model_name=mc.model_name,
        max_doc_tokens=mc.max_doc_tokens,
        max_total_doc_tokens=mc.max_total_doc_tokens,
        max_question_tokens=mc.max_question_tokens,
        max_answer_tokens=mc.max_answer_tokens,
    )


def make_dataloaders(tokenizer, model_config: V2ModelConfig, cfg: V2TrainingConfig):
    from leefrag.data.dataset import create_dataset  # lazy: pulls in `datasets`

    mc = _to_model_config(model_config)
    train = IndexedDataset(
        create_dataset(cfg.dataset_name, tokenizer, mc, split="train",
                       eval_split_ratio=cfg.eval_split_ratio, seed=cfg.seed)
    )
    eval_ = IndexedDataset(
        create_dataset(cfg.dataset_name, tokenizer, mc, split="eval",
                       eval_split_ratio=cfg.eval_split_ratio, seed=cfg.seed)
    )
    collate = V2Collator(tokenizer)
    train_loader = DataLoader(
        train, batch_size=1, shuffle=True, collate_fn=collate,
        num_workers=cfg.dataloader_num_workers,
    )
    eval_loader = DataLoader(
        eval_, batch_size=1, shuffle=False, collate_fn=collate,
        num_workers=cfg.dataloader_num_workers,
    )
    return train_loader, eval_loader


def build_blocks(collated: dict, device) -> dict | None:
    """Assemble [preamble | doc0 | ... | docK | Q+A] + full-sequence labels.

    Block 0 merges preamble + doc0 (matches the v1 trainer convention). Returns
    None for empty examples.
    """
    doc_token_ids = collated["doc_token_ids"]
    doc_lengths = collated["doc_lengths"]
    if not doc_token_ids or sum(doc_lengths) == 0:
        return None

    preamble_ids = collated["preamble_ids"]
    qa_ids = collated["stage_b_input_ids"]      # [1, qa_len]
    qa_labels = collated["stage_b_labels"]      # [1, qa_len]
    preamble_len = preamble_ids.shape[0]

    block_lengths = [preamble_len + doc_lengths[0]] + list(doc_lengths[1:])
    doc_concat = torch.cat(doc_token_ids, dim=0)
    full = torch.cat([preamble_ids, doc_concat, qa_ids.squeeze(0)], dim=0).unsqueeze(0)
    full = full.to(device)

    D = preamble_len + sum(doc_lengths)
    qa_len = qa_ids.shape[1]

    labels = torch.full((1, full.shape[1]), -100, dtype=torch.long, device=device)
    labels[:, D:] = qa_labels.to(device)

    return {
        "input_ids": full,
        "labels": labels,
        "block_lengths": block_lengths,
        "doc_total": D,
        "qa_len": qa_len,
        "example_idx": int(collated.get("example_idx", -1)),
    }
