import re

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from config import ModelConfig

HOTPOTQA_SYSTEM_PROMPT = "Answer the question based on the provided documents. Give a short, direct answer."


def parse_documents(documents_text: str) -> list[str]:
    """Parse Document:N formatted text into individual document strings."""
    parts = re.split(r"(?=Document:\d+)", documents_text.strip())
    docs = [p.strip() for p in parts if p.strip()]
    return docs


class RAGDataset(Dataset):
    """Dataset for glaiveai/RAG-v1.

    Columns: split, question, documents, system_prompt, answer, answer_mode.
    User message format: documents + "\\n\\nAnswer Mode: " + answer_mode + "\\n\\nQuestion: " + question
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        split: str = "train",
        eval_split_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.config = model_config

        raw = load_dataset("glaiveai/RAG-v1", split="train")
        raw = raw.shuffle(seed=seed)
        split_idx = int(len(raw) * (1 - eval_split_ratio))
        if split == "train":
            self.data = raw.select(range(split_idx))
        else:
            self.data = raw.select(range(split_idx, len(raw)))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]

        system_prompt = row["system_prompt"]
        question = row["question"]
        answer = row["answer"]
        answer_mode = row["answer_mode"]
        documents_text = row["documents"]

        # Parse and tokenize documents individually
        doc_texts = parse_documents(documents_text)
        doc_token_ids = self._tokenize_documents(doc_texts)

        # Build the question suffix (answer_mode + question, no docs)
        question_suffix = f"\n\nAnswer Mode: {answer_mode}\n\nQuestion: {question}"

        # Tokenize question suffix and answer (no special tokens - collator handles template)
        question_ids = self.tokenizer.encode(
            question_suffix,
            add_special_tokens=False,
            max_length=self.config.max_question_tokens,
            truncation=True,
        )
        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
            max_length=self.config.max_answer_tokens,
            truncation=True,
        )

        return {
            "doc_texts": doc_texts,
            "doc_token_ids": doc_token_ids,
            "question_suffix": question_suffix,
            "answer": answer,
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "system_prompt": system_prompt,
        }

    def _tokenize_documents(self, doc_texts: list[str]) -> list[torch.Tensor]:
        """Tokenize each document independently, respecting per-doc and total limits."""
        doc_token_ids = []
        total_tokens = 0

        for doc in doc_texts:
            ids = self.tokenizer.encode(
                doc,
                add_special_tokens=False,
                max_length=self.config.max_doc_tokens,
                truncation=True,
            )
            if total_tokens + len(ids) > self.config.max_total_doc_tokens:
                remaining = self.config.max_total_doc_tokens - total_tokens
                if remaining > 0:
                    ids = ids[:remaining]
                else:
                    break
            doc_token_ids.append(torch.tensor(ids, dtype=torch.long))
            total_tokens += len(ids)

        return doc_token_ids


class HotPotQADataset(Dataset):
    """Dataset for HotpotQA distractor split.

    Produces the same dict interface as RAGDataset so the collator and trainer
    work without changes.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        split: str = "train",
        eval_split_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.config = model_config

        # HotpotQA has train/validation splits; map our eval to validation
        if split == "train":
            self.data = load_dataset(
                "hotpot_qa", "distractor", split="train", trust_remote_code=True,
            )
        else:
            self.data = load_dataset(
                "hotpot_qa", "distractor", split="validation", trust_remote_code=True,
            )
        self.data = self.data.shuffle(seed=seed)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]

        # Extract documents from structured context
        titles = row["context"]["title"]
        sentences_list = row["context"]["sentences"]
        doc_texts = []
        for title, sentences in zip(titles, sentences_list):
            text = f"{title}: {''.join(sentences)}"
            doc_texts.append(text)

        question = row["question"]
        answer = row["answer"]

        # Tokenize documents
        doc_token_ids = self._tokenize_documents(doc_texts)

        # Build question suffix (matches eval.py pattern)
        question_suffix = f"Question: {question}"

        # Tokenize question suffix and answer
        question_ids = self.tokenizer.encode(
            question_suffix,
            add_special_tokens=False,
            max_length=self.config.max_question_tokens,
            truncation=True,
        )
        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
            max_length=self.config.max_answer_tokens,
            truncation=True,
        )

        return {
            "doc_texts": doc_texts,
            "doc_token_ids": doc_token_ids,
            "question_suffix": question_suffix,
            "answer": answer,
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "system_prompt": HOTPOTQA_SYSTEM_PROMPT,
        }

    def _tokenize_documents(self, doc_texts: list[str]) -> list[torch.Tensor]:
        """Tokenize each document independently, respecting per-doc and total limits."""
        doc_token_ids = []
        total_tokens = 0

        for doc in doc_texts:
            ids = self.tokenizer.encode(
                doc,
                add_special_tokens=False,
                max_length=self.config.max_doc_tokens,
                truncation=True,
            )
            if total_tokens + len(ids) > self.config.max_total_doc_tokens:
                remaining = self.config.max_total_doc_tokens - total_tokens
                if remaining > 0:
                    ids = ids[:remaining]
                else:
                    break
            doc_token_ids.append(torch.tensor(ids, dtype=torch.long))
            total_tokens += len(ids)

        return doc_token_ids


def create_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    model_config: ModelConfig,
    split: str = "train",
    eval_split_ratio: float = 0.1,
    seed: int = 42,
) -> Dataset:
    """Factory function to create the appropriate dataset."""
    if dataset_name == "rag_v1":
        return RAGDataset(
            tokenizer=tokenizer,
            model_config=model_config,
            split=split,
            eval_split_ratio=eval_split_ratio,
            seed=seed,
        )
    elif dataset_name == "hotpotqa":
        return HotPotQADataset(
            tokenizer=tokenizer,
            model_config=model_config,
            split=split,
            eval_split_ratio=eval_split_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Choose from: rag_v1, hotpotqa")
