import json
import logging
import re

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from leefrag.config import ModelConfig

logger = logging.getLogger(__name__)

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

        # Build preamble: <|system|>\n{system_prompt}\n\n
        preamble_text = f"<|system|>\n{system_prompt}\n\n"
        preamble_ids = self.tokenizer.encode(
            preamble_text, add_special_tokens=False,
        )

        # Build QA suffix: <|user|>\n{question_suffix}\n<|assistant|>\n
        question_suffix = f"\n\nAnswer Mode: {answer_mode}\n\nQuestion: {question}"
        qa_suffix_text = f"<|user|>\n{question_suffix}\n<|assistant|>\n"
        qa_suffix_ids = self.tokenizer.encode(
            qa_suffix_text, add_special_tokens=False,
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
            "preamble_ids": torch.tensor(preamble_ids, dtype=torch.long),
            "qa_suffix_ids": torch.tensor(qa_suffix_ids, dtype=torch.long),
            "question_suffix": question_suffix,
            "answer": answer,
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

        # Build preamble: <|system|>\n{system_prompt}\n\n
        preamble_text = f"<|system|>\n{HOTPOTQA_SYSTEM_PROMPT}\n\n"
        preamble_ids = self.tokenizer.encode(
            preamble_text, add_special_tokens=False,
        )

        # Build QA suffix: <|user|>\n{question_suffix}\n<|assistant|>\n
        question_suffix = f"Question: {question}"
        qa_suffix_text = f"<|user|>\n{question_suffix}\n<|assistant|>\n"
        qa_suffix_ids = self.tokenizer.encode(
            qa_suffix_text, add_special_tokens=False,
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
            "preamble_ids": torch.tensor(preamble_ids, dtype=torch.long),
            "qa_suffix_ids": torch.tensor(qa_suffix_ids, dtype=torch.long),
            "question_suffix": question_suffix,
            "answer": answer,
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


class GeneralTextDataset(Dataset):
    """General text dataset for NPT pretraining (SlimPajama, following ReFRAG).

    Loads text from SlimPajama-6B, filters by source domain (Book + ArXiv by
    default). Long documents are chunked with a stride into multiple examples,
    each split into context (compressed by Q-Former) and continuation (target).

    No block attention chunking — context is treated as one contiguous sequence.
    """

    SOURCES = {
        "book": "RedPajamaBook",
        "arxiv": "RedPajamaArXiv",
        "wikipedia": "RedPajamaWikipedia",
        "c4": "RedPajamaC4",
        "commoncrawl": "RedPajamaCommonCrawl",
        "stackexchange": "RedPajamaStackExchange",
        "github": "RedPajamaGithub",
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        split: str = "train",
        eval_split_ratio: float = 0.1,
        seed: int = 42,
        sources: tuple[str, ...] = ("book", "arxiv"),
        max_documents: int = 0,
        max_continuation_tokens: int = 1024,
        min_chars: int = 4000,
    ):
        self.tokenizer = tokenizer
        self.config = model_config
        self.max_continuation_tokens = max_continuation_tokens

        source_names = {self.SOURCES[s] for s in sources}

        logger.info(f"Loading SlimPajama-6B (sources={list(sources)})...")
        ds = load_dataset("DKYoon/SlimPajama-6B", split="train")

        def _source_filter(example):
            meta = example["meta"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            return (
                meta.get("redpajama_set_name") in source_names
                and len(example["text"]) >= min_chars
            )

        ds = ds.filter(_source_filter, num_proc=4, desc="Filtering by source")
        ds = ds.shuffle(seed=seed)

        if max_documents > 0:
            ds = ds.select(range(min(max_documents, len(ds))))

        split_idx = int(len(ds) * (1 - eval_split_ratio))
        if split == "train":
            raw_data = ds.select(range(split_idx))
        else:
            raw_data = ds.select(range(split_idx, len(ds)))

        # Tokenize docs and chunk into fixed-size windows
        window = model_config.max_total_doc_tokens + max_continuation_tokens
        self.chunks = []
        for i in range(len(raw_data)):
            token_ids = tokenizer.encode(raw_data[i]["text"], add_special_tokens=False)
            if len(token_ids) < 256:
                continue
            for start in range(0, len(token_ids) - 256, window):
                self.chunks.append(token_ids[start : start + window])

        logger.info(
            f"GeneralTextDataset ({split}): {len(self.chunks)} chunks"
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        token_ids = self.chunks[idx]

        max_context = self.config.max_total_doc_tokens
        max_cont = self.max_continuation_tokens

        # Split: context gets up to max_context, rest is continuation
        context_len = min(len(token_ids) - 128, max_context)
        context_len = max(context_len, 1)

        context_ids = token_ids[:context_len]
        continuation_ids = token_ids[context_len : context_len + max_cont]

        # Single document — no block attention chunking
        doc_token_ids = [torch.tensor(context_ids, dtype=torch.long)]

        # Minimal preamble: BOS token
        bos_id = self.tokenizer.bos_token_id
        if bos_id is not None:
            preamble_ids = torch.tensor([bos_id], dtype=torch.long)
        else:
            preamble_ids = torch.tensor([], dtype=torch.long)

        return {
            "doc_texts": [],
            "doc_token_ids": doc_token_ids,
            "preamble_ids": preamble_ids,
            "qa_suffix_ids": torch.tensor([], dtype=torch.long),
            "question_suffix": "",
            "answer": "",
            "answer_ids": torch.tensor(continuation_ids, dtype=torch.long),
            "system_prompt": "",
        }


def create_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    model_config: ModelConfig,
    split: str = "train",
    eval_split_ratio: float = 0.1,
    seed: int = 42,
    **kwargs,
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
    elif dataset_name == "slimpajama":
        return GeneralTextDataset(
            tokenizer=tokenizer,
            model_config=model_config,
            split=split,
            eval_split_ratio=eval_split_ratio,
            seed=seed,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            "Choose from: rag_v1, hotpotqa, slimpajama"
        )
