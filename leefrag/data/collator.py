import torch
from transformers import PreTrainedTokenizer


class RAGCollator:
    """Collate RAGDataset items for the two-stage training pipeline.

    Since batch_size=1, this mostly restructures a single sample.
    Builds Stage B input from qa_suffix_ids + answer + EOS.
    Documents are encoded via the preamble + block attention in Stage A.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        # batch_size=1
        item = batch[0]

        doc_token_ids = item["doc_token_ids"]  # list of tensors
        doc_lengths = [t.shape[0] for t in doc_token_ids]
        preamble_ids = item["preamble_ids"]  # tensor

        # Build Stage B input: qa_suffix + answer + EOS
        stage_b_ids, answer_start, answer_end = self._build_stage_b_tokens(item)

        # Labels: -100 everywhere except answer tokens
        labels = torch.full_like(stage_b_ids, -100)
        labels[answer_start:answer_end] = stage_b_ids[answer_start:answer_end]

        return {
            "doc_token_ids": doc_token_ids,
            "doc_lengths": doc_lengths,
            "preamble_ids": preamble_ids,
            "stage_b_input_ids": stage_b_ids.unsqueeze(0),  # [1, seq_len]
            "stage_b_labels": labels.unsqueeze(0),  # [1, seq_len]
            "answer_start": answer_start,
            "answer_end": answer_end,
        }

    def _build_stage_b_tokens(self, item: dict) -> tuple[torch.Tensor, int, int]:
        """Build tokenized Stage B input from QA suffix + answer.

        Stage B input is just the QA block that follows the compressed KV prefix:
        <|user|>\n{question}\n<|assistant|>\n{answer}<|end_of_text|>

        The preamble and documents are already encoded in the KV prefix
        (from Stage A / Q-Former compression).

        Returns:
            input_ids: Full tokenized sequence.
            answer_start: Index where answer tokens begin (for labels).
            answer_end: Index where answer tokens end.
        """
        qa_suffix_ids = item["qa_suffix_ids"].tolist()
        answer_ids = item["answer_ids"].tolist()
        end_token = self.tokenizer.eos_token_id

        full_ids = qa_suffix_ids + answer_ids + [end_token]
        answer_start = len(qa_suffix_ids)
        answer_end = len(qa_suffix_ids) + len(answer_ids) + 1  # include end token in loss

        return torch.tensor(full_ids, dtype=torch.long), answer_start, answer_end
