import torch
from transformers import PreTrainedTokenizer


class NPTCollator:
    """Collate RAGDataset items for next-token prediction pretraining.

    Documents are compressed into a KV prefix by the Q-Former. The full
    Q+A sequence serves as the continuation to predict — ALL continuation
    tokens are supervised, unlike RAGCollator which only labels answer tokens.

    This trains the Q-Former to produce compressed representations that
    preserve enough information for general language modeling, which is
    the most downstream-aligned pretraining objective for RAG.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        # batch_size=1
        item = batch[0]

        doc_token_ids = item["doc_token_ids"]  # list of tensors
        doc_lengths = [t.shape[0] for t in doc_token_ids]
        preamble_ids = item["preamble_ids"]  # tensor

        # Build continuation: qa_suffix + answer + EOS
        stage_b_ids = self._build_continuation(item)

        # Labels: ALL continuation tokens are prediction targets
        labels = stage_b_ids.clone()

        return {
            "doc_token_ids": doc_token_ids,
            "doc_lengths": doc_lengths,
            "preamble_ids": preamble_ids,
            "stage_b_input_ids": stage_b_ids.unsqueeze(0),  # [1, seq_len]
            "stage_b_labels": labels.unsqueeze(0),           # [1, seq_len]
        }

    def _build_continuation(self, item: dict) -> torch.Tensor:
        """Build the continuation sequence from Q+A tokens.

        The full sequence: qa_suffix + answer + EOS
        All tokens are prediction targets for the NTP objective.
        """
        qa_suffix_ids = item["qa_suffix_ids"].tolist()
        answer_ids = item["answer_ids"].tolist()
        eos_id = self.tokenizer.eos_token_id

        full_ids = qa_suffix_ids + answer_ids + [eos_id]
        return torch.tensor(full_ids, dtype=torch.long)
