import torch
from transformers import PreTrainedTokenizer


class RAGCollator:
    """Collate RAGDataset items for the two-stage training pipeline.

    Since batch_size=1, this mostly restructures a single sample.
    Builds the chat template tokens for Stage B (question + answer).
    Uses the Tulu 3 chat template format.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        # batch_size=1
        item = batch[0]

        doc_token_ids = item["doc_token_ids"]  # list of tensors
        doc_lengths = [t.shape[0] for t in doc_token_ids]

        # Build Stage B input: chat template wrapping question + answer
        stage_b_ids, answer_start, answer_end = self._build_stage_b_tokens(item)

        # Labels: -100 everywhere except answer tokens
        labels = torch.full_like(stage_b_ids, -100)
        labels[answer_start:answer_end] = stage_b_ids[answer_start:answer_end]

        return {
            "doc_token_ids": doc_token_ids,
            "doc_lengths": doc_lengths,
            "stage_b_input_ids": stage_b_ids.unsqueeze(0),  # [1, seq_len]
            "stage_b_labels": labels.unsqueeze(0),  # [1, seq_len]
            "answer_start": answer_start,
            "answer_end": answer_end,
        }

    def _build_stage_b_tokens(self, item: dict) -> tuple[torch.Tensor, int, int]:
        """Build tokenized Stage B input using the Tulu 3 chat template.

        Format (Tulu 3 chat template):
        <|system|>
        {system_prompt}
        <|user|>
        {question_suffix}
        <|assistant|>
        {answer}<|end_of_text|>

        Note: Documents are NOT included here - they come from the KV cache prefix.
        The question_suffix contains "Answer Mode: X\\n\\nQuestion: Y".

        Returns:
            input_ids: Full tokenized sequence.
            answer_start: Index where answer tokens begin (for labels).
            answer_end: Index where answer tokens end.
        """
        # Build chat messages (without documents - those come from KV cache)
        messages = [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["question_suffix"]},
        ]

        # Tokenize prompt (everything before the answer)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        answer_ids = item["answer_ids"].tolist()

        # Add EOS after answer (Tulu 3 uses <|end_of_text|> as the end-of-turn token)
        end_token = self.tokenizer.eos_token_id

        full_ids = prompt_ids + answer_ids + [end_token]
        answer_start = len(prompt_ids)
        answer_end = len(prompt_ids) + len(answer_ids) + 1  # include end token in loss

        return torch.tensor(full_ids, dtype=torch.long), answer_start, answer_end
