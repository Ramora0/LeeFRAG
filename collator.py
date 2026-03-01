import torch
from transformers import PreTrainedTokenizer


class RAGCollator:
    """Collate RAGDataset items for the two-stage training pipeline.

    Since batch_size=1, this mostly restructures a single sample.
    Builds the chat template tokens for Stage B (question + answer).
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
        """Build tokenized Stage B input using the chat template.

        Format (Llama 3 chat template):
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {question_suffix}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {answer}<|eot_id|>

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

        # Add EOS after answer
        eos_id = self.tokenizer.eos_token_id
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        end_token = eot_id if eot_id is not None and eot_id != self.tokenizer.unk_token_id else eos_id

        full_ids = prompt_ids + answer_ids + [end_token]
        answer_start = len(prompt_ids)
        answer_end = len(prompt_ids) + len(answer_ids) + 1  # include end token in loss

        return torch.tensor(full_ids, dtype=torch.long), answer_start, answer_end
