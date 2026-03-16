import torch
from transformers import PreTrainedTokenizer


class ReconstructionCollator:
    """Collate RAGDataset items for reconstruction pretraining.

    Each document gets its own block:
        block_i = [recon_prompt_ids | doc_i_token_ids | EOS]
        labels_i = [-100 * prompt_len | doc_i_token_ids | EOS]

    The reconstruction prompt uses the Tulu 3 chat template:
        <|user|>\nRepeat the following document exactly:\n<|assistant|>\n
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

        # Pre-tokenize the reconstruction prompt once
        prompt_text = "<|user|>\nRepeat the following document exactly:\n<|assistant|>\n"
        self.recon_prompt_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False,
        )
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        # batch_size=1
        item = batch[0]

        doc_token_ids = item["doc_token_ids"]  # list of tensors
        doc_lengths = [t.shape[0] for t in doc_token_ids]
        preamble_ids = item["preamble_ids"]  # tensor

        # Build Stage B input: per-doc reconstruction blocks
        all_input_ids = []
        all_labels = []
        input_block_lengths = []
        prompt_len = len(self.recon_prompt_ids)

        for doc_ids in doc_token_ids:
            doc_list = doc_ids.tolist()
            block_ids = self.recon_prompt_ids + doc_list + [self.eos_id]
            block_labels = [-100] * prompt_len + doc_list + [self.eos_id]

            all_input_ids.extend(block_ids)
            all_labels.extend(block_labels)
            input_block_lengths.append(len(block_ids))

        stage_b_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        stage_b_labels = torch.tensor(all_labels, dtype=torch.long)

        return {
            "doc_token_ids": doc_token_ids,
            "doc_lengths": doc_lengths,
            "preamble_ids": preamble_ids,
            "stage_b_input_ids": stage_b_input_ids.unsqueeze(0),  # [1, total_seq]
            "stage_b_labels": stage_b_labels.unsqueeze(0),        # [1, total_seq]
            "input_block_lengths": input_block_lengths,
        }
