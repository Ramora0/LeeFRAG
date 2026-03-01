import torch


def build_block_causal_mask(
    doc_lengths: list[int],
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a block-diagonal causal attention mask for concatenated documents.

    Each document can attend causally within itself but not across documents.

    Args:
        doc_lengths: Length (in tokens) of each document.
        dtype: Mask dtype (use float16/bfloat16, masked positions = -inf).
        device: Target device.

    Returns:
        mask: [1, 1, total_len, total_len] attention mask.
              0.0 for allowed positions, -inf for masked positions.
    """
    total_len = sum(doc_lengths)
    # Start with all masked
    mask = torch.full(
        (total_len, total_len),
        float("-inf"),
        dtype=dtype,
        device=device,
    )

    offset = 0
    for length in doc_lengths:
        # Causal mask within this block: lower-triangular
        block_mask = torch.triu(
            torch.full((length, length), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
        mask[offset : offset + length, offset : offset + length] = block_mask
        offset += length

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total_len, total_len]


def build_block_causal_mask_with_qa(
    doc_lengths: list[int],
    qa_length: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build attention mask for [docs | Q+A] in a single forward pass.

    Documents: block-diagonal causal (same as build_block_causal_mask).
    Q+A tokens: causal within themselves, full attention to ALL document tokens.

         doc0  doc1  doc2  Q+A
    doc0 [caus   -     -     -  ]
    doc1 [  -  caus    -     -  ]
    doc2 [  -    -   caus    -  ]
    Q+A  [full full  full  caus ]

    Args:
        doc_lengths: Length (in tokens) of each document.
        qa_length: Length of question+answer tokens.
        dtype: Mask dtype.
        device: Target device.

    Returns:
        mask: [1, 1, total_len, total_len] where total_len = sum(doc_lengths) + qa_length.
    """
    doc_total = sum(doc_lengths)
    total_len = doc_total + qa_length

    mask = torch.full(
        (total_len, total_len),
        float("-inf"),
        dtype=dtype,
        device=device,
    )

    # Document blocks: block-diagonal causal
    offset = 0
    for length in doc_lengths:
        block_mask = torch.triu(
            torch.full((length, length), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
        mask[offset : offset + length, offset : offset + length] = block_mask
        offset += length

    # Q+A rows: attend to all doc tokens
    mask[doc_total:, :doc_total] = 0.0

    # Q+A rows: causal within Q+A
    qa_causal = torch.tril(
        torch.zeros(qa_length, qa_length, dtype=dtype, device=device)
    )
    mask[doc_total:, doc_total:] = qa_causal

    return mask.unsqueeze(0).unsqueeze(0)


def build_prefix_causal_mask(
    prefix_length: int,
    seq_length: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build causal mask for sequence attending to a KV cache prefix.

    The sequence tokens attend causally to themselves and fully to the prefix.

    Args:
        prefix_length: Number of prefix KV cache tokens.
        seq_length: Number of new sequence tokens.
        dtype: Mask dtype.
        device: Target device.

    Returns:
        mask: [1, 1, seq_length, prefix_length + seq_length]
    """
    total_len = prefix_length + seq_length
    mask = torch.full(
        (seq_length, total_len),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    # Attend to all prefix tokens
    mask[:, :prefix_length] = 0.0
    # Causal attention among sequence tokens
    seq_mask = torch.tril(
        torch.zeros(seq_length, seq_length, dtype=dtype, device=device)
    )
    mask[:, prefix_length:] = seq_mask

    return mask.unsqueeze(0).unsqueeze(0)
