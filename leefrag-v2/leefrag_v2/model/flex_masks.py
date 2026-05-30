"""FlexAttention mask for the chunk self-attention path.

Chunk tokens are block-causal-isolated: each document attends causally within
itself and not across documents. The same BlockMask is reused across all 32
layers (it depends only on the doc layout), so we cache it per doc-length tuple.

Also provides a dense additive mask builder for the CPU/reference fallback path
(no CUDA / no torch.compile needed) and for parity checking.
"""

from __future__ import annotations

import torch

_BLOCK_MASK_CACHE: dict = {}


def build_doc_id_per_pos(doc_lengths: list[int], device) -> torch.Tensor:
    """[D] tensor mapping each chunk position to its document index."""
    ids = torch.cat(
        [
            torch.full((length,), i, dtype=torch.long, device=device)
            for i, length in enumerate(doc_lengths)
        ]
    )
    return ids


def make_block_causal_isolated_mask_mod(doc_id_per_pos: torch.Tensor):
    """mask_mod(b, h, q_idx, kv_idx) -> bool.

    True iff same document AND causal (kv_idx <= q_idx). Captures the doc-id
    tensor by closure (indexed on-device inside the kernel).
    """

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = doc_id_per_pos[q_idx] == doc_id_per_pos[kv_idx]
        causal = kv_idx <= q_idx
        return same_doc & causal

    return mask_mod


def build_chunk_block_mask(doc_lengths: list[int], device):
    """Create (and cache) a FlexAttention BlockMask over the D x D chunk region.

    Imported lazily so the package imports on machines without a CUDA-capable
    FlexAttention build (the dense fallback path does not need this).
    """
    from torch.nn.attention.flex_attention import create_block_mask

    key = (tuple(doc_lengths), str(device))
    if key in _BLOCK_MASK_CACHE:
        return _BLOCK_MASK_CACHE[key]

    D = sum(doc_lengths)
    doc_id = build_doc_id_per_pos(doc_lengths, device)
    mask_mod = make_block_causal_isolated_mask_mod(doc_id)
    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=D, KV_LEN=D, device=device
    )
    _BLOCK_MASK_CACHE[key] = block_mask
    return block_mask


def build_block_causal_isolated_dense(
    doc_lengths: list[int],
    dtype: torch.dtype = torch.float32,
    device="cpu",
) -> torch.Tensor:
    """Dense additive mask [1,1,D,D] for the chunk region (CPU/reference path).

    0.0 where attention is allowed (same doc, causal), -inf otherwise. Matches
    the pattern of leefrag.model.block_attention.build_block_causal_mask.
    """
    D = sum(doc_lengths)
    mask = torch.full((D, D), float("-inf"), dtype=dtype, device=device)
    offset = 0
    for length in doc_lengths:
        block = torch.triu(
            torch.full((length, length), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
        mask[offset : offset + length, offset : offset + length] = block
        offset += length
    return mask.unsqueeze(0).unsqueeze(0)


def build_qa_causal_bias(
    qa_len: int, dtype: torch.dtype = torch.float32, device="cpu"
) -> torch.Tensor:
    """Additive [1,1,Sa,Sa] causal bias for the Q+A self-attention block."""
    bias = torch.triu(
        torch.full((qa_len, qa_len), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )
    return bias.unsqueeze(0).unsqueeze(0)


def clear_cache() -> None:
    _BLOCK_MASK_CACHE.clear()
