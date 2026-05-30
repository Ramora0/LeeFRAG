"""GPU test: FlexAttention block mask == trusted dense block-causal-isolated mask.

Skips on machines without CUDA (FlexAttention needs a CUDA build + torch.compile).
"""

import torch

from leefrag_v2.model.flex_masks import (
    build_block_causal_isolated_dense,
    build_chunk_block_mask,
    build_doc_id_per_pos,
    make_block_causal_isolated_mask_mod,
)
from leefrag_v2.model.gated_attention import _dense_attention


def test_flex_mask_matches_dense():
    if not torch.cuda.is_available():
        print("SKIP test_flex_mask_matches_dense (no CUDA)")
        return
    from torch.nn.attention.flex_attention import flex_attention

    device = "cuda"
    doc_lengths = [5, 3, 4]
    D = sum(doc_lengths)

    # (1) mask_mod logic == dense pattern (vectorized)
    doc_id = build_doc_id_per_pos(doc_lengths, device)
    mm = make_block_causal_isolated_mask_mod(doc_id)
    qi = torch.arange(D, device=device).view(D, 1).expand(D, D)
    kj = torch.arange(D, device=device).view(1, D).expand(D, D)
    mod_bool = mm(0, 0, qi, kj)
    ref = build_block_causal_isolated_dense(doc_lengths, torch.float32, device)[0, 0]
    assert torch.equal(mod_bool, ref == 0.0)

    # (2) flex output == dense attention output (same q/k/v)
    Hq, Hkv, hd = 8, 2, 16
    q = torch.randn(1, Hq, D, hd, device=device)
    k = torch.randn(1, Hkv, D, hd, device=device)
    v = torch.randn(1, Hkv, D, hd, device=device)
    bm = build_chunk_block_mask(doc_lengths, device)
    out_flex = flex_attention(q, k, v, block_mask=bm, scale=hd**-0.5, enable_gqa=True)
    out_dense = _dense_attention(q, k, v, ref.unsqueeze(0).unsqueeze(0), hd**-0.5, Hq // Hkv)
    diff = (out_flex - out_dense).abs().max().item()
    assert torch.allclose(out_flex, out_dense, atol=1e-2, rtol=1e-2), diff
    print(f"PASS test_flex_mask_matches_dense (max diff {diff:.2e})")


if __name__ == "__main__":
    test_flex_mask_matches_dense()
