"""GPU parity: factorized gated attention (FlexAttention path) == dense reference.

With gate=ones, the two-path attention must equal a single dense block-causal +
Q+A attention. Validates the FlexAttention chunk path + the manual Q+A path
together. Skips without CUDA.
"""

import torch

from leefrag_v2.model.gated_attention import (
    patched_llama_attention_forward,
    reference_dense_attention,
)
from leefrag_v2.model.patch import new_context


def _tiny_attention(device, dtype):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention

    cfg = LlamaConfig(
        hidden_size=64, num_attention_heads=8, num_key_value_heads=2,
        head_dim=8, num_hidden_layers=2, intermediate_size=128, vocab_size=128,
    )
    cfg._attn_implementation = "eager"
    return LlamaAttention(cfg, layer_idx=0).to(device=device, dtype=dtype).eval(), cfg


def test_factorized_flex_equals_reference():
    if not torch.cuda.is_available():
        print("SKIP test_factorized_flex_equals_reference (no CUDA)")
        return
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    device = "cuda"
    dtype = torch.float32
    attn, cfg = _tiny_attention(device, dtype)
    Hq, Hkv, hd, H = (
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    )

    for doc_lengths in ([3, 2], [7], [1, 1, 1, 1], [5, 4, 3]):
        qa_len = 4
        D = sum(doc_lengths)
        S = D + qa_len
        hidden = torch.randn(1, S, H, device=device, dtype=dtype)
        cos = torch.ones(1, S, hd, device=device, dtype=dtype)
        sin = torch.zeros(1, S, hd, device=device, dtype=dtype)

        ctx = new_context(
            doc_lengths, qa_len, device, cfg.num_hidden_layers,
            use_flex=True, selector=None,
        )
        attn._v2ctx = ctx
        with torch.no_grad():
            patched_out, _ = patched_llama_attention_forward(attn, hidden, (cos, sin))
            q = attn.q_proj(hidden).view(1, S, Hq, hd).transpose(1, 2)
            k = attn.k_proj(hidden).view(1, S, Hkv, hd).transpose(1, 2)
            v = attn.v_proj(hidden).view(1, S, Hkv, hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            ref = reference_dense_attention(q, k, v, doc_lengths, qa_len, attn.scaling, Hq, Hkv)
            ref_out = attn.o_proj(ref.transpose(1, 2).reshape(1, S, Hq * hd))

        diff = (patched_out - ref_out).abs().max().item()
        assert torch.allclose(patched_out, ref_out, atol=2e-2, rtol=2e-2), (doc_lengths, diff)
        print(f"PASS doc_lengths={doc_lengths} (max diff {diff:.2e})")


if __name__ == "__main__":
    test_factorized_flex_equals_reference()
