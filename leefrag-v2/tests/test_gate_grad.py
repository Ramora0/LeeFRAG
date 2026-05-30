"""CPU tests for the correctness-critical attention math (no CUDA needed).

Covers:
  1. Straight-through Gumbel-sigmoid gate: forward == hard sample, and gradient
     reaches EVERY chunk token (incl. gated-out ones) via the gate path.
  2. Post-softmax multiplicative gate is "zeroed out" (no renormalization).
  3. Factorized two-path attention == single dense reference when gate=ones.

Run: pytest leefrag-v2/tests/test_gate_grad.py -q
"""

import torch

from leefrag_v2.model.gated_attention import (
    patched_llama_attention_forward,
    reference_dense_attention,
)
from leefrag_v2.model.selector import gumbel_sigmoid_ste
from leefrag_v2.model.patch import new_context


def test_ste_forward_is_hard_and_grad_reaches_all_tokens():
    torch.manual_seed(0)
    D = 6
    logits = torch.linspace(-3.0, 3.0, D).view(1, D).clone().requires_grad_(True)
    noise = torch.zeros(1, D)  # deterministic: gate = (logit > 0)

    gate = gumbel_sigmoid_ste(logits, tau=1.0, noise=noise, hard=True, training=True)

    # forward is exactly the hard sample
    expected_hard = (logits.detach() > 0).float()
    assert torch.equal(gate.detach(), expected_hard)
    assert set(gate.detach().unique().tolist()) <= {0.0, 1.0}

    # post-softmax-gated toy attention; loss backward must reach every logit
    Hq, Sa, hd = 2, 3, 4
    probs_c = torch.rand(1, Hq, Sa, D).softmax(-1)
    v_c = torch.randn(1, Hq, D, hd)
    out = torch.matmul(probs_c * gate.view(1, 1, 1, D), v_c)
    out.sum().backward()

    assert logits.grad is not None
    # every token, including the gated-out (logit<0) ones, receives selector grad
    assert (logits.grad.abs() > 0).all(), logits.grad


def test_gate_zeroes_contribution_without_renorm():
    torch.manual_seed(1)
    Hq, Sa, D, hd = 2, 3, 5, 4
    scores = torch.randn(1, Hq, Sa, D)
    probs = scores.softmax(-1)
    v = torch.randn(1, Hq, D, hd)

    drop = 2
    gate = torch.ones(1, D)
    gate[0, drop] = 0.0

    out_gated = torch.matmul(probs * gate.view(1, 1, 1, D), v)
    out_full = torch.matmul(probs, v)
    contribution = probs[..., drop : drop + 1] * v[:, :, drop : drop + 1, :]
    # zeroed-out == full minus that token's contribution (NOT renormalized)
    assert torch.allclose(out_gated, out_full - contribution, atol=1e-6)


def _tiny_attention():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention

    cfg = LlamaConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        intermediate_size=64,
        vocab_size=128,
    )
    cfg._attn_implementation = "eager"
    attn = LlamaAttention(cfg, layer_idx=0).float().eval()
    return attn, cfg


def test_factorized_equals_dense_reference_gate_ones():
    torch.manual_seed(2)
    attn, cfg = _tiny_attention()

    doc_lengths = [3, 2]
    qa_len = 4
    D = sum(doc_lengths)
    S = D + qa_len
    H = cfg.hidden_size
    hd = cfg.head_dim
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads

    hidden = torch.randn(1, S, H)
    cos = torch.ones(1, S, hd)
    sin = torch.zeros(1, S, hd)  # identity RoPE

    ctx = new_context(
        doc_lengths, qa_len, device="cpu", num_layers=cfg.num_hidden_layers,
        use_flex=False, selector=None,
    )
    attn._v2ctx = ctx

    with torch.no_grad():
        patched_out, _ = patched_llama_attention_forward(attn, hidden, (cos, sin))

        # reference: same projections + identity RoPE, single dense attention
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        q = attn.q_proj(hidden).view(1, S, Hq, hd).transpose(1, 2)
        k = attn.k_proj(hidden).view(1, S, Hkv, hd).transpose(1, 2)
        v = attn.v_proj(hidden).view(1, S, Hkv, hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        ref = reference_dense_attention(
            q, k, v, doc_lengths, qa_len, attn.scaling, Hq, Hkv, gate=None
        )
        ref_out = attn.o_proj(ref.transpose(1, 2).reshape(1, S, Hq * hd))

    assert torch.allclose(patched_out, ref_out, atol=1e-4, rtol=1e-4), (
        (patched_out - ref_out).abs().max()
    )
