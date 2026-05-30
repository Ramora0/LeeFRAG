"""The gated two-path attention that replaces LlamaAttention.forward.

Factorization (numerically identical to a single dense attention when gate=1):
  - Path 1: chunk self-attention (block-causal-isolated). Independent of Q+A and
    of the gate, so it uses fused FlexAttention (or a dense fallback on CPU).
  - Path 2: Q+A attention over keys = [all chunk tokens | Q+A tokens]. Softmax is
    taken over the FULL key set, THEN the per-layer keep-gate multiplies the
    post-softmax weights on the chunk-key columns (no renormalization). This is
    what lets every chunk key receive gradient even when gated to ~0.

The selector input for the budget loss is captured (detached+cloned) so the
budget loss can be computed outside the gradient-checkpointed region.
"""

from __future__ import annotations

import torch

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from leefrag_v2.model.selector import gumbel_sigmoid_ste, topk_keep_gate


def _dense_attention(q, k, v, add_mask, scaling, n_rep):
    """Standard attention with an additive mask; GQA via repeat_kv. fp32 softmax."""
    k_rep = repeat_kv(k, n_rep)
    v_rep = repeat_kv(v, n_rep)
    scores = torch.matmul(q, k_rep.transpose(-1, -2)) * scaling
    scores = scores + add_mask.to(scores.dtype)
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(probs, v_rep)


def _compute_gate(attn_module, ctx, hidden_states, layer_idx, D):
    """Return the keep-gate [1, D] (fp32) for this layer, or None (= all ones).

    Also captures the detached chunk hidden state for the decoupled budget loss.
    """
    if ctx.teacher_mode:
        return None
    if ctx.oracle_masks is not None:
        return ctx.oracle_masks[layer_idx].to(torch.float32).view(1, D)
    if ctx.selector is None:
        return None

    chunk_hidden = hidden_states[:, :D, :]
    if ctx.capture_for_budget and ctx.captured_chunk_hidden is not None:
        # detached + cloned -> safe to use for a separate backward after the
        # gradient-checkpointed forward has freed its intermediates.
        ctx.captured_chunk_hidden[layer_idx] = chunk_hidden.detach().clone()

    logits = ctx.selector(chunk_hidden, layer_idx)  # [1, D] fp32, full graph

    if attn_module.training:
        noise = ctx.noise[layer_idx] if ctx.noise is not None else None
        gate = gumbel_sigmoid_ste(logits, tau=ctx.tau, noise=noise, hard=True, training=True)
    elif ctx.eval_pi is not None:
        gate = topk_keep_gate(logits, ctx.eval_pi)
    else:
        gate = (torch.sigmoid(logits) > 0.5).float()
    return gate.view(1, D)


def patched_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,  # ignored; structure comes from ctx
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    ctx = self._v2ctx
    layer_idx = self.layer_idx
    bsz, S, _ = hidden_states.shape
    D = ctx.doc_total
    Sa = S - D

    Hq = self.config.num_attention_heads
    Hkv = self.config.num_key_value_heads
    n_rep = Hq // Hkv
    hd = self.head_dim
    scaling = self.scaling

    q = self.q_proj(hidden_states).view(bsz, S, Hq, hd).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, S, Hkv, hd).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, S, Hkv, hd).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    q_c, q_a = q[:, :, :D, :], q[:, :, D:, :]
    k_c, k_a = k[:, :, :D, :], k[:, :, D:, :]
    v_c, v_a = v[:, :, :D, :], v[:, :, D:, :]

    # ---------------- Path 1: chunk self-attention (no gate) ----------------
    if D > 0:
        if ctx.use_flex:
            from torch.nn.attention.flex_attention import flex_attention

            out_c = flex_attention(
                q_c.contiguous(),
                k_c.contiguous(),
                v_c.contiguous(),
                block_mask=ctx.block_mask,
                scale=scaling,
                enable_gqa=True,
            )
        else:
            out_c = _dense_attention(q_c, k_c, v_c, ctx.chunk_dense_mask, scaling, n_rep)
    else:
        out_c = q.new_zeros((bsz, Hq, 0, hd))

    # ---------------- Path 2: Q+A attention (gated, post-softmax) ----------------
    if Sa > 0:
        k_a_rep = repeat_kv(k_a, n_rep)
        v_a_rep = repeat_kv(v_a, n_rep)

        scores_aa = torch.matmul(q_a, k_a_rep.transpose(-1, -2)) * scaling
        scores_aa = scores_aa + ctx.qa_causal_bias.to(scores_aa.dtype)

        if D > 0:
            k_c_rep = repeat_kv(k_c, n_rep)
            v_c_rep = repeat_kv(v_c, n_rep)
            scores_ca = torch.matmul(q_a, k_c_rep.transpose(-1, -2)) * scaling
            scores = torch.cat([scores_ca, scores_aa], dim=-1)
            probs = torch.softmax(scores.float(), dim=-1)  # over the FULL key set
            probs_c, probs_a = probs[..., :D], probs[..., D:]

            gate = _compute_gate(self, ctx, hidden_states, layer_idx, D)
            if gate is not None:
                probs_c = probs_c * gate.view(1, 1, 1, D)  # post-softmax, no renorm
                if ctx.gate_renorm:  # ablation only
                    denom = (
                        probs_c.sum(-1, keepdim=True) + probs_a.sum(-1, keepdim=True)
                    ).clamp_min(1e-9)
                    probs_c, probs_a = probs_c / denom, probs_a / denom

            out_a = torch.matmul(probs_c.to(v.dtype), v_c_rep) + torch.matmul(
                probs_a.to(v.dtype), v_a_rep
            )
        else:
            probs = torch.softmax(scores_aa.float(), dim=-1)
            out_a = torch.matmul(probs.to(v.dtype), v_a_rep)
    else:
        out_a = q.new_zeros((bsz, Hq, 0, hd))

    # ---------------- recombine ----------------
    attn = torch.cat([out_c, out_a], dim=2)
    attn = attn.transpose(1, 2).reshape(bsz, S, Hq * hd)
    attn = self.o_proj(attn)
    return attn, None


def reference_dense_attention(
    q, k, v, block_lengths, qa_len, scaling, num_q_heads, num_kv_heads, gate=None
):
    """Ground-truth single dense attention for parity checks (milestone 1).

    q/k/v are already RoPE'd, shapes [1, Hq/Hkv, S, hd]. Builds the full
    [preamble|docs|Q+A] mask via the trusted v1 builder, optionally applies the
    post-softmax gate to the Q+A->chunk block.
    """
    from leefrag.model.block_attention import build_block_causal_mask_with_qa

    D = sum(block_lengths)
    mask = build_block_causal_mask_with_qa(
        block_lengths, qa_len, dtype=torch.float32, device=q.device
    )  # [1,1,S,S]
    n_rep = num_q_heads // num_kv_heads
    k_rep = repeat_kv(k, n_rep)
    v_rep = repeat_kv(v, n_rep)
    scores = torch.matmul(q, k_rep.transpose(-1, -2)) * scaling + mask
    probs = torch.softmax(scores.float(), dim=-1)
    if gate is not None:
        mult = torch.ones_like(probs)
        mult[:, :, D:, :D] = gate.view(1, 1, 1, D)
        probs = probs * mult
    return torch.matmul(probs.to(v.dtype), v_rep)
