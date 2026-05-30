"""Monkeypatch installer + per-step forward context for the gated attention.

We patch the LlamaAttention *module* forward (not a registered attention
function) because only the module forward sees the pre-projection hidden states
the selector needs. The per-step context (doc layout, masks, selector, gate
settings) is attached to every attention module as `._v2ctx`.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field

import torch

from leefrag_v2.model import flex_masks
from leefrag_v2.model.gated_attention import patched_llama_attention_forward
from leefrag_v2.model.selector import sample_logistic_noise


@dataclass
class ForwardContext:
    """Everything the patched attention needs for one forward pass."""

    doc_total: int
    doc_lengths: list[int]
    num_layers: int
    use_flex: bool

    # masks
    block_mask: object | None = None          # FlexAttention BlockMask (chunk path)
    chunk_dense_mask: torch.Tensor | None = None  # dense fallback [1,1,D,D]
    qa_causal_bias: torch.Tensor | None = None    # [1,1,Sa,Sa]

    # gate
    selector: object | None = None
    tau: float = 1.0
    noise: list | None = None                 # per-layer pre-sampled logistic noise [1,D]
    teacher_mode: bool = False                 # True -> gate = all ones (full KV)
    oracle_masks: list | None = None           # per-layer fixed hard gate [D]
    eval_pi: float | None = None               # eval top-k keep fraction
    gate_renorm: bool = False

    # budget-loss decoupling
    capture_for_budget: bool = False
    captured_chunk_hidden: list | None = None  # filled per layer (detached+cloned)


def _is_llama_attention(module) -> bool:
    if type(module).__name__ == "LlamaAttention":
        return True
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention

        return isinstance(module, LlamaAttention)
    except Exception:
        return False


def install_gated_attention(model) -> list:
    """Patch every LlamaAttention.forward in-place. Returns the patched modules."""
    patched = []
    for module in model.modules():
        if _is_llama_attention(module):
            if not hasattr(module, "_v2_orig_forward"):
                module._v2_orig_forward = module.forward
            module.forward = types.MethodType(patched_llama_attention_forward, module)
            module._v2ctx = None
            patched.append(module)
    if not patched:
        raise RuntimeError("No LlamaAttention modules found to patch.")
    model._v2_patched_attn = patched
    return patched


def uninstall_gated_attention(model) -> None:
    for module in getattr(model, "_v2_patched_attn", []):
        if hasattr(module, "_v2_orig_forward"):
            module.forward = module._v2_orig_forward
            del module._v2_orig_forward
        if hasattr(module, "_v2ctx"):
            del module._v2ctx
    if hasattr(model, "_v2_patched_attn"):
        del model._v2_patched_attn


def set_context(model, ctx: ForwardContext | None) -> None:
    """Attach (or clear) the forward context on all patched attention modules."""
    for module in getattr(model, "_v2_patched_attn", []):
        module._v2ctx = ctx


def sample_step_noise(num_layers: int, D: int, device, generator=None) -> list:
    """Pre-sample per-layer logistic noise [1, D] (so checkpoint recompute replays it)."""
    return [
        sample_logistic_noise((1, D), device, generator=generator)
        for _ in range(num_layers)
    ]


def new_context(
    doc_lengths: list[int],
    qa_len: int,
    device,
    num_layers: int,
    *,
    use_flex: bool,
    selector=None,
    tau: float = 1.0,
    teacher_mode: bool = False,
    oracle_masks: list | None = None,
    eval_pi: float | None = None,
    gate_renorm: bool = False,
    capture_for_budget: bool = False,
    noise: list | None = None,
    mask_dtype: torch.dtype = torch.float32,
) -> ForwardContext:
    """Build a ForwardContext with masks (and noise) ready for one forward."""
    D = sum(doc_lengths)

    block_mask = None
    chunk_dense_mask = None
    if D > 0:
        if use_flex:
            block_mask = flex_masks.build_chunk_block_mask(doc_lengths, device)
        else:
            chunk_dense_mask = flex_masks.build_block_causal_isolated_dense(
                doc_lengths, dtype=mask_dtype, device=device
            )
    qa_causal_bias = (
        flex_masks.build_qa_causal_bias(qa_len, dtype=mask_dtype, device=device)
        if qa_len > 0
        else None
    )

    captured = [None] * num_layers if capture_for_budget else None

    return ForwardContext(
        doc_total=D,
        doc_lengths=doc_lengths,
        num_layers=num_layers,
        use_flex=use_flex,
        block_mask=block_mask,
        chunk_dense_mask=chunk_dense_mask,
        qa_causal_bias=qa_causal_bias,
        selector=selector,
        tau=tau,
        noise=noise,
        teacher_mode=teacher_mode,
        oracle_masks=oracle_masks,
        eval_pi=eval_pi,
        gate_renorm=gate_renorm,
        capture_for_budget=capture_for_budget,
        captured_chunk_hidden=captured,
    )
