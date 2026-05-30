"""CPU integration: full single forward through a tiny patched LlamaForCausalLM.

Exercises new_context -> set_context -> patched attention (dense fallback) inside
HF's LlamaModel.forward -> CE + budget loss -> backward -> selector grads.
No CUDA / peft needed.
"""

import torch

from leefrag_v2.config import SelectorConfig
from leefrag_v2.model.patch import (
    install_gated_attention,
    new_context,
    sample_step_noise,
    set_context,
)
from leefrag_v2.model.selector import LayerSelector
from leefrag_v2.training.budget import budget_binomial_loss
from leefrag_v2.training.losses import ce_on_answer


def test_full_forward_backward_cpu():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=64, num_attention_heads=8, num_key_value_heads=2, head_dim=8,
        num_hidden_layers=3, intermediate_size=128, vocab_size=256,
        max_position_embeddings=512,
    )
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(cfg).float()
    model.train()
    install_gated_attention(model)

    selector = LayerSelector(
        SelectorConfig(trunk_dim=32, per_head=False),
        cfg.num_hidden_layers, cfg.hidden_size,
    ).float()
    selector.train()

    doc_lengths = [4, 3]
    D = sum(doc_lengths)
    qa_len = 5
    S = D + qa_len
    input_ids = torch.randint(0, cfg.vocab_size, (1, S))
    labels = torch.full((1, S), -100)
    labels[0, D + 2 :] = input_ids[0, D + 2 :]  # supervise a few answer tokens

    noise = sample_step_noise(cfg.num_hidden_layers, D, "cpu")
    ctx = new_context(
        doc_lengths, qa_len, "cpu", cfg.num_hidden_layers, use_flex=False,
        selector=selector, tau=1.0, capture_for_budget=True, noise=noise,
    )
    set_context(model, ctx)

    out = model(input_ids=input_ids, use_cache=False)
    assert out.logits.shape == (1, S, cfg.vocab_size)

    ce = ce_on_answer(out.logits, labels)
    budget, keep = budget_binomial_loss(
        ctx.captured_chunk_hidden, selector, 0.25, 1.0, ctx.noise, "cpu"
    )
    loss = ce + 0.1 * budget
    assert torch.isfinite(loss)
    loss.backward()

    grads = [
        p.grad for p in selector.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(grads) > 0, "selector received no gradient"
    # captured one chunk-hidden per layer
    assert sum(h is not None for h in ctx.captured_chunk_hidden) == cfg.num_hidden_layers
    print(f"PASS test_full_forward_backward_cpu (ce={ce.item():.3f}, keep={keep:.3f})")


if __name__ == "__main__":
    test_full_forward_backward_cpu()
