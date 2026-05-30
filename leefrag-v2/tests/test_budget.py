"""CPU tests for the binomial budget loss.

Verifies it is minimized at keep-count = D*pi (targets the prior from both sides)
and that gradient reaches the selector.
"""

import torch

from leefrag_v2.config import SelectorConfig
from leefrag_v2.model.selector import LayerSelector
from leefrag_v2.training.budget import budget_binomial_loss


def _const_logit_selector(bias: float, hidden_size: int):
    """Selector whose logit == `bias` for every token (zero all weights, keep biases)."""
    s = LayerSelector(
        SelectorConfig(trunk_dim=16, head_bias_init=bias, per_head=False), 1, hidden_size
    )
    with torch.no_grad():
        for p in s.parameters():
            if p.dim() > 1:  # weight matrices -> 0  (biases, incl. head bias, kept)
                p.zero_()
    return s


def test_binomial_budget_targets_prior():
    torch.manual_seed(0)
    H, D = 64, 40
    captured = [torch.randn(1, D, H)]
    noise = [torch.zeros(1, D)]  # deterministic
    pi = 0.5  # target keep-count = 20

    # soft count K = D * sigmoid(bias): bias=0 -> K=20 (=D*pi), bias=3 -> K~38
    l_at_target, keep_t = budget_binomial_loss(
        captured, _const_logit_selector(0.0, H), pi, tau=1.0, noise=noise,
        device="cpu", hard=False,
    )
    l_far, keep_f = budget_binomial_loss(
        captured, _const_logit_selector(3.0, H), pi, tau=1.0, noise=noise,
        device="cpu", hard=False,
    )
    assert keep_t < keep_f
    assert l_at_target < l_far, (l_at_target.item(), l_far.item())
    assert abs(keep_t - 0.5) < 0.05  # near the prior at bias=0

    # symmetric: dropping too much (bias<0) is also penalized more than the target
    l_low, _ = budget_binomial_loss(
        captured, _const_logit_selector(-3.0, H), pi, tau=1.0, noise=noise,
        device="cpu", hard=False,
    )
    assert l_at_target < l_low

    # gradient reaches the selector
    s = _const_logit_selector(3.0, H)
    loss, _ = budget_binomial_loss(captured, s, pi, 1.0, noise, "cpu", hard=False)
    loss.backward()
    grads = [p.grad for p in s.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(grads) > 0
    print(f"PASS test_binomial_budget_targets_prior (keep@bias0={keep_t:.3f})")


if __name__ == "__main__":
    test_binomial_budget_targets_prior()
