"""CPU tests for the binomial budget loss.

Verifies it is minimized at keep-count = D*pi (targets the prior from both sides)
and that gradient reaches the selector.
"""

import torch

from leefrag_v2.config import SelectorConfig
from leefrag_v2.model.selector import LayerSelector
from leefrag_v2.training.budget import KeepRateScheduler, budget_binomial_loss


def test_cr_linear_matches_dms():
    """DMS schedule: CR(t) = 1 + t/cr_steps_per_unit, held at the target; pi = 1/CR.

    Warm-starts at CR=1 (pi=1.0), steps linearly in CR space, holds at 1/pi_min.
    """
    sched = [1.0, 0.5, 0.25, 0.125]
    s = KeepRateScheduler(
        sched, total_steps=2000, mode="cr_linear",
        pi_min=0.125, pi_max=1.0, cr_steps_per_unit=100.0,
    )
    assert not s.is_phased  # continuous ramp -> no per-phase LR restarts
    assert abs(s.get_pi(0) - 1.0) < 1e-9          # warm start: keep everything
    assert abs(s.get_pi(100) - 0.5) < 1e-9        # 100 steps/CR -> CR 2
    assert abs(1 / s.get_pi(300) - 4.0) < 1e-6    # CR 4
    assert abs(1 / s.get_pi(700) - 8.0) < 1e-6    # reaches target CR 8
    assert abs(1 / s.get_pi(2000) - 8.0) < 1e-6   # and HOLDS there

    # keep-rate is monotone non-increasing along the ramp
    prev = 2.0
    for t in range(0, 2000, 7):
        cur = s.get_pi(t)
        assert cur <= prev + 1e-12
        prev = cur

    # auto-span: a short run still reaches the target CR by its last step
    s2 = KeepRateScheduler(
        sched, total_steps=300, mode="cr_linear",
        pi_min=0.125, pi_max=1.0, cr_steps_per_unit=None,
    )
    assert abs(s2.get_pi(0) - 1.0) < 1e-9
    assert abs(1 / s2.get_pi(299) - 8.0) < 1e-6

    # phases mode (oracle path) is unchanged: discrete, constant pi, is_phased
    s3 = KeepRateScheduler([0.25], total_steps=400, mode="phases")
    assert s3.is_phased and all(s3.get_pi(t) == 0.25 for t in (0, 100, 399))
    print("PASS test_cr_linear_matches_dms")


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


def _const_logit_selector_perhead(bias: float, hidden_size: int, num_kv_heads: int):
    """Per-head const-logit selector: every (head, token) logit == `bias`."""
    s = LayerSelector(
        SelectorConfig(trunk_dim=16, head_bias_init=bias, per_head=True),
        1, hidden_size, num_kv_heads=num_kv_heads,
    )
    with torch.no_grad():
        for p in s.parameters():
            if p.dim() > 1:  # zero weight matrices, keep biases
                p.zero_()
    return s


def test_binomial_budget_per_head():
    """Per-head binomial: each KV head is its own Binomial(D, pi), pinned to pi.

    Regression guard: before the per-head fix this returned NaN because K summed
    over all Hkv*D gates while D was only the token count.
    """
    torch.manual_seed(0)
    H, D, Hkv = 64, 40, 8
    captured = [torch.randn(1, D, H)]
    noise = [torch.zeros(1, Hkv, D)]  # deterministic, per-head shape
    pi = 0.5

    l_at_target, keep_t = budget_binomial_loss(
        captured, _const_logit_selector_perhead(0.0, H, Hkv), pi, tau=1.0,
        noise=noise, device="cpu", hard=False,
    )
    assert torch.isfinite(l_at_target), l_at_target  # was NaN before the fix
    assert abs(keep_t - 0.5) < 0.05  # every head near the prior at bias=0

    l_far, keep_f = budget_binomial_loss(
        captured, _const_logit_selector_perhead(3.0, H, Hkv), pi, tau=1.0,
        noise=noise, device="cpu", hard=False,
    )
    assert keep_f > keep_t
    assert l_far > l_at_target  # over-keeping every head is penalized

    # gradient reaches the selector
    s = _const_logit_selector_perhead(3.0, H, Hkv)
    loss, _ = budget_binomial_loss(captured, s, pi, 1.0, noise, "cpu", hard=False)
    loss.backward()
    grads = [p.grad for p in s.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(grads) > 0
    print(f"PASS test_binomial_budget_per_head (keep@bias0={keep_t:.3f})")


if __name__ == "__main__":
    test_cr_linear_matches_dms()
    test_binomial_budget_targets_prior()
    test_binomial_budget_per_head()
