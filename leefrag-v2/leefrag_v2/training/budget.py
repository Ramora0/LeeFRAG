"""Keep-rate (compression) schedule and the binomial budget loss.

The budget loss treats the realized keep-count K = sum of the SAMPLED gates at a
layer as a draw from Binomial(D, pi) and penalizes its negative log-likelihood,
  -log Binom(K; D, pi) = -[ logC(D,K) + K*log(pi) + (D-K)*log(1-pi) ],
using lgamma for a differentiable continuous relaxation. This is minimized at
K = D*pi (the binomial mode), so it pulls the keep-count toward the prior rate
from both sides — unlike a per-sample Bernoulli cross-entropy, which collapses.

Gradient flows only into the selector: we re-sample on the captured (detached)
chunk hidden states using the SAME pre-sampled noise as the forward pass, so the
penalized count matches the gate that was actually applied in attention.
"""

from __future__ import annotations

import torch

from leefrag_v2.model.selector import gumbel_sigmoid_ste


class KeepRateScheduler:
    """Maps a training step to the prior keep fraction pi (= 1 / compression)."""

    def __init__(self, schedule: list[float], total_steps: int):
        assert len(schedule) > 0
        self.schedule = schedule
        self.total_steps = total_steps
        self.steps_per_phase = max(1, total_steps // len(schedule))

    def get_pi(self, step: int) -> float:
        return self.schedule[self.get_phase(step)]

    def get_phase(self, step: int) -> int:
        return min(step // self.steps_per_phase, len(self.schedule) - 1)

    @property
    def num_phases(self) -> int:
        return len(self.schedule)


def budget_binomial_loss(
    captured_chunk_hidden: list,
    selector,
    pi: float,
    tau: float,
    noise: list | None,
    device,
    hard: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Mean over layers of -log Binomial(K_l; D, pi), K_l = sum of sampled gates.

    Returns (loss, mean_keep_fraction). `selector` is re-applied to the captured
    (detached) hidden states with the same `noise` as the forward, so gradient
    reaches the selector only and the penalized count matches the applied gate.
    """
    pi_t = min(1.0 - eps, max(eps, pi))
    log_pi = torch.log(torch.tensor(pi_t, device=device))
    log_1mpi = torch.log(torch.tensor(1.0 - pi_t, device=device))

    total = torch.zeros((), device=device)
    mean_keep = 0.0
    n = 0
    for layer_idx, hidden in enumerate(captured_chunk_hidden):
        if hidden is None:
            continue
        logits = selector(hidden, layer_idx)  # [1, D], grad -> selector only
        D = logits.shape[-1]
        layer_noise = noise[layer_idx] if noise is not None else None
        g = gumbel_sigmoid_ste(logits, tau=tau, noise=layer_noise, hard=hard, training=True)
        K = g.sum()  # realized keep-count (differentiable via STE / concrete sample)

        Dt = torch.tensor(float(D), device=device)
        log_comb = torch.lgamma(Dt + 1) - torch.lgamma(K + 1) - torch.lgamma(Dt - K + 1)
        log_pmf = log_comb + K * log_pi + (Dt - K) * log_1mpi
        total = total + (-(log_pmf) / Dt)  # normalize by D for a per-token scale

        mean_keep += (K.item() / D)
        n += 1

    if n == 0:
        return total, 0.0
    return total / n, mean_keep / n
