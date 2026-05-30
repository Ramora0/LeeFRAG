"""Keep-rate (compression) schedule and the binomial budget loss.

The budget loss treats the realized keep-count K = sum of the SAMPLED gates over
tokens -- for the whole layer (per-layer selector) or for each KV head (per-head
selector) -- as a draw from Binomial(D, pi) and penalizes its negative log-
likelihood,
  -log Binom(K; D, pi) = -[ logC(D,K) + K*log(pi) + (D-K)*log(1-pi) ],
using lgamma for a differentiable continuous relaxation. This is minimized at
K = D*pi (the binomial mode), so it pulls the keep-count toward the prior rate
from both sides — unlike a per-sample Bernoulli cross-entropy, which collapses.

Gradient flows only into the selector: we re-sample on the captured (detached)
chunk hidden states using the SAME pre-sampled noise as the forward pass, so the
penalized count matches the gate that was actually applied in attention.
"""

from __future__ import annotations

import math

import torch

from leefrag_v2.model.selector import gumbel_sigmoid_ste


class KeepRateScheduler:
    """Maps a training step to the prior keep fraction pi (= 1 / compression).

    mode:
      "cr_linear"     - DMS's schedule (default): ramp the compression ratio
                        LINEARLY from start_cr = 1/pi_max up to target_cr = 1/pi_min,
                        then hold. DMS uses CR(t) = 1 + t/100 ("100 optimiser steps
                        per unit of CR"); `cr_steps_per_unit` sets that rate, or
                        None auto-spans the whole run. Since pi = 1/CR, the keep-
                        rate decays harmonically -- more steps land at the higher
                        compressions -- and (with pi_max=1.0) training warm-starts
                        at CR=1, keeping everything before any eviction pressure.
      "phases"        - step through `schedule` (discrete CR jumps).
      "linear"        - anneal pi in log-space from pi_max -> pi_min over training
                        (CR moves smoothly; one run sweeps the whole family).
      "sampled_range" - draw pi ~ log-uniform[pi_min, pi_max] each step so the
                        adapters generalize across CRs (eval dials pi via top-k).
    """

    def __init__(
        self,
        schedule: list[float],
        total_steps: int,
        mode: str = "cr_linear",
        pi_min: float = 0.125,
        pi_max: float = 1.0,
        cr_steps_per_unit: float | None = 100.0,
    ):
        assert len(schedule) > 0
        self.schedule = schedule
        self.total_steps = total_steps
        self.mode = mode
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.cr_steps_per_unit = cr_steps_per_unit
        self.start_cr = 1.0 / max(1e-6, pi_max)   # pi_max=1.0 -> CR 1 (DMS warm start)
        self.target_cr = 1.0 / max(1e-6, pi_min)  # pi_min=0.125 -> CR 8
        self.steps_per_phase = max(1, total_steps // len(schedule))

    @property
    def is_phased(self) -> bool:
        """Only the discrete-phase schedule drives per-phase LR restarts."""
        return self.mode == "phases"

    def get_pi(self, step: int) -> float:
        if self.mode == "cr_linear":
            spu = self.cr_steps_per_unit
            if spu is None:  # auto: span start_cr -> target_cr across the whole run
                spu = max(
                    1.0,
                    (self.total_steps - 1) / max(1e-6, self.target_cr - self.start_cr),
                )
            cr = min(self.target_cr, self.start_cr + step / spu)
            return 1.0 / cr
        if self.mode == "linear":
            t = min(1.0, step / max(1, self.total_steps - 1))
            return math.exp(math.log(self.pi_max) * (1 - t) + math.log(self.pi_min) * t)
        if self.mode == "sampled_range":
            u = float(torch.rand(()).item())
            return math.exp(math.log(self.pi_min) * (1 - u) + math.log(self.pi_max) * u)
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
    """Mean over layers (and KV heads, if per-head) of -log Binomial(K; D, pi).

    K is the keep-count summed over TOKENS per binomial row: the whole layer for a
    per-layer selector ([1, D]), or each KV head independently for a per-head
    selector ([1, Hkv, D]). Each row is its own Binomial(D, pi); the per-row NLL
    is averaged over rows (so per-head pins every KV head to keep-rate pi).

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
        logits = selector(hidden, layer_idx)  # [1, D] per-layer or [1, Hkv, D] per-head
        D = logits.shape[-1]  # tokens = number of Bernoulli slots in each binomial row
        layer_noise = noise[layer_idx] if noise is not None else None
        g = gumbel_sigmoid_ste(logits, tau=tau, noise=layer_noise, hard=hard, training=True)
        # Keep-count per binomial "row": sum over TOKENS only. A per-layer gate has
        # one row (the layer); a per-head gate has one row per KV head, each its own
        # Binomial(D, pi). Summing over tokens (not all gates) keeps K <= D per row.
        K = g.sum(dim=-1)  # [1] (per-layer) or [1, Hkv] (per-head)

        Dt = torch.tensor(float(D), device=device)
        log_comb = torch.lgamma(Dt + 1) - torch.lgamma(K + 1) - torch.lgamma(Dt - K + 1)
        log_pmf = log_comb + K * log_pi + (Dt - K) * log_1mpi
        total = total + (-(log_pmf) / Dt).mean()  # per-token scale, mean over rows/heads

        mean_keep += (K / D).mean().item()
        n += 1

    if n == 0:
        return total, 0.0
    return total / n, mean_keep / n


def budget_onesided_global(
    captured_chunk_hidden: list,
    selector,
    pi: float,
    tau: float,
    noise: list | None,
    device,
    hard: bool = True,
) -> tuple[torch.Tensor, float]:
    """One-sided global budget: penalize only TOTAL kept > pi * total_slots.

    Sums the sampled keep-count over every layer (and KV head, when the selector
    is per-head), then applies a ReLU hinge on the aggregate. Because nothing
    pins an individual layer/head/position to pi, the model is free to compress
    unequally (DMS-style adaptive CR) as long as the total stays under budget.

    Gradient reaches the selector only: it is re-applied to the captured
    (detached) hidden states with the SAME noise as the forward pass, so the
    penalized count matches the gate that was actually applied in attention.
    """
    total_kept = torch.zeros((), device=device)
    total_slots = 0
    for layer_idx, hidden in enumerate(captured_chunk_hidden):
        if hidden is None:
            continue
        logits = selector(hidden, layer_idx)  # [1, D] or [1, Hkv, D]
        layer_noise = noise[layer_idx] if noise is not None else None
        g = gumbel_sigmoid_ste(
            logits, tau=tau, noise=layer_noise, hard=hard, training=True
        )
        total_kept = total_kept + g.sum()
        total_slots += g.numel()

    if total_slots == 0:
        return total_kept, 0.0
    target = pi * total_slots
    loss = torch.relu(total_kept - target) / total_slots
    return loss, float(total_kept.item() / total_slots)
