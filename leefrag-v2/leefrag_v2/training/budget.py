"""Keep-rate (compression) schedule and the budget loss.

The budget loss pushes the per-token keep probabilities toward a Bernoulli prior
with mean pi (the target keep fraction), annealed 0.5 -> 0.25 -> 0.125 across
phases. It is computed on the captured (detached) chunk hidden states so its
gradient flows only into the selector, not the LLM.
"""

from __future__ import annotations

import torch


class KeepRateScheduler:
    """Maps a training step to the target keep fraction pi (= 1 / compression)."""

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


def budget_kl_loss(
    captured_chunk_hidden: list,
    selector,
    pi: float,
    device,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, float]:
    """Per-token KL(Bernoulli(p) || Bernoulli(pi)) averaged over tokens and layers.

    Returns (loss, mean_keep_prob). `selector` is re-applied to the captured
    (detached) hidden states so gradient reaches the selector only.
    """
    pi_t = min(1.0 - eps, max(eps, pi))
    total = torch.zeros((), device=device)
    mean_keep = 0.0
    n = 0
    for layer_idx, hidden in enumerate(captured_chunk_hidden):
        if hidden is None:
            continue
        logits = selector(hidden, layer_idx)  # [1, D], grad -> selector only
        p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
        kl = p * torch.log(p / pi_t) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - pi_t))
        total = total + kl.mean()
        mean_keep += p.mean().item()
        n += 1
    if n == 0:
        return total, 0.0
    return total / n, mean_keep / n
