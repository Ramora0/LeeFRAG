"""Per-layer, query-agnostic keep-selector and the Gumbel-sigmoid STE gate.

The selector reads each chunk token's hidden state at a given layer and emits one
keep-logit. Keep-prob is sigmoid(logit). During training the gate is a
straight-through Gumbel-sigmoid sample (hard forward, soft backward); at eval we
take the top fraction pi per layer for a hard memory budget.

Design notes:
  - The selector NEVER sees Q+A tokens (query-agnostic) — callers pass only the
    chunk slice of the hidden states.
  - Gumbel noise is passed in (pre-sampled per step) so gradient-checkpoint
    recompute replays the identical gate. If noise is None it is sampled inline.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from leefrag_v2.config import SelectorConfig

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


def sample_logistic_noise(
    shape, device, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Logistic(0,1) noise = logit(U), the binary-concrete / Gumbel-sigmoid noise."""
    u = torch.rand(shape, device=device, dtype=torch.float32, generator=generator)
    u = u.clamp_(1e-6, 1.0 - 1e-6)
    return torch.log(u) - torch.log1p(-u)


def gumbel_sigmoid_ste(
    logits: torch.Tensor,
    tau: float,
    noise: torch.Tensor | None = None,
    hard: bool = True,
    training: bool = True,
) -> torch.Tensor:
    """Binary-concrete relaxation with optional straight-through hard forward.

    Returns a gate in [0,1] (soft) or {0,1} (hard, with soft gradient via STE).
    Computed in fp32 for stability; caller casts as needed.
    """
    logits = logits.float()
    if training:
        if noise is None:
            noise = sample_logistic_noise(logits.shape, logits.device)
        soft = torch.sigmoid((logits + noise.float()) / tau)
    else:
        # deterministic at eval (no noise); threshold-based callers use this
        soft = torch.sigmoid(logits / tau)
    if hard:
        hard_val = (soft > 0.5).to(soft.dtype)
        # group (soft - soft.detach()) so it is exactly 0.0 in fp -> forward == hard
        return hard_val + (soft - soft.detach())
    return soft


def topk_keep_gate(logits: torch.Tensor, pi: float) -> torch.Tensor:
    """Hard top-fraction gate for eval: keep ceil(pi * D) tokens by logit.

    logits: [B, D] -> gate: [B, D] in {0,1}. Used for a fixed memory budget.
    """
    B, D = logits.shape
    k = max(1, int(round(pi * D)))
    k = min(k, D)
    gate = torch.zeros_like(logits)
    idx = logits.topk(k, dim=-1).indices
    gate.scatter_(-1, idx, 1.0)
    return gate


class LayerSelector(nn.Module):
    """Shared trunk + per-layer head (or fully independent per-layer MLPs).

    Kept in fp32 (small) regardless of the LLM dtype; callers pass fp16 hidden
    states which are upcast internally.
    """

    def __init__(self, cfg: SelectorConfig, num_layers: int, hidden_size: int):
        super().__init__()
        self.cfg = cfg
        self.num_layers = num_layers
        act = _ACT[cfg.activation]

        if cfg.arch == "shared_trunk":
            trunk: list[nn.Module] = []
            in_dim = hidden_size
            for _ in range(cfg.trunk_layers):
                trunk += [nn.Linear(in_dim, cfg.trunk_dim), act()]
                in_dim = cfg.trunk_dim
            self.trunk = nn.Sequential(*trunk)
            self.heads = nn.ModuleList(
                [nn.Linear(cfg.trunk_dim, 1) for _ in range(num_layers)]
            )
            self.nets = None
        elif cfg.arch == "independent":
            self.trunk = None
            self.heads = None
            self.nets = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size, cfg.trunk_dim),
                        act(),
                        nn.Linear(cfg.trunk_dim, 1),
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown selector arch: {cfg.arch!r}")

        self.float()  # keep selector in fp32
        self._init_bias(cfg.head_bias_init)

    def _init_bias(self, bias: float) -> None:
        """Start mostly-keep: bias the final layer so sigmoid(logit) ~ high."""
        if self.heads is not None:
            for h in self.heads:
                nn.init.constant_(h.bias, bias)
        if self.nets is not None:
            for net in self.nets:
                nn.init.constant_(net[-1].bias, bias)

    def forward(self, chunk_hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """chunk_hidden: [B, D, H] (chunk tokens only) -> logits: [B, D]."""
        x = chunk_hidden.float()
        if self.cfg.arch == "shared_trunk":
            logit = self.heads[layer_idx](self.trunk(x))
        else:
            logit = self.nets[layer_idx](x)
        return logit.squeeze(-1)
