"""Configuration dataclasses for leefrag-v2.

Three configs:
  - V2ModelConfig:    base model + architecture constants + tokenization limits.
  - SelectorConfig:   the per-layer keep-selector (MLP + Gumbel-sigmoid gate).
  - V2TrainingConfig: optimisation, DoRA/norm tuning, budget loss, KL teacher.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class V2ModelConfig:
    model_name: str = "ldsjmdy/Tulu3-Block-FT"
    torch_dtype: str = "float16"

    # LLaMA 3.1 8B (Tulu3-Block-FT) architecture
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096

    # Tokenization limits (mirrors the v1 ModelConfig so the reused dataset works)
    max_doc_tokens: int = 1024
    max_total_doc_tokens: int = 4096
    max_question_tokens: int = 256
    max_answer_tokens: int = 512
    max_qa_tokens: int = 768

    # Attention backend used by the chunk self-attention path.
    attn_implementation: str = "flex_attention"

    @property
    def gqa_group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads


@dataclass
class SelectorConfig:
    """Per-head, query-agnostic keep-selector.

    Reads each chunk token's (post-input-layernorm) hidden state at a layer and
    emits one keep-logit per KV head. The default arch is a simple per-layer
    linear projection (hidden_size -> num_kv_heads) -- DMS-style: no trunk, no
    re-projection of the hidden dim, fewest params.
    """

    arch: str = "linear"  # "linear" (per-layer linear probe, default) | "shared_trunk" | "independent"
    trunk_dim: int = 512
    trunk_layers: int = 1
    activation: str = "gelu"
    per_layer_head: bool = True
    head_bias_init: float = 2.0  # start mostly-keep (sigmoid(2.0) ~ 0.88)

    # Gumbel-sigmoid temperature anneal (cosine from start -> end over training).
    tau_start: float = 2.0
    tau_end: float = 0.3
    tau_anneal: str = "cosine"

    # Inference gate: "topk_pi" picks the top fraction pi per layer (hard budget);
    # "threshold" keeps tokens with prob > 0.5.
    eval_mode: str = "topk_pi"

    query_agnostic: bool = True  # asserted in the selector; never reads Q+A.

    # Per-head selection: emit one keep-logit per *KV head* (num_kv_heads) per
    # token instead of one per token for the whole layer. Granularity is per KV
    # head (NOT per query head) so it respects GQA: a query-head group shares one
    # physical KV cache, so all query heads in a group share the keep-set. With a
    # global budget, KV heads may keep unequal fractions (different lengths) and
    # only the total is constrained. Eval pools logits over (head, token) for a
    # global top-k so the realized budget stays exact.
    # Default ON (DMS decides eviction per head; per-head adaptive CR is a core
    # source of its accuracy-at-budget). Requires the per-head-aware budget loss
    # (budget_mode="onesided_global"); the binomial budget is per-layer only.
    per_head: bool = True
    # Per-head eval budget rule: "equal" keeps the same count (ceil(pi*D)) in
    # every KV head -> all heads same length, exact budget, no ragged cache.
    # "global" pools (head, token) and keeps the top pi overall -> heads keep
    # unequal counts (adaptive CR) at the cost of ragged per-head lengths.
    per_head_eval: str = "equal"

    # Oracle / frozen keep-set for milestone 2 (no learned selector).
    #   None | "random_fixed" | "teacher_topk"
    oracle_keep_set: str | None = None
    freeze_selector: bool = False

    # Renormalize Q+A attention over kept chunk keys after gating. With a hard
    # gate this makes post-softmax gating identical to pre-softmax masking, i.e.
    # softmax over (kept chunk keys + Q+A) -- exactly what real KV eviction does
    # at inference. ON by default so train == eval == deploy. (False = legacy
    # "zeroed out, not renormalized", kept only for ablation.)
    gate_renorm: bool = True


@dataclass
class V2TrainingConfig:
    output_dir: str = "outputs_v2"

    # Optimisation
    num_epochs: int = 4
    batch_size: int = 1  # always 1 (variable doc counts)
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    norm_lr: float = 1e-4
    selector_lr: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = True

    # Keep-rate (compression) schedule: pi = fraction kept. 0.5/0.25/0.125 = 2x/4x/8x.
    # mode: "phases" steps through keep_rate_schedule; "linear" anneals pi in
    # log-space from keep_rate_max -> keep_rate_min over training; "sampled_range"
    # draws pi ~ log-uniform[keep_rate_min, keep_rate_max] each step so one run
    # yields adapters robust across the whole CR family (eval dials pi via topk).
    keep_rate_mode: str = "phases"
    keep_rate_schedule: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.125])
    keep_rate_min: float = 0.125
    keep_rate_max: float = 0.5
    # Eval sweeps these keep fractions (a "family of models" from one checkpoint).
    eval_pis: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.125])

    # Budget loss. "onesided_global": one-sided ReLU hinge on the TOTAL kept count
    # across all layers/heads vs pi * total_slots -- only over-budget is penalized,
    # so layers/heads/positions may compress unequally (DMS-style adaptive CR).
    # "binomial": legacy per-layer binomial NLL pinned to pi (both sides).
    budget_mode: str = "onesided_global"
    budget_weight: float = 0.1
    budget_hard_count: bool = True  # count hard STE samples (else soft concrete)
    entropy_weight: float = 0.0
    load_balance_weight: float = 0.0

    # Losses. DMS retrofits purely via logit distillation (teacher = original,
    # un-adapted, full-KV model). We follow that: KL to the offline teacher is
    # the primary objective and on by default; CE on the gold answer is a light
    # auxiliary anchor. (Requires precomputed teacher logits -- run
    # scripts/precompute_teacher.py first; otherwise KL is silently skipped.)
    ce_weight: float = 0.1
    use_kl_teacher: bool = True
    kl_weight: float = 1.0
    kl_top_k: int = 128
    teacher_dir: str = "outputs_v2/teacher"  # offline precomputed top-k logits

    # DoRA (heavy on read path q/o/FFN, light on cache path k/v)
    dora_rank_heavy: int = 32
    dora_rank_light: int = 8
    dora_alpha_heavy: int = 64
    dora_alpha_light: int = 16
    dora_dropout: float = 0.05
    dora_targets_heavy: list[str] = field(
        default_factory=lambda: ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    dora_targets_light: list[str] = field(
        default_factory=lambda: ["k_proj", "v_proj"]
    )
    train_norms: bool = True

    # Runtime
    use_flex: bool = True  # FlexAttention chunk path (GPU); False -> dense fallback (CPU)
    gradient_checkpointing: bool = True
    checkpoint_use_reentrant: bool = False
    preserve_rng_state: bool = True
    dataset_name: str = "rag_v1"  # "rag_v1" | "hotpotqa"
    eval_split_ratio: float = 0.1
    dataloader_num_workers: int = 4

    # Logging / checkpointing
    use_wandb: bool = True
    wandb_project: str = "leefrag-v2"
    logging_steps: int = 10
    eval_steps: int | None = None  # auto: steps_per_phase // 4
    save_steps: int = 500

    # Mode: "learned" (trainable selector + budget loss) | "oracle" (frozen keep-set)
    mode: str = "learned"
