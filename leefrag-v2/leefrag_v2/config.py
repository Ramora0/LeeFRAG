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
    """Per-layer, query-agnostic keep-selector.

    Reads each chunk token's (post-input-layernorm) hidden state at a layer and
    emits one keep-logit. A shared trunk + per-layer head keeps params small
    while letting each layer choose different tokens.
    """

    arch: str = "shared_trunk"  # "shared_trunk" | "independent"
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

    # Oracle / frozen keep-set for milestone 2 (no learned selector).
    #   None | "random_fixed" | "teacher_topk"
    oracle_keep_set: str | None = None
    freeze_selector: bool = False

    # Ablation only: renormalize Q+A attention over kept chunk keys. Default OFF
    # per the spec ("zeroed out", not renormalized).
    gate_renorm: bool = False


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
    keep_rate_schedule: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.125])

    # Budget loss: binomial NLL of the sampled keep-count vs prior rate pi.
    budget_weight: float = 0.1
    budget_hard_count: bool = True  # count hard STE samples (else soft concrete)
    entropy_weight: float = 0.0
    load_balance_weight: float = 0.0

    # Losses
    ce_weight: float = 1.0
    use_kl_teacher: bool = False
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
