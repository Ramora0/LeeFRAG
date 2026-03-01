from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "ldsjmdy/Tulu3-Block-FT"
    torch_dtype: str = "float16"
    max_doc_tokens: int = 1024
    max_total_doc_tokens: int = 4096
    max_question_tokens: int = 256
    max_answer_tokens: int = 512
    # LLaMA 3.1 8B architecture (Tulu3-Block-FT)
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096


@dataclass
class QFormerConfig:
    hidden_size: int = 1024
    num_attention_heads: int = 8
    ffn_dim: int = 2048
    lora_rank: int = 32
    dropout: float = 0.1
    max_query_tokens: int = 512
    cross_attn_mode: str = "global"  # "global" or "windowed"
    gradient_checkpointing: bool = False


@dataclass
class TrainingConfig:
    output_dir: str = "outputs"
    num_epochs: int = 4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    compression_schedule: list[int] = field(default_factory=lambda: [2, 4, 8, 16])
    eval_split_ratio: float = 0.1
    eval_steps: int = None  # auto: steps_per_epoch // 3
    save_steps: int = 500
    logging_steps: int = 10
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    use_wandb: bool = True
    kl_weight: float = 1.0  # alpha for KL divergence loss: total = CE + alpha * KL
    kl_top_k: int = 0  # if > 0, compute KL only over top-k teacher logits (memory saving)
    hidden_state_loss: bool = False  # use hidden state matching instead of KL
    hidden_state_weight: float = 1.0  # weight for hidden state matching loss
    hidden_state_layers: str = "all"  # "all" or "last_N" (e.g. "last_8")
    gradient_checkpoint_llm: bool = False
    offload_stage_a_to_cpu: bool = False
