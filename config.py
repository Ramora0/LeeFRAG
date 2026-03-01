from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    torch_dtype: str = "float16"
    max_doc_tokens: int = 1024
    max_total_doc_tokens: int = 4096
    max_question_tokens: int = 256
    max_answer_tokens: int = 512
    # LLaMA 3 8B architecture
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096


@dataclass
class QFormerConfig:
    num_qformer_layers: int = 8
    layers_per_group: int = 4  # 8 * 4 = 32 LLM layers
    hidden_size: int = 1024  # 8 kv_heads * 128 head_dim
    num_attention_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    max_query_tokens: int = 512
    cross_attn_mode: str = "global"  # "global" or "windowed"
    gradient_checkpointing: bool = True


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
    eval_steps: int = 200
    save_steps: int = 500
    logging_steps: int = 10
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    use_wandb: bool = True
    kl_weight: float = 1.0  # alpha for KL divergence loss: total = CE + alpha * KL
    kl_top_k: int = 0  # if > 0, compute KL only over top-k teacher logits (memory saving)
    gradient_checkpoint_llm: bool = False
    offload_stage_a_to_cpu: bool = False
