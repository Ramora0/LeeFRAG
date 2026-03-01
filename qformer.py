import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig, QFormerConfig


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(xW1) * xW2."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


def _build_windowed_cross_attn_mask(
    num_queries: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a mask where each query attends only to its local window of inputs.

    Args:
        num_queries: Number of query tokens.
        kv_len: Document length.
        dtype: Mask dtype.
        device: Target device.

    Returns:
        mask: [1, 1, num_queries, kv_len]  (0.0 = attend, -inf = masked)
    """
    stride = kv_len / num_queries
    window_size = math.ceil(stride) + 1

    q_idx = torch.arange(num_queries, device=device).unsqueeze(1)  # [Q, 1]
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)      # [1, D]

    centers = q_idx.float() * stride + stride / 2  # [Q, 1]
    half_win = window_size / 2

    in_window = (kv_idx.float() >= (centers - half_win)) & (kv_idx.float() < (centers + half_win))
    mask = torch.where(in_window, 0.0, float("-inf")).to(dtype=dtype)  # [Q, D]
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, D]


class QFormerTrunk(nn.Module):
    """Shared trunk: self-attention among queries, cross-attention to hidden states, FFN.

    Cross-attention is asymmetric: queries at hidden_size, keys/values
    projected directly from LLM hidden_size (4096) — no premature bottleneck.
    """

    def __init__(self, config: QFormerConfig, model_config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Layer norms (pre-norm architecture)
        self.self_attn_ln = nn.LayerNorm(self.hidden_size)
        self.cross_attn_ln = nn.LayerNorm(self.hidden_size)
        self.cross_attn_kv_ln = nn.LayerNorm(model_config.hidden_size)
        self.ffn_ln = nn.LayerNorm(self.hidden_size)

        # Self-attention (at hidden_size)
        self.self_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.self_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.self_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.self_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.self_attn_dropout = nn.Dropout(config.dropout)

        # Cross-attention (asymmetric: queries from hidden_size, K/V from LLM hidden_size)
        self.cross_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.cross_k = nn.Linear(model_config.hidden_size, self.hidden_size, bias=False)
        self.cross_v = nn.Linear(model_config.hidden_size, self.hidden_size, bias=False)
        self.cross_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.cross_attn_dropout = nn.Dropout(config.dropout)

        # FFN
        self.ffn = SwiGLU(self.hidden_size, config.ffn_dim, config.dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_tokens: [num_layers, num_queries, hidden_size]
            hidden_states: [num_layers, doc_len, llm_hidden_size (4096)]

        Returns:
            query_tokens: [num_layers, num_queries, hidden_size]
        """
        batch, num_q, _ = query_tokens.shape
        hs_len = hidden_states.shape[1]

        # 1. Self-attention
        residual = query_tokens
        x = self.self_attn_ln(query_tokens)
        q = self._reshape_heads(self.self_q(x), batch, num_q)
        k = self._reshape_heads(self.self_k(x), batch, num_q)
        v = self._reshape_heads(self.self_v(x), batch, num_q)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(batch, num_q, self.hidden_size)
        query_tokens = residual + self.self_attn_dropout(self.self_o(attn_out))

        # 2. Cross-attention (asymmetric: K/V projected from 4096-dim hidden states)
        residual = query_tokens
        x = self.cross_attn_ln(query_tokens)
        hs_input = self.cross_attn_kv_ln(hidden_states)
        q = self._reshape_heads(self.cross_q(x), batch, num_q)
        k = self._reshape_heads(self.cross_k(hs_input), batch, hs_len)
        v = self._reshape_heads(self.cross_v(hs_input), batch, hs_len)

        cross_attn_mask = None
        if self.config.cross_attn_mode == "windowed":
            cross_attn_mask = _build_windowed_cross_attn_mask(
                num_q, hs_len, dtype=q.dtype, device=q.device,
            )

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=cross_attn_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, num_q, self.hidden_size)
        query_tokens = residual + self.cross_attn_dropout(self.cross_o(attn_out))

        # 3. FFN
        residual = query_tokens
        query_tokens = residual + self.ffn(self.ffn_ln(query_tokens))

        return query_tokens

    def _reshape_heads(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """Reshape [batch, seq, hidden] -> [batch, heads, seq, head_dim]."""
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


class OutputKVHeads(nn.Module):
    """Shared base K/V projections + per-layer LoRA deltas.

    W_i = W_shared + A_i @ B_i

    B is initialized to zero so all layers start identical to the shared
    base and diverge during training.
    """

    def __init__(self, config: QFormerConfig, model_config: ModelConfig):
        super().__init__()
        self.num_layers = model_config.num_layers
        self.num_kv_heads = model_config.num_kv_heads
        self.head_dim = model_config.head_dim
        kv_dim = model_config.num_kv_heads * model_config.head_dim
        rank = config.lora_rank

        # Shared base projections
        self.shared_k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.shared_v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)

        # Per-layer LoRA deltas: A initialized small random, B initialized zero
        self.lora_k_A = nn.Parameter(torch.randn(model_config.num_layers, config.hidden_size, rank) * 0.01)
        self.lora_k_B = nn.Parameter(torch.zeros(model_config.num_layers, rank, kv_dim))
        self.lora_v_A = nn.Parameter(torch.randn(model_config.num_layers, config.hidden_size, rank) * 0.01)
        self.lora_v_B = nn.Parameter(torch.zeros(model_config.num_layers, rank, kv_dim))

    def forward(self, query_out: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query_out: [num_layers, num_queries, hidden_size]

        Returns:
            List of (K, V) per LLM layer.
            Each shape: [1, num_kv_heads, num_queries, head_dim]
        """
        num_layers, num_q, _ = query_out.shape

        # Shared base: [num_layers, Q, kv_dim]
        shared_k = self.shared_k_proj(query_out)
        shared_v = self.shared_v_proj(query_out)

        # Per-layer LoRA deltas via batched matmul: [num_layers, Q, kv_dim]
        delta_k = torch.bmm(torch.bmm(query_out, self.lora_k_A), self.lora_k_B)
        delta_v = torch.bmm(torch.bmm(query_out, self.lora_v_A), self.lora_v_B)

        k = shared_k + delta_k
        v = shared_v + delta_v

        # Reshape to [num_layers, num_kv_heads, Q, head_dim]
        k = k.view(num_layers, num_q, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(num_layers, num_q, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Return as list of per-layer (K, V) tuples with batch dim restored
        return [(k[i : i + 1], v[i : i + 1]) for i in range(num_layers)]


class QFormerKVCompressor(nn.Module):
    """Q-Former that compresses per-document hidden states into KV caches.

    Architecture: one shared trunk (SA + CA + FFN) processes all LLM layers
    in a single batched call. Each layer is independent — conditioned only by
    a learned layer embedding and that layer's hidden states. Cross-attention
    is asymmetric: K/V are projected directly from 4096-dim LLM hidden states.
    Output KV heads use shared base projections + per-layer LoRA deltas.

    Query embeddings: A single learned content vector broadcast to
    doc_len // compression_ratio queries at runtime. Sinusoidal positional
    embeddings differentiate queries by their relative position in [0, 1].
    """

    def __init__(self, config: QFormerConfig, model_config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.num_layers = model_config.num_layers

        # Single learned query embedding broadcast to num_queries at runtime
        self.query_embedding = nn.Parameter(
            torch.randn(1, 1, config.hidden_size) * 0.02
        )

        # Sinusoidal positional embedding frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, config.hidden_size, 2).float() / config.hidden_size))
        self.register_buffer("pos_inv_freq", inv_freq)

        # MLP to produce position-dependent query representations from
        # (learned_embedding + sinusoidal_pos)
        self.query_init_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=False),
        )

        # Learned per-layer embeddings for conditioning the shared trunk
        self.layer_embeddings = nn.Parameter(
            torch.randn(model_config.num_layers, 1, config.hidden_size) * 0.02
        )

        # Shared trunk (single batched call for all layers)
        self.trunk = QFormerTrunk(config, model_config)

        # Shared base + per-layer LoRA output KV heads
        self.output_heads = OutputKVHeads(config, model_config)

        self.gradient_checkpointing = config.gradient_checkpointing

    def _get_query_pos_embeddings(self, num_queries: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal positional embeddings for query tokens.

        Positions are linearly spaced in [0, 1] so the encoding is
        independent of the actual document length.

        Returns: [1, num_queries, hidden_size]
        """
        positions = torch.linspace(0, 1, num_queries, device=device).unsqueeze(1)  # [Q, 1]
        angles = positions * self.pos_inv_freq.unsqueeze(0) * 2 * math.pi  # [Q, D/2]
        pos_emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [Q, D]
        return pos_emb.unsqueeze(0)  # [1, Q, D]

    def forward(
        self,
        doc_hidden_states: list[torch.Tensor],
        compression_ratio: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compress a single document's hidden states into KV caches.

        Each LLM layer is compressed independently via a shared trunk
        conditioned on a per-layer embedding. All layers are batched
        into a single trunk forward pass.

        Args:
            doc_hidden_states: List of hidden states per LLM layer.
                Each shape: [1, doc_len, llm_hidden_size (4096)]
            compression_ratio: How much to compress (e.g., 4 means doc_len/4 queries).

        Returns:
            compressed_kv: List of (key, value) per LLM layer.
                Each shape: [1, num_kv_heads, num_queries, head_dim]
        """
        doc_len = doc_hidden_states[0].shape[1]
        num_queries = max(1, doc_len // compression_ratio)
        num_queries = min(num_queries, self.config.max_query_tokens)

        # Base queries: [1, Q, H] — learned content + sinusoidal position → MLP
        query_pos = self._get_query_pos_embeddings(num_queries, self.query_embedding.device)
        base_queries = self.query_init_mlp(self.query_embedding + query_pos)

        # Condition with per-layer embeddings: [num_layers, Q, H]
        conditioned_queries = base_queries + self.layer_embeddings

        # Stack all layers' hidden states: [num_layers, doc_len, 4096]
        all_hs = torch.cat(doc_hidden_states, dim=0)

        # Single batched trunk call (layers act as batch dim)
        if self.gradient_checkpointing and self.training:
            query_out = checkpoint(
                self.trunk, conditioned_queries, all_hs,
                use_reentrant=False,
            )
        else:
            query_out = self.trunk(conditioned_queries, all_hs)

        # Per-layer output KV projections (shared base + LoRA)
        return self.output_heads(query_out)
