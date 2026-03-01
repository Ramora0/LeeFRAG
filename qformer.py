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
    layers_per_group: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a mask where each query attends only to its local window of inputs.

    The kv sequence is layers_per_group concatenated copies of doc_len tokens.
    Within each layer copy, query i attends to input tokens
    [i * stride, i * stride + window_size) where stride = doc_len / num_queries.

    The mask is applied identically across all layer copies.

    Args:
        num_queries: Number of query tokens.
        kv_len: Total KV length (layers_per_group * doc_len).
        layers_per_group: How many layer copies are concatenated.
        dtype: Mask dtype.
        device: Target device.

    Returns:
        mask: [1, 1, num_queries, kv_len]  (0.0 = attend, -inf = masked)
    """
    doc_len = kv_len // layers_per_group
    stride = doc_len / num_queries
    # Window size: at least stride, rounded up, so windows tile with overlap
    window_size = math.ceil(stride) + 1

    # Build mask for one layer copy [num_queries, doc_len]
    q_idx = torch.arange(num_queries, device=device).unsqueeze(1)  # [Q, 1]
    kv_idx = torch.arange(doc_len, device=device).unsqueeze(0)     # [1, D]

    # Center of each query's window
    centers = q_idx.float() * stride + stride / 2  # [Q, 1]
    half_win = window_size / 2

    # Query i attends to kv positions within its window
    in_window = (kv_idx.float() >= (centers - half_win)) & (kv_idx.float() < (centers + half_win))
    layer_mask = torch.where(in_window, 0.0, float("-inf")).to(dtype=dtype)  # [Q, D]

    # Tile across all layer copies
    mask = layer_mask.repeat(1, layers_per_group)  # [Q, kv_len]
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, kv_len]


class QFormerLayer(nn.Module):
    """Single Q-Former layer responsible for one group of LLM layers.

    1. Self-attention among query tokens
    2. Cross-attention to hidden states from `layers_per_group` LLM layers
       - "global": each query attends to all inputs
       - "windowed": each query attends to its local window
    3. SwiGLU FFN
    4. Output K/V projections for each LLM layer in the group
    """

    def __init__(self, config: QFormerConfig, model_config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layers_per_group = config.layers_per_group
        self.cross_attn_mode = config.cross_attn_mode

        # Layer norms (pre-norm architecture)
        self.self_attn_ln = nn.LayerNorm(self.hidden_size)
        self.cross_attn_ln = nn.LayerNorm(self.hidden_size)
        self.cross_attn_kv_ln = nn.LayerNorm(self.hidden_size)
        self.ffn_ln = nn.LayerNorm(self.hidden_size)

        # Self-attention
        self.self_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.self_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.self_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.self_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.self_attn_dropout = nn.Dropout(config.dropout)

        # Cross-attention (queries from Q-Former, keys/values from LLM hidden states)
        self.cross_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.cross_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.cross_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.cross_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.cross_attn_dropout = nn.Dropout(config.dropout)

        # FFN
        self.ffn = SwiGLU(self.hidden_size, config.ffn_dim, config.dropout)

        # Output projections: produce K and V for each LLM layer in this group
        kv_dim = model_config.num_kv_heads * model_config.head_dim
        self.out_k_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, kv_dim, bias=False)
            for _ in range(self.layers_per_group)
        ])
        self.out_v_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, kv_dim, bias=False)
            for _ in range(self.layers_per_group)
        ])

    def forward(
        self,
        query_tokens: torch.Tensor,
        hidden_states_concat: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            query_tokens: [batch, num_queries, qformer_hidden_size]
            hidden_states_concat: [batch, layers_per_group * doc_len, qformer_hidden_size]

        Returns:
            query_tokens: Updated query tokens [batch, num_queries, qformer_hidden_size]
            kv_pairs: List of (K, V) for each LLM layer in the group.
                Each K, V shape: [batch, num_kv_heads, num_queries, head_dim]
        """
        batch, num_q, _ = query_tokens.shape

        # 1. Self-attention
        residual = query_tokens
        x = self.self_attn_ln(query_tokens)
        q = self._reshape_heads(self.self_q(x), batch, num_q)
        k = self._reshape_heads(self.self_k(x), batch, num_q)
        v = self._reshape_heads(self.self_v(x), batch, num_q)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(batch, num_q, self.hidden_size)
        query_tokens = residual + self.self_attn_dropout(self.self_o(attn_out))

        # 2. Cross-attention to LLM hidden states
        residual = query_tokens
        x = self.cross_attn_ln(query_tokens)
        hs_input = self.cross_attn_kv_ln(hidden_states_concat)
        hs_len = hs_input.shape[1]
        q = self._reshape_heads(self.cross_q(x), batch, num_q)
        k = self._reshape_heads(self.cross_k(hs_input), batch, hs_len)
        v = self._reshape_heads(self.cross_v(hs_input), batch, hs_len)

        # Build cross-attention mask for windowed mode
        cross_attn_mask = None
        if self.cross_attn_mode == "windowed":
            cross_attn_mask = _build_windowed_cross_attn_mask(
                num_q, hs_len, self.layers_per_group,
                dtype=q.dtype, device=q.device,
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

        # 4. Output K/V projections for each LLM layer in the group
        kv_pairs = []
        for i in range(self.layers_per_group):
            out_k = self.out_k_projs[i](query_tokens)  # [batch, num_q, kv_dim]
            out_v = self.out_v_projs[i](query_tokens)
            # Reshape to [batch, num_kv_heads, num_queries, head_dim]
            out_k = out_k.view(batch, num_q, self.model_config.num_kv_heads, self.model_config.head_dim).transpose(1, 2)
            out_v = out_v.view(batch, num_q, self.model_config.num_kv_heads, self.model_config.head_dim).transpose(1, 2)
            kv_pairs.append((out_k, out_v))

        return query_tokens, kv_pairs

    def _reshape_heads(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """Reshape [batch, seq, hidden] -> [batch, heads, seq, head_dim]."""
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


class QFormerKVCompressor(nn.Module):
    """Q-Former that compresses per-document hidden states into KV caches.

    Cross-attends to LLM hidden states (4096-dim), then projects to K/V (1024-dim).

    Query embeddings: A pool of 512 learned embeddings, sliced to
    doc_len // compression_ratio at runtime. Sinusoidal positional embeddings
    are added based on each query's relative position in [0, 1], giving
    the queries a sense of where in the document they correspond to.

    Cross-attention modes:
    - "global": every query attends to all input hidden states
    - "windowed": query i attends only to its local stride of input tokens
    """

    def __init__(self, config: QFormerConfig, model_config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_config = model_config

        # Learned query embeddings (content component)
        self.query_embeddings = nn.Parameter(
            torch.randn(1, config.max_query_tokens, config.hidden_size) * 0.02
        )

        # Sinusoidal positional embedding frequencies (for relative position encoding)
        # Precompute inverse frequencies for sin/cos encoding
        inv_freq = 1.0 / (10000 ** (torch.arange(0, config.hidden_size, 2).float() / config.hidden_size))
        self.register_buffer("pos_inv_freq", inv_freq)

        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(config, model_config) for _ in range(config.num_qformer_layers)
        ])

        # Input projection: LLM hidden_size (4096) → Q-Former hidden_size (1024)
        self.input_proj = nn.Linear(
            model_config.hidden_size, config.hidden_size, bias=False
        )

        self.gradient_checkpointing = config.gradient_checkpointing

    def _get_query_pos_embeddings(self, num_queries: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal positional embeddings for query tokens.

        Positions are linearly spaced in [0, 1] so the encoding is
        independent of the actual document length.

        Returns: [1, num_queries, hidden_size]
        """
        positions = torch.linspace(0, 1, num_queries, device=device).unsqueeze(1)  # [Q, 1]
        # Scale positions to a reasonable range for sin/cos
        angles = positions * self.pos_inv_freq.unsqueeze(0) * 2 * math.pi  # [Q, D/2]
        pos_emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [Q, D]
        return pos_emb.unsqueeze(0)  # [1, Q, D]

    def forward(
        self,
        doc_hidden_states: list[torch.Tensor],
        compression_ratio: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compress a single document's hidden states into KV caches.

        Args:
            doc_hidden_states: List of hidden states per LLM layer.
                Each shape: [batch, doc_len, llm_hidden_size (4096)]
            compression_ratio: How much to compress (e.g., 4 means doc_len/4 queries).

        Returns:
            compressed_kv: List of (key, value) per LLM layer.
                Each shape: [batch, num_kv_heads, num_queries, head_dim]
        """
        batch = doc_hidden_states[0].shape[0]
        doc_len = doc_hidden_states[0].shape[1]
        num_queries = max(1, doc_len // compression_ratio)
        num_queries = min(num_queries, self.config.max_query_tokens)

        # Slice learned query embeddings + add positional encoding
        query_content = self.query_embeddings[:, :num_queries, :]
        query_pos = self._get_query_pos_embeddings(num_queries, query_content.device)
        query_tokens = (query_content + query_pos).expand(batch, -1, -1)

        # Process each group of LLM layers
        all_kv_pairs = []
        for group_idx, qformer_layer in enumerate(self.layers):
            start_layer = group_idx * self.config.layers_per_group
            end_layer = start_layer + self.config.layers_per_group

            # Project and concatenate hidden states from this group's layers
            group_hs_list = []
            for layer_idx in range(start_layer, end_layer):
                hs = doc_hidden_states[layer_idx]  # [batch, doc_len, 4096]
                hs_proj = self.input_proj(hs)  # [batch, doc_len, 1024]
                group_hs_list.append(hs_proj)

            # [batch, layers_per_group * doc_len, qformer_hidden_size]
            hs_concat = torch.cat(group_hs_list, dim=1)

            if self.gradient_checkpointing and self.training:
                query_tokens, kv_pairs = checkpoint(
                    qformer_layer, query_tokens, hs_concat,
                    use_reentrant=False,
                )
            else:
                query_tokens, kv_pairs = qformer_layer(query_tokens, hs_concat)

            all_kv_pairs.extend(kv_pairs)

        return all_kv_pairs
