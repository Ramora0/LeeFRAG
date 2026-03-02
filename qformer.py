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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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
    """Shared trunk: self-attention (with RoPE) among queries, cross-attention to hidden states, FFN.

    Self-attention uses RoPE so the model understands query ordering.
    Cross-attention is asymmetric: queries at hidden_size, keys/values
    projected directly from LLM hidden_size (4096).

    Residual projections (self_o, cross_o, ffn.w3) are zero-initialized
    so the trunk starts as an identity function via skip connections.
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

        # RoPE for self-attention
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_inv_freq", inv_freq)

        # Zero-init residual projections so trunk starts as identity
        nn.init.zeros_(self.self_o.weight)
        nn.init.zeros_(self.cross_o.weight)
        nn.init.zeros_(self.ffn.w3.weight)

    def _apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary position embedding. x: [batch, heads, seq, head_dim]."""
        return (x * cos) + (_rotate_half(x) * sin)

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

        # Precompute RoPE cos/sin for self-attention
        positions = torch.arange(num_q, device=query_tokens.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.rope_inv_freq)  # [Q, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [Q, head_dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, Q, head_dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, Q, head_dim]

        # 1. Self-attention with RoPE
        residual = query_tokens
        x = self.self_attn_ln(query_tokens)
        q = self._reshape_heads(self.self_q(x), batch, num_q)
        k = self._reshape_heads(self.self_k(x), batch, num_q)
        v = self._reshape_heads(self.self_v(x), batch, num_q)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
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

    Shared base is identity-initialized so Q-Former starts as passthrough.
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

        # Shared base projections (identity-initialized for passthrough at init)
        self.shared_k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.shared_v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        nn.init.eye_(self.shared_k_proj.weight)
        nn.init.eye_(self.shared_v_proj.weight)

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

    Query initialization: Mean-pool input hidden states with window size
    equal to the compression ratio, then project to Q-Former dim. At ratio 1
    this is identity (no pooling), giving the model a near-passthrough starting
    point. RoPE in self-attention encodes query ordering.
    """

    def __init__(self, config: QFormerConfig, model_config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.num_layers = model_config.num_layers

        # Project mean-pooled hidden states to Q-Former dim
        self.input_proj = nn.Linear(model_config.hidden_size, config.hidden_size, bias=False)

        # Learned per-layer embeddings for conditioning the shared trunk
        self.layer_embeddings = nn.Parameter(
            torch.randn(model_config.num_layers, 1, config.hidden_size) * 0.02
        )

        # Shared trunk (single batched call for all layers)
        self.trunk = QFormerTrunk(config, model_config)

        # Shared base + per-layer LoRA output KV heads
        self.output_heads = OutputKVHeads(config, model_config)

        self.gradient_checkpointing = config.gradient_checkpointing

    def init_from_llm(self, llm: nn.Module):
        """Initialize input_proj and output heads from the LLM's K/V weights.

        Factorizes the LLM's per-layer K/V projections through the Q-Former's
        1024-dim bottleneck so the module starts near-identity: at ratio 1
        the compressed KV cache closely matches the LLM's real KV cache.

        Steps:
          1. Stack all per-layer k_proj and v_proj weights (both 1024x4096).
          2. SVD on their concatenation to find the best shared 1024-dim
             subspace of the 4096-dim hidden states for K/V reconstruction.
          3. Set input_proj to project into this subspace.
          4. Solve shared_k/v_proj via least-squares to reconstruct the
             layer-average K and V projections from the subspace.
          5. Initialize per-layer LoRA A/B to capture each layer's deviation
             from the average.
        """
        device = self.input_proj.weight.device
        dtype = self.input_proj.weight.dtype

        # Collect per-layer K/V projection weights: each [kv_dim, hidden_size]
        all_k_weights = []
        all_v_weights = []
        for layer in llm.model.layers:
            all_k_weights.append(layer.self_attn.k_proj.weight.detach().float())
            all_v_weights.append(layer.self_attn.v_proj.weight.detach().float())

        # [num_layers, kv_dim, llm_hidden_size]
        Wk = torch.stack(all_k_weights)
        Wv = torch.stack(all_v_weights)

        Wk_avg = Wk.mean(dim=0)  # [kv_dim, llm_hidden_size] = [1024, 4096]
        Wv_avg = Wv.mean(dim=0)

        # SVD on stacked [Wk_avg; Wv_avg] (2048 x 4096) to find best shared
        # 1024-dim subspace of the 4096-dim hidden states
        stacked = torch.cat([Wk_avg, Wv_avg], dim=0)  # [2048, 4096]
        U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
        # Vh[:1024] are the top-1024 right singular vectors: [1024, 4096]
        # This is the input projection: 4096 -> 1024
        input_proj_weight = Vh[:self.config.hidden_size]  # [1024, 4096]

        # Solve for shared output projections via least-squares:
        # shared_k_proj @ input_proj ≈ Wk_avg  =>  shared_k_proj ≈ Wk_avg @ pinv(input_proj)
        # Since input_proj is [1024, 4096] with full row rank,
        # pinv = V^T (S^{-1}) U^T in thin SVD, but simpler:
        # pinv(A) = A^T @ (A @ A^T)^{-1} for full row-rank A
        # But since input_proj = Vh[:1024] (orthonormal rows), pinv = Vh[:1024]^T
        input_pinv = input_proj_weight.T  # [4096, 1024] (rows are orthonormal)

        shared_k_weight = Wk_avg @ input_pinv  # [1024, 1024]
        shared_v_weight = Wv_avg @ input_pinv  # [1024, 1024]

        # Apply to modules
        self.input_proj.weight.data.copy_(input_proj_weight.to(dtype).to(device))
        self.output_heads.shared_k_proj.weight.data.copy_(shared_k_weight.to(dtype).to(device))
        self.output_heads.shared_v_proj.weight.data.copy_(shared_v_weight.to(dtype).to(device))

        # Initialize per-layer LoRA to capture each layer's deviation from avg.
        # For layer i: W_k_i ≈ (shared_k + A_i @ B_i) @ input_proj
        # Residual in output space: R_k_i = W_k_i @ input_pinv - shared_k  [1024x1024]
        # Factor R_k_i ≈ A_i @ B_i via truncated SVD at lora_rank
        rank = self.config.lora_rank
        for i in range(self.num_layers):
            Rk = (all_k_weights[i] @ input_pinv) - shared_k_weight  # [1024, 1024]
            Rv = (all_v_weights[i] @ input_pinv) - shared_v_weight

            Uk, Sk, Vhk = torch.linalg.svd(Rk, full_matrices=False)
            Uv, Sv, Vhv = torch.linalg.svd(Rv, full_matrices=False)

            # A: [hidden_size, rank], B: [rank, kv_dim]
            # R ≈ Uk[:,:r] @ diag(Sk[:r]) @ Vhk[:r,:] = (Uk[:,:r] @ sqrt(S)) @ (sqrt(S) @ Vhk[:r,:])
            sqrt_Sk = Sk[:rank].sqrt()
            self.output_heads.lora_k_A.data[i] = (Uk[:, :rank] * sqrt_Sk.unsqueeze(0)).to(dtype).to(device)
            self.output_heads.lora_k_B.data[i] = (Vhk[:rank, :] * sqrt_Sk.unsqueeze(1)).to(dtype).to(device)

            sqrt_Sv = Sv[:rank].sqrt()
            self.output_heads.lora_v_A.data[i] = (Uv[:, :rank] * sqrt_Sv.unsqueeze(0)).to(dtype).to(device)
            self.output_heads.lora_v_B.data[i] = (Vhv[:rank, :] * sqrt_Sv.unsqueeze(1)).to(dtype).to(device)

        # Zero out layer embeddings so they don't perturb the identity at init
        self.layer_embeddings.data.zero_()

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

        # Stack all layers' hidden states: [num_layers, doc_len, 4096]
        all_hs = torch.cat(doc_hidden_states, dim=0)

        # Mean-pool hidden states: [num_layers, num_queries, 4096]
        # adaptive_avg_pool1d operates on last dim, expects [N, C, L]
        pooled = F.adaptive_avg_pool1d(
            all_hs.permute(0, 2, 1),  # [num_layers, 4096, doc_len]
            num_queries,
        ).permute(0, 2, 1)  # [num_layers, num_queries, 4096]

        # Project to Q-Former dim: [num_layers, num_queries, hidden_size]
        queries = self.input_proj(pooled)

        # Condition with per-layer embeddings
        queries = queries + self.layer_embeddings

        # Single batched trunk call (layers act as batch dim)
        if self.gradient_checkpointing and self.training:
            query_out = checkpoint(
                self.trunk, queries, all_hs,
                use_reentrant=False,
            )
        else:
            query_out = self.trunk(queries, all_hs)

        # Per-layer output KV projections (shared base + LoRA)
        return self.output_heads(query_out)
