import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig, QFormerConfig


class QFormerKVCompressor(nn.Module):
    """Attention-routing compressor: learns to pool hidden states, applies frozen LLM KV projections.

    Architecture:
        1. Mean-pool hidden states at 4096-dim → initial queries
        2. Self-attention: queries coordinate to specialize before routing
        3. Per-layer learned embeddings condition the shared routing
        4. V-free cross-attention with relative position bias: small Wq/Wk compute
           attention routing weights, raw 4096-dim hidden states are the values (no Wv)
        5. Gated residual: output = gate * cross_attn + (1-gate) * queries
        6. SwiGLU FFN with gated residual to break the convex hull of input hidden states
        7. Frozen copies of the LLM's own k_proj/v_proj produce final KV cache entries

    At init with compression_ratio=1:
        - Mean-pool is identity (window size 1)
        - Self-attention gate ≈ 0 → no-op
        - Cross-attention gated residual with gate ≈ 0 → output ≈ queries
        - FFN residual gate ≈ 0 → FFN is a no-op
        - Frozen KV projections reproduce the LLM's own KV cache
        → output ≈ LLM's real KV cache
    """

    def __init__(self, config: QFormerConfig, model_config: ModelConfig, llm=None):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.num_layers = model_config.num_layers
        self.max_query_tokens = config.max_query_tokens

        hidden = model_config.hidden_size  # 4096
        attn_dim = config.attn_dim
        num_kv_heads = model_config.num_kv_heads
        head_dim = model_config.head_dim
        kv_dim = num_kv_heads * head_dim  # 1024

        # Per-layer embeddings for conditioning (in attn_dim since they only affect routing)
        self.layer_embeddings = nn.Parameter(
            torch.randn(model_config.num_layers, 1, attn_dim) * 0.02
        )

        # Cross-attention routing: project to small attn_dim for computing weights
        # Initialized identically so dot products favor nearby (similar) positions
        self.cross_q = nn.Linear(hidden, attn_dim, bias=False)
        self.cross_k = nn.Linear(hidden, attn_dim, bias=False)
        with torch.no_grad():
            self.cross_k.weight.copy_(self.cross_q.weight)

        # Relative position bias: learnable slope (ALiBi-style)
        # Initialized to favor nearby positions; learned during training
        self.pos_bias_slope = nn.Parameter(torch.tensor(2.0))

        # Gated residual: sigmoid(-3) ≈ 0.047, so output ≈ queries at init
        self.residual_gate = nn.Parameter(torch.tensor(-3.0))

        # Shared SwiGLU FFN: breaks the convex hull of input hidden states
        ffn_dim = config.ffn_dim
        self.ffn_gate = nn.Linear(hidden, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(hidden, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, hidden, bias=False)
        # Zero-init output so FFN is a no-op at start
        nn.init.zeros_(self.ffn_down.weight)
        # FFN gated residual (separate from cross-attn gate)
        self.ffn_residual_gate = nn.Parameter(torch.tensor(-3.0))

        # Frozen per-layer KV projections from the LLM
        # Shape: [num_layers, kv_dim, hidden] (transposed for F.linear)
        if llm is not None:
            k_weights = torch.stack([
                llm.model.layers[i].self_attn.k_proj.weight.detach().clone()
                for i in range(model_config.num_layers)
            ])
            v_weights = torch.stack([
                llm.model.layers[i].self_attn.v_proj.weight.detach().clone()
                for i in range(model_config.num_layers)
            ])
            ln_weights = torch.stack([
                llm.model.layers[i].input_layernorm.weight.detach().clone()
                for i in range(model_config.num_layers)
            ])
        else:
            # Placeholder for loading from checkpoint without LLM access
            k_weights = torch.zeros(model_config.num_layers, kv_dim, hidden)
            v_weights = torch.zeros(model_config.num_layers, kv_dim, hidden)
            ln_weights = torch.ones(model_config.num_layers, hidden)

        self.register_buffer("frozen_k_proj", k_weights)  # [32, 1024, 4096]
        self.register_buffer("frozen_v_proj", v_weights)  # [32, 1024, 4096]
        self.register_buffer("frozen_ln_weight", ln_weights)  # [32, 4096]

        self.cross_attn_mode = config.cross_attn_mode

        # Learned query for chunked cross-attention mode
        if self.cross_attn_mode == "chunked":
            self.learned_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        self.gradient_checkpointing = config.gradient_checkpointing

    def _cross_attend(
        self,
        queries: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_emb: torch.Tensor,
    ) -> torch.Tensor:
        """V-free cross-attention with position bias and gated residual.

        Args:
            queries: [num_layers, num_queries, 4096]
            hidden_states: [num_layers, doc_len, 4096]
            layer_emb: [num_layers, 1, attn_dim]

        Returns:
            compressed: [num_layers, num_queries, 4096]
        """
        batch = queries.shape[0]
        num_q = queries.shape[1]
        kv_len = hidden_states.shape[1]

        # Project to small attn_dim for routing, then add per-layer conditioning
        q = self.cross_q(queries) + layer_emb  # [B, Q, attn_dim]
        k = self.cross_k(hidden_states)  # [B, D, attn_dim]

        # Attention scores: [B, Q, D]
        scale = q.shape[-1] ** -0.5
        attn_logits = torch.bmm(q, k.transpose(-2, -1)) * scale

        # Relative position bias: ALiBi-style distance penalty
        # Positions normalized to [0, 1] so bias is doc-length-invariant
        q_pos = torch.arange(num_q, device=queries.device, dtype=queries.dtype)
        k_pos = torch.arange(kv_len, device=queries.device, dtype=queries.dtype)
        q_pos = (q_pos + 0.5) / num_q
        k_pos = (k_pos + 0.5) / kv_len
        rel_dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()  # [Q, D]
        attn_logits = attn_logits - self.pos_bias_slope * rel_dist

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, Q, D]
        cross_attn_out = torch.bmm(attn_weights, hidden_states)  # [B, Q, 4096]

        # Gated residual: at init gate ≈ 0, output ≈ queries (identity)
        gate = torch.sigmoid(self.residual_gate)
        compressed = gate * cross_attn_out + (1.0 - gate) * queries

        return compressed

    def _cross_attend_chunked(
        self,
        queries: torch.Tensor,
        chunked_hs: torch.Tensor,
        layer_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked cross-attention: one learned query attends to each chunk independently.

        Args:
            queries: [num_layers, num_chunks, 4096] (expanded learned query)
            chunked_hs: [num_layers, num_chunks, chunk_size, 4096]
            layer_emb: [num_layers, 1, attn_dim]

        Returns:
            compressed: [num_layers, num_chunks, 4096]
        """
        num_layers, num_chunks, chunk_size, _ = chunked_hs.shape

        # Flatten layers and chunks into batch dim for efficient batched attention
        q_flat = queries.reshape(num_layers * num_chunks, 1, -1)       # [N, 1, 4096]
        hs_flat = chunked_hs.reshape(num_layers * num_chunks, chunk_size, -1)  # [N, cs, 4096]
        le_flat = layer_emb.expand(-1, num_chunks, -1).reshape(
            num_layers * num_chunks, 1, -1
        )  # [N, 1, attn_dim]

        # Project to attn_dim for routing
        q = self.cross_q(q_flat) + le_flat  # [N, 1, attn_dim]
        k = self.cross_k(hs_flat)           # [N, cs, attn_dim]

        scale = q.shape[-1] ** -0.5
        attn_logits = torch.bmm(q, k.transpose(-2, -1)) * scale  # [N, 1, cs]

        # Relative position bias within each chunk
        k_pos = torch.arange(chunk_size, device=queries.device, dtype=queries.dtype)
        k_pos = (k_pos + 0.5) / chunk_size
        q_pos = torch.tensor([0.5], device=queries.device, dtype=queries.dtype)
        rel_dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()  # [1, cs]
        attn_logits = attn_logits - self.pos_bias_slope * rel_dist

        attn_weights = F.softmax(attn_logits, dim=-1)        # [N, 1, cs]
        cross_attn_out = torch.bmm(attn_weights, hs_flat)    # [N, 1, 4096]

        return cross_attn_out.reshape(num_layers, num_chunks, -1)  # [num_layers, num_chunks, 4096]

    def _ffn(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Shared SwiGLU FFN with gated residual back to original queries.

        Args:
            x: [num_layers, num_queries, 4096] — cross-attention output
            residual: [num_layers, num_queries, 4096] — original mean-pooled queries

        Returns:
            output: [num_layers, num_queries, 4096]
        """
        ffn_out = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        gate = torch.sigmoid(self.ffn_residual_gate)
        return gate * ffn_out + residual

    def _apply_frozen_kv_proj(
        self, compressed: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Apply frozen per-layer RMSNorm + KV projections.

        The LLM computes KV as: k_proj(RMSNorm(hidden_states)). The hidden
        states from extract_doc_hidden_states are pre-norm (residual stream),
        so we must apply the frozen input_layernorm before the KV projections.

        Args:
            compressed: [num_layers, num_queries, 4096]

        Returns:
            List of (K, V) per LLM layer.
            Each shape: [1, num_kv_heads, num_queries, head_dim]
        """
        num_layers, num_q, _ = compressed.shape
        num_kv_heads = self.model_config.num_kv_heads
        head_dim = self.model_config.head_dim

        # Apply frozen RMSNorm (input_layernorm) before KV projections
        variance = compressed.pow(2).mean(-1, keepdim=True)
        compressed = compressed * torch.rsqrt(variance + 1e-5)
        compressed = compressed * self.frozen_ln_weight.unsqueeze(1)  # [32, 1, 4096]

        # Batched matmul: [num_layers, Q, 4096] @ [num_layers, 4096, kv_dim] → [num_layers, Q, kv_dim]
        k = torch.bmm(compressed, self.frozen_k_proj.transpose(1, 2))
        v = torch.bmm(compressed, self.frozen_v_proj.transpose(1, 2))

        # Reshape to [num_layers, num_kv_heads, Q, head_dim]
        k = k.view(num_layers, num_q, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(num_layers, num_q, num_kv_heads, head_dim).transpose(1, 2)

        return [(k[i : i + 1], v[i : i + 1]) for i in range(num_layers)]

    def _forward_inner(
        self,
        queries: torch.Tensor,
        all_hs: torch.Tensor,
        layer_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Inner forward for gradient checkpointing."""
        x = self._cross_attend(queries, all_hs, layer_emb)
        x = self._ffn(x, x)
        return x

    def _forward_inner_chunked(
        self,
        queries: torch.Tensor,
        chunked_hs: torch.Tensor,
        layer_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Inner forward (chunked mode) for gradient checkpointing."""
        x = self._cross_attend_chunked(queries, chunked_hs, layer_emb)
        x = self._ffn(x, x)
        return x

    def forward(
        self,
        doc_hidden_states: list[torch.Tensor],
        compression_ratio: int,
        bypass: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compress a single document's hidden states into KV caches.

        Args:
            doc_hidden_states: List of hidden states per LLM layer.
                Each shape: [1, doc_len, llm_hidden_size (4096)]
            compression_ratio: How much to compress (e.g., 4 means doc_len/4 queries).
            bypass: If True, skip cross-attention entirely and pass hidden states
                directly through RMSNorm + frozen KV projections. For diagnostics.

        Returns:
            compressed_kv: List of (key, value) per LLM layer.
                Each shape: [1, num_kv_heads, num_queries, head_dim]
        """
        doc_len = doc_hidden_states[0].shape[1]
        num_queries = max(1, doc_len // compression_ratio)
        num_queries = min(num_queries, self.max_query_tokens)

        # Stack all layers' hidden states: [num_layers, doc_len, 4096]
        all_hs = torch.cat(doc_hidden_states, dim=0)

        if bypass:
            # Diagnostic: skip all learnable parameters, just RMSNorm + frozen KV proj
            return self._apply_frozen_kv_proj(all_hs)

        if self.cross_attn_mode == "chunked":
            # Chunked mode: one learned query attends to each chunk_size segment
            chunk_size = compression_ratio
            num_chunks = doc_len // chunk_size
            num_chunks = min(num_chunks, self.max_query_tokens)
            usable_len = num_chunks * chunk_size

            # Reshape into chunks: [num_layers, num_chunks, chunk_size, 4096]
            chunked_hs = all_hs[:, :usable_len].reshape(
                self.num_layers, num_chunks, chunk_size, -1
            )

            # Expand learned query: [1, 1, 4096] → [num_layers, num_chunks, 4096]
            queries = self.learned_query.expand(self.num_layers, num_chunks, -1)

            if self.gradient_checkpointing and self.training:
                compressed = checkpoint(
                    self._forward_inner_chunked, queries, chunked_hs, self.layer_embeddings,
                    use_reentrant=False,
                )
            else:
                compressed = self._cross_attend_chunked(queries, chunked_hs, self.layer_embeddings)
                compressed = self._ffn(compressed, compressed)
        else:
            # Global mode: mean-pooled queries attend to entire sequence
            # Mean-pool hidden states: [num_layers, num_queries, 4096]
            if num_queries == doc_len:
                pooled = all_hs
            else:
                pooled = F.adaptive_avg_pool1d(
                    all_hs.permute(0, 2, 1),
                    num_queries,
                ).permute(0, 2, 1)

            # V-free cross-attention: learned routing, raw hidden state values
            if self.gradient_checkpointing and self.training:
                compressed = checkpoint(
                    self._forward_inner, pooled, all_hs, self.layer_embeddings,
                    use_reentrant=False,
                )
            else:
                compressed = self._cross_attend(pooled, all_hs, self.layer_embeddings)
                compressed = self._ffn(compressed, compressed)

        # Apply frozen LLM KV projections
        return self._apply_frozen_kv_proj(compressed)
