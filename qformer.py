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
        2. Per-layer learned embeddings condition the shared routing
        3. Pre-norm multi-head cross-attention with bottleneck V projection:
           RMSNorm → Q/K/V project to attn_dim → output projected back to hidden_size
        4. Pre-norm SwiGLU FFN: RMSNorm → SwiGLU → additive residual
        5. Frozen copies of the LLM's own k_proj/v_proj produce final KV cache entries

    At init with compression_ratio=1:
        - Mean-pool is identity (window size 1)
        - Cross-attention output projection zero-initialized → gated residual ≈ identity
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

        # Per-layer embeddings for conditioning Q and K in cross-attention
        self.layer_embeddings_q = nn.Parameter(
            torch.randn(model_config.num_layers, 1, attn_dim) * 0.02
        )
        self.layer_embeddings_k = nn.Parameter(
            torch.randn(model_config.num_layers, 1, attn_dim) * 0.02
        )

        # Multi-head attention dimensions
        self.num_attn_heads = config.num_attn_heads
        self.attn_head_dim = attn_dim // config.num_attn_heads

        # Self-attention among queries (before cross-attention)
        self.self_attn_norm = nn.RMSNorm(hidden)
        self.self_q = nn.Linear(hidden, attn_dim, bias=False)
        self.self_k = nn.Linear(hidden, attn_dim, bias=False)
        self.self_v = nn.Linear(hidden, attn_dim, bias=False)
        self.self_out = nn.Linear(attn_dim, hidden, bias=False)
        nn.init.zeros_(self.self_out.weight)  # no-op at init

        # Cross-attention with bottleneck V
        self.cross_q = nn.Linear(hidden, attn_dim, bias=False)
        self.cross_k = nn.Linear(hidden, attn_dim, bias=False)
        self.cross_v = nn.Linear(hidden, attn_dim, bias=False)
        self.cross_out = nn.Linear(attn_dim, hidden, bias=False)
        # Zero-init output so cross-attention is a no-op at start
        nn.init.zeros_(self.cross_out.weight)

        # Per-head ALiBi slopes for relative position bias
        self.pos_bias_slopes = nn.Parameter(
            torch.linspace(0.5, 4.0, config.num_attn_heads)
        )

        # Pre-norm RMSNorm for each sub-layer
        self.attn_norm = nn.RMSNorm(hidden)
        self.ffn_norm = nn.RMSNorm(hidden)

        # Shared SwiGLU FFN: breaks the convex hull of input hidden states
        ffn_dim = config.ffn_dim
        self.ffn_gate = nn.Linear(hidden, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(hidden, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, hidden, bias=False)
        # Zero-init output so FFN is a no-op at start
        nn.init.zeros_(self.ffn_down.weight)

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

        # Per-layer low-rank adapter (applied before frozen KV proj)
        self.layer_adapter_rank = config.layer_adapter_rank
        if self.layer_adapter_rank > 0:
            r = self.layer_adapter_rank
            # Zero-init down so adapter starts as no-op
            self.layer_adapter_down = nn.Parameter(torch.zeros(model_config.num_layers, hidden, r))
            self.layer_adapter_up = nn.Parameter(torch.randn(model_config.num_layers, r, hidden) * 0.01)

        self.cross_attn_mode = config.cross_attn_mode

        # Learned query for chunked cross-attention mode
        if self.cross_attn_mode == "chunked":
            self.learned_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        self.gradient_checkpointing = config.gradient_checkpointing

    @staticmethod
    def _sinusoidal_pe(seq_len: int, dim: int, device, dtype) -> torch.Tensor:
        """Generate sinusoidal positional encoding. [seq_len, dim]"""
        pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        dim_idx = torch.arange(0, dim, 2, device=device, dtype=dtype)
        freq = 1.0 / (10000.0 ** (dim_idx / dim))
        pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        return pe

    def _self_attend(self, queries: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention among queries with sinusoidal PE.

        Args:
            queries: [num_layers, num_queries, 4096]

        Returns:
            output: [num_layers, num_queries, 4096]
        """
        B, num_q, _ = queries.shape
        H = self.num_attn_heads
        D = self.attn_head_dim

        q_normed = self.self_attn_norm(queries)
        pe = self._sinusoidal_pe(num_q, q_normed.shape[-1], q_normed.device, q_normed.dtype)
        q_normed = q_normed + pe

        q = self.self_q(q_normed).view(B, num_q, H, D).transpose(1, 2)
        k = self.self_k(q_normed).view(B, num_q, H, D).transpose(1, 2)
        v = self.self_v(q_normed).view(B, num_q, H, D).transpose(1, 2)

        # Causal-free self-attention (all queries attend to all queries)
        scale = D ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, num_q, -1)
        return queries + self.self_out(attn_out)

    def _cross_attend(
        self,
        queries: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_emb_q: torch.Tensor,
        layer_emb_k: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-head cross-attention with bottleneck V and ALiBi position bias.

        Args:
            queries: [num_layers, num_queries, 4096]
            hidden_states: [num_layers, doc_len, 4096]
            layer_emb_q: [num_layers, 1, attn_dim]
            layer_emb_k: [num_layers, 1, attn_dim]

        Returns:
            output: [num_layers, num_queries, 4096]
        """
        B = queries.shape[0]
        num_q = queries.shape[1]
        kv_len = hidden_states.shape[1]
        H = self.num_attn_heads
        D = self.attn_head_dim

        # Pre-norm before cross-attention
        q_normed = self.attn_norm(queries)
        hs_normed = self.attn_norm(hidden_states)

        # Project Q/K/V to attn_dim, add layer conditioning to Q and K
        q = self.cross_q(q_normed) + layer_emb_q  # [B, Q, attn_dim]
        k = self.cross_k(hs_normed) + layer_emb_k # [B, kv_len, attn_dim]
        v = self.cross_v(hs_normed)                # [B, kv_len, attn_dim]

        # Reshape to multi-head: [B, H, seq, head_dim]
        q = q.view(B, num_q, H, D).transpose(1, 2)
        k = k.view(B, kv_len, H, D).transpose(1, 2)
        v = v.view(B, kv_len, H, D).transpose(1, 2)

        # Attention scores: [B, H, Q, kv_len]
        scale = D ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Per-head ALiBi position bias
        q_pos = torch.arange(num_q, device=queries.device, dtype=queries.dtype)
        k_pos = torch.arange(kv_len, device=queries.device, dtype=queries.dtype)
        q_pos = (q_pos + 0.5) / num_q
        k_pos = (k_pos + 0.5) / kv_len
        rel_dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()  # [Q, kv_len]
        pos_bias = -self.pos_bias_slopes.view(-1, 1, 1) * rel_dist  # [H, Q, kv_len]
        attn_logits = attn_logits + pos_bias.unsqueeze(0)  # broadcast over B

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, Q, kv_len]
        attn_out = torch.matmul(attn_weights, v)  # [B, H, Q, head_dim]

        # Concatenate heads and project back to hidden_size
        attn_out = attn_out.transpose(1, 2).reshape(B, num_q, -1)  # [B, Q, attn_dim]
        cross_attn_out = self.cross_out(attn_out)  # [B, Q, 4096]

        # Residual: cross_out is zero-init so this starts as identity
        return queries + cross_attn_out

    def _cross_attend_chunked(
        self,
        queries: torch.Tensor,
        chunked_hs: torch.Tensor,
        layer_emb_q: torch.Tensor,
        layer_emb_k: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked multi-head cross-attention: one learned query attends to each chunk.

        Args:
            queries: [num_layers, num_chunks, 4096] (expanded learned query)
            chunked_hs: [num_layers, num_chunks, chunk_size, 4096]
            layer_emb_q: [num_layers, 1, attn_dim]
            layer_emb_k: [num_layers, 1, attn_dim]

        Returns:
            compressed: [num_layers, num_chunks, 4096]
        """
        num_layers, num_chunks, chunk_size, _ = chunked_hs.shape
        H = self.num_attn_heads
        D = self.attn_head_dim

        # Flatten layers and chunks into batch dim
        N = num_layers * num_chunks
        q_flat = queries.reshape(N, 1, -1)       # [N, 1, 4096]
        hs_flat = chunked_hs.reshape(N, chunk_size, -1)  # [N, cs, 4096]
        le_q_flat = layer_emb_q.expand(-1, num_chunks, -1).reshape(N, 1, -1)
        le_k_flat = layer_emb_k.expand(-1, num_chunks, -1).reshape(N, 1, -1)

        # Pre-norm before cross-attention
        q_normed = self.attn_norm(q_flat)
        hs_normed = self.attn_norm(hs_flat)

        # Project Q/K/V
        q = self.cross_q(q_normed) + le_q_flat  # [N, 1, attn_dim]
        k = self.cross_k(hs_normed) + le_k_flat # [N, cs, attn_dim]
        v = self.cross_v(hs_normed)              # [N, cs, attn_dim]

        # Multi-head reshape: [N, H, seq, head_dim]
        q = q.view(N, 1, H, D).transpose(1, 2)
        k = k.view(N, chunk_size, H, D).transpose(1, 2)
        v = v.view(N, chunk_size, H, D).transpose(1, 2)

        # Attention scores: [N, H, 1, cs]
        scale = D ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Per-head ALiBi position bias within each chunk
        k_pos = torch.arange(chunk_size, device=queries.device, dtype=queries.dtype)
        k_pos = (k_pos + 0.5) / chunk_size
        q_pos = torch.tensor([0.5], device=queries.device, dtype=queries.dtype)
        rel_dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()  # [1, cs]
        pos_bias = -self.pos_bias_slopes.view(-1, 1, 1) * rel_dist  # [H, 1, cs]
        attn_logits = attn_logits + pos_bias.unsqueeze(0)

        attn_weights = F.softmax(attn_logits, dim=-1)  # [N, H, 1, cs]
        attn_out = torch.matmul(attn_weights, v)  # [N, H, 1, head_dim]

        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).reshape(N, 1, -1)  # [N, 1, attn_dim]
        cross_attn_out = self.cross_out(attn_out)  # [N, 1, 4096]

        # Residual
        return (q_flat + cross_attn_out).reshape(num_layers, num_chunks, -1)

    def _ffn(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Shared SwiGLU FFN with gated residual back to original queries.

        Args:
            x: [num_layers, num_queries, 4096] — cross-attention output
            residual: [num_layers, num_queries, 4096] — original mean-pooled queries

        Returns:
            output: [num_layers, num_queries, 4096]
        """
        x_normed = self.ffn_norm(x)
        ffn_out = self.ffn_down(F.silu(self.ffn_gate(x_normed)) * self.ffn_up(x_normed))
        return residual + ffn_out

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
        layer_emb_q: torch.Tensor,
        layer_emb_k: torch.Tensor,
    ) -> torch.Tensor:
        """Inner forward for gradient checkpointing."""
        x = self._self_attend(queries)
        x = self._cross_attend(x, all_hs, layer_emb_q, layer_emb_k)
        x = self._ffn(x, x)
        return x

    def _forward_inner_chunked(
        self,
        queries: torch.Tensor,
        chunked_hs: torch.Tensor,
        layer_emb_q: torch.Tensor,
        layer_emb_k: torch.Tensor,
    ) -> torch.Tensor:
        """Inner forward (chunked mode) for gradient checkpointing."""
        x = self._self_attend(queries)
        x = self._cross_attend_chunked(x, chunked_hs, layer_emb_q, layer_emb_k)
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
                    self._forward_inner_chunked, queries, chunked_hs,
                    self.layer_embeddings_q, self.layer_embeddings_k,
                    use_reentrant=False,
                )
            else:
                compressed = self._self_attend(queries)
                compressed = self._cross_attend_chunked(
                    compressed, chunked_hs,
                    self.layer_embeddings_q, self.layer_embeddings_k,
                )
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
                    self._forward_inner, pooled, all_hs,
                    self.layer_embeddings_q, self.layer_embeddings_k,
                    use_reentrant=False,
                )
            else:
                compressed = self._self_attend(pooled)
                compressed = self._cross_attend(
                    compressed, all_hs,
                    self.layer_embeddings_q, self.layer_embeddings_k,
                )
                compressed = self._ffn(compressed, compressed)

        # Per-layer low-rank adapter
        if self.layer_adapter_rank > 0:
            compressed = compressed + torch.bmm(
                torch.bmm(compressed, self.layer_adapter_down),
                self.layer_adapter_up,
            )

        # Apply frozen LLM KV projections
        return self._apply_frozen_kv_proj(compressed)
