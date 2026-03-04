import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache


def extract_doc_hidden_states(
    all_hidden_states: tuple[torch.Tensor, ...],
    doc_lengths: list[int],
    num_layers: int,
) -> list[list[torch.Tensor]]:
    """Extract per-document hidden states from a concatenated forward pass.

    Returns the INPUT to each LLM layer (not the output), since the KV cache
    at layer i is computed from layer i's input.

    Args:
        all_hidden_states: Tuple of hidden states per layer from model output.
            Each shape: [batch, total_seq_len, hidden_size].
            Index 0 is embedding output (= input to layer 0).
            Index i is output of layer i-1 (= input to layer i).
        doc_lengths: Token count for each document.
        num_layers: Number of LLM layers (we use indices 0..num_layers-1).

    Returns:
        List of per-document hidden states. Each document is a list of
        tensors per layer, shape [batch, doc_len, hidden_size].
    """
    per_doc_hidden = []
    offset = 0

    for doc_len in doc_lengths:
        doc_hs = []
        # Index i = input to LLM layer i (embedding for i=0, output of layer i-1 otherwise)
        for layer_idx in range(num_layers):
            hs = all_hidden_states[layer_idx][:, offset : offset + doc_len, :]
            doc_hs.append(hs)
        per_doc_hidden.append(doc_hs)
        offset += doc_len

    return per_doc_hidden


def build_dynamic_cache(
    kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> DynamicCache:
    """Build a DynamicCache from a list of (key, value) tuples per layer.

    Args:
        kv_pairs: List of (key, value) tensors per layer.
                  Each shape: [batch, num_kv_heads, seq_len, head_dim].

    Returns:
        DynamicCache with the KV pairs loaded.
    """
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_pairs):
        cache.update(k, v, layer_idx)
    return cache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_to_cache(
    cache: DynamicCache,
    num_layers: int,
    rotary_emb: nn.Module,
) -> DynamicCache:
    """Apply RoPE to K values in the compressed cache at positions [0, prefix_len).

    The Q-Former outputs K/V without RoPE. The downstream LLM expects cached K
    to already have RoPE applied (as it would during normal autoregressive
    decoding where K is rotated before being stored). V is never RoPE'd.

    Args:
        cache: DynamicCache with compressed K/V from Q-Former.
        num_layers: Number of LLM layers.
        rotary_emb: The LLM's LlamaRotaryEmbedding module.

    Returns:
        The same cache with RoPE applied to all K values in-place.
    """
    prefix_len = cache.get_seq_length()
    device = cache.layers[0].keys.device
    position_ids = torch.arange(prefix_len, device=device).unsqueeze(0)

    # Get cos/sin from the model's rotary embedding
    cos, sin = rotary_emb(cache.layers[0].keys, position_ids=position_ids)
    # cos, sin: [1, prefix_len, head_dim]
    cos = cos.unsqueeze(1)  # [1, 1, prefix_len, head_dim]
    sin = sin.unsqueeze(1)  # [1, 1, prefix_len, head_dim]

    for layer_idx in range(num_layers):
        k = cache.layers[layer_idx].keys  # [batch, num_kv_heads, prefix_len, head_dim]
        cache.layers[layer_idx].keys = (k * cos) + (_rotate_half(k) * sin)

    return cache


def apply_rope_to_cache_blocked(
    cache: DynamicCache,
    num_layers: int,
    rotary_emb: nn.Module,
    prefix_lengths: list[int],
) -> DynamicCache:
    """Apply RoPE to K values with per-block position IDs.

    Each doc's prefix segment gets positions [0, prefix_len_i) rather than
    continuous positions across all docs. This matches the block-diagonal
    prefix mask where each doc's compressed cache is independent context.

    Args:
        cache: DynamicCache with compressed K/V from Q-Former.
        num_layers: Number of LLM layers.
        rotary_emb: The LLM's LlamaRotaryEmbedding module.
        prefix_lengths: Per-document compressed KV lengths.

    Returns:
        The same cache with RoPE applied to all K values in-place.
    """
    total_prefix = sum(prefix_lengths)
    device = cache.layers[0].keys.device

    # Build per-block position IDs: each block starts at 0
    position_ids = torch.zeros(total_prefix, dtype=torch.long, device=device)
    offset = 0
    for p_len in prefix_lengths:
        position_ids[offset : offset + p_len] = torch.arange(p_len, device=device)
        offset += p_len
    position_ids = position_ids.unsqueeze(0)  # [1, total_prefix]

    # Get cos/sin from the model's rotary embedding
    cos, sin = rotary_emb(cache.layers[0].keys, position_ids=position_ids)
    cos = cos.unsqueeze(1)  # [1, 1, total_prefix, head_dim]
    sin = sin.unsqueeze(1)

    for layer_idx in range(num_layers):
        k = cache.layers[layer_idx].keys
        cache.layers[layer_idx].keys = (k * cos) + (_rotate_half(k) * sin)

    return cache


def build_blocked_position_ids(
    prefix_lengths: list[int],
    input_block_lengths: list[int],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build position IDs for input tokens with per-block offsets.

    Each block's positions start at its prefix_length_i (continuing from the
    compressed prefix). Positions are NOT continuous across blocks — each block
    restarts from its own prefix length.

    Args:
        prefix_lengths: Per-document compressed KV lengths.
        input_block_lengths: Per-document input block lengths.
        device: Target device.

    Returns:
        position_ids: [1, total_input_len] tensor.
    """
    total_input = sum(input_block_lengths)
    position_ids = torch.zeros(total_input, dtype=torch.long, device=device)

    offset = 0
    for p_len, i_len in zip(prefix_lengths, input_block_lengths):
        position_ids[offset : offset + i_len] = torch.arange(
            p_len, p_len + i_len, device=device,
        )
        offset += i_len

    return position_ids.unsqueeze(0)  # [1, total_input_len]


def concat_compressed_caches(
    per_doc_compressed: list[list[tuple[torch.Tensor, torch.Tensor]]],
    num_layers: int,
) -> DynamicCache:
    """Concatenate per-document compressed KV caches into a single DynamicCache.

    Args:
        per_doc_compressed: List of per-document compressed caches.
            Each is a list of (key, value) per layer.
        num_layers: Number of LLM layers.

    Returns:
        DynamicCache with all documents' compressed KV concatenated along seq dim.
    """
    kv_pairs = []
    for layer_idx in range(num_layers):
        keys = [doc_cache[layer_idx][0] for doc_cache in per_doc_compressed]
        values = [doc_cache[layer_idx][1] for doc_cache in per_doc_compressed]
        concat_k = torch.cat(keys, dim=2)  # cat along seq_len
        concat_v = torch.cat(values, dim=2)
        kv_pairs.append((concat_k, concat_v))
    return build_dynamic_cache(kv_pairs)
