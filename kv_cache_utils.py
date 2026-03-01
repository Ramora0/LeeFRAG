import torch
from transformers.cache_utils import DynamicCache


def extract_doc_hidden_states(
    all_hidden_states: tuple[torch.Tensor, ...],
    doc_lengths: list[int],
    num_layers: int,
) -> list[list[torch.Tensor]]:
    """Extract per-document hidden states from a concatenated forward pass.

    Args:
        all_hidden_states: Tuple of hidden states per layer from model output.
            Each shape: [batch, total_seq_len, hidden_size].
            Index 0 is embedding output; indices 1..num_layers are layer outputs.
        doc_lengths: Token count for each document.
        num_layers: Number of LLM layers (we use indices 1..num_layers).

    Returns:
        List of per-document hidden states. Each document is a list of
        tensors per layer, shape [batch, doc_len, hidden_size].
    """
    per_doc_hidden = []
    offset = 0

    for doc_len in doc_lengths:
        doc_hs = []
        # Skip index 0 (embedding output), use layer outputs 1..num_layers
        for layer_idx in range(1, num_layers + 1):
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
