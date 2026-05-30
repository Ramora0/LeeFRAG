"""leefrag-v2: per-layer differentiable KV selection for RAG compression.

The base LLM (Tulu3-Block-FT, LLaMA 3.1 8B) encodes documents as isolated
causal chunks. A per-layer, query-agnostic selector gates which chunk KV tokens
the Q+A tokens may read (different tokens per layer, same expected fraction).
The model is co-adapted (DoRA + RMSNorm tuning + selector) to answer from the
pruned cache. See the package README / plan for the full design.
"""

__version__ = "0.1.0"
