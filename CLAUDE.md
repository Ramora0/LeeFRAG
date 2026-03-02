# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeeFRAG is a KV cache compression training pipeline for RAG (Retrieval-Augmented Generation). It trains a Q-Former model to compress document KV caches so a frozen LLM can use compressed context instead of full documents, reducing memory/compute at inference time.

**Base model**: `ldsjmdy/Tulu3-Block-FT` (LLaMA 3.1 8B with Tulu 3 chat template, frozen during training)
**Dataset**: `glaiveai/RAG-v1` (loaded via HuggingFace `datasets`)

## Commands

```bash
# Install dependencies
pip install -e .

# Train Q-Former compressor
python train.py --use_wandb --gradient_checkpoint_llm

# Train with custom settings
python train.py --learning_rate 1e-4 --num_epochs 4 --batch_size 1 --kl_weight 1.0 --cross_attn_mode global

# Resume from checkpoint
python train.py --resume_from outputs/checkpoint-500/checkpoint.pt

# Run baseline evaluation (full-context, no compression)
python test.py
```

## Architecture

### Two-Stage Training (trainer.py)

The core training loop in `TwoStageTrainer` splits each step into two stages:

- **Stage A** (no grad): Runs the frozen LLM on `[docs | Q+A]` with a block-diagonal causal attention mask. Documents are isolated from each other but Q+A tokens attend to all documents. Extracts per-document hidden states and teacher logits.

- **Stage B** (grad): Q-Former compresses the extracted hidden states into a compact KV cache prefix. The frozen LLM runs again using this compressed prefix + Q+A tokens. Loss = CE on answer tokens + KL divergence (student vs teacher logits).

Only Q-Former parameters are trained. Gradients flow through the compressed KV cache back into the Q-Former.

### Q-Former (qformer.py)

`QFormerKVCompressor` processes all 32 LLM layers through a shared parameter block with four sub-layers:

1. **Cross-attention** (queries ← hidden states): Each query gathers from document hidden states. Per-layer learned embeddings condition the Q projection. ALiBi position bias. (~4.2M params)
2. **Within-layer self-attention** (query ↔ query): Queries within each layer coordinate what each captures (e.g., entities vs reasoning). ALiBi position bias. (~4.2M params)
3. **Cross-layer self-attention** (layer ↔ layer): Transposes `[32, Q, 4096]` → `[Q, 32, 4096]` so each query position attends across all 32 layers. No positional bias (layer ordering is semantic). (~4.2M params)
4. **SwiGLU FFN** (4096→384→4096): Smaller nonlinear transform since attention layers carry more load. (~4.7M params)
5. **Frozen KV projections**: LLM's own k_proj/v_proj produce final KV cache entries.

All three attention output projections are zero-initialized → at init the block is identity → frozen KV proj reproduces the LLM's own cache.

Query count is dynamic: `doc_len // compression_ratio` (max 512). ALiBi slopes provide relative position bias in cross-attention and within-layer self-attention.

Cross-attention modes: `global` (mean-pooled queries attend all) or `chunked` (one learned query per chunk).

The two self-attention modules can be disabled via `--no_within_layer_self_attn` and `--no_cross_layer_pooling`. FFN dim is configurable via `--ffn_dim`.

### Compression Schedule (scheduler.py)

Training progresses through increasing compression ratios: `[2, 4, 8, 16]`, each getting equal training steps.

### Attention Masks (block_attention.py)

Three mask types:
- `build_block_causal_mask`: Block-diagonal causal for isolated documents
- `build_block_causal_mask_with_qa`: Stage A mask — docs isolated, Q+A attends to all docs
- `build_prefix_causal_mask`: Stage B mask — sequence attends to KV prefix + causal self

### Data Pipeline

- `dataset.py`: `RAGDataset` parses `Document:N` formatted text, tokenizes docs individually with per-doc (1024) and total (4096) token limits
- `collator.py`: `RAGCollator` builds Stage B input using Tulu 3 chat template. Labels mask everything except answer tokens (-100)

### KV Cache Utilities (kv_cache_utils.py)

- `extract_doc_hidden_states`: Slices per-document hidden states from concatenated forward pass output (skips embedding layer, uses layer outputs 1..N)
- `concat_compressed_caches`: Concatenates per-document compressed KV pairs into a single `DynamicCache` along the sequence dimension

## Key Design Decisions

- Batch size is always 1 (multi-document samples with variable doc counts)
- Mixed precision (fp16) with `torch.amp.GradScaler`
- `--offload_stage_a_to_cpu`: Moves Stage A outputs to CPU between stages to save GPU memory
- `--gradient_checkpoint_llm`: Enables gradient checkpointing on the frozen LLM for Stage B memory savings
- Q-Former uses gradient checkpointing by default (`QFormerConfig.gradient_checkpointing = True`)
- All attention masks use additive format (0.0 = attend, -inf = masked)
