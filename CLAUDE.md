# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeeFRAG is a KV cache compression training pipeline for RAG (Retrieval-Augmented Generation). It trains a Q-Former model to compress document KV caches so a frozen LLM can use compressed context instead of full documents, reducing memory/compute at inference time.

**Base model**: `glaiveai/Llama-3-8B-RAG-v1` (LLaMA 3 8B, frozen during training)
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

`QFormerKVCompressor` contains 8 Q-Former layers, each responsible for a group of 4 LLM layers (8 × 4 = 32 total). Each layer:
1. Self-attention among learned query tokens
2. Cross-attention to projected LLM hidden states (4096 → 1024 dim)
3. SwiGLU FFN
4. Output K/V projections back to LLM's KV dimensions per layer in the group

Query count is dynamic: `doc_len // compression_ratio` (max 512). Sinusoidal positional embeddings encode relative position in `[0, 1]`.

Cross-attention modes: `global` (all queries attend to all inputs) or `windowed` (local attention windows).

### Compression Schedule (scheduler.py)

Training progresses through increasing compression ratios: `[2, 4, 8, 16]`, each getting equal training steps.

### Attention Masks (block_attention.py)

Three mask types:
- `build_block_causal_mask`: Block-diagonal causal for isolated documents
- `build_block_causal_mask_with_qa`: Stage A mask — docs isolated, Q+A attends to all docs
- `build_prefix_causal_mask`: Stage B mask — sequence attends to KV prefix + causal self

### Data Pipeline

- `dataset.py`: `RAGDataset` parses `Document:N` formatted text, tokenizes docs individually with per-doc (1024) and total (4096) token limits
- `collator.py`: `RAGCollator` builds Stage B input using Llama 3 chat template. Labels mask everything except answer tokens (-100)

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
