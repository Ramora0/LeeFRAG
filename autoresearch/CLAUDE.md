# Autonomous NPT Research

You are an autonomous researcher running experiments on the NPT (Next-token Prediction) pretraining pipeline for KV cache compression. Your goal is to improve training quality (lower eval loss) by modifying the NPT pipeline and Q-Former architecture.

## File Permissions

### EDITABLE — NPT pipeline files you may modify

These are the only files you are allowed to change:

| File | Purpose |
|------|---------|
| `leefrag/training/npt_trainer.py` | NPT training loop (Stage A/B, loss, LR schedule, eval) |
| `leefrag/data/npt_collator.py` | NPT data collation (continuation building, label construction) |
| `scripts/train_npt.py` | NPT entry point (arg parsing, model/data setup, trainer init) |
| `leefrag/model/qformer.py` | Q-Former architecture (layers, cross-attention, projections) |
| `leefrag/model/block_attention.py` | Attention mask builders (block causal, prefix causal) |
| `leefrag/utils/kv_cache_utils.py` | KV cache extraction, concatenation, RoPE application |
| `leefrag/config.py` | ModelConfig, QFormerConfig, TrainingConfig dataclasses |
| `leefrag/training/scheduler.py` | CompressionScheduler (ratio schedule + LR warm-restart) |

### READ-ONLY — Shared data pipeline (do NOT modify)

| File | Purpose |
|------|---------|
| `leefrag/data/dataset.py` | RAGDataset (document parsing, tokenization, train/eval split) |

### OFF-LIMITS — Unrelated files (do NOT read or modify)

Everything else in the repo is unrelated to NPT training. Do not touch:

- `leefrag/training/trainer.py` (RAG fine-tuning trainer)
- `leefrag/training/absorber_trainer.py` (absorber LoRA trainer)
- `leefrag/training/reconstruction_trainer.py` (reconstruction trainer)
- `leefrag/data/collator.py` (RAG collator)
- `leefrag/data/reconstruction_collator.py` (reconstruction collator)
- `leefrag/evaluation/` (all eval scripts)
- `scripts/train.py`, `scripts/absorber_train.py`, `scripts/train_reconstruction.py`
- `slurms/`, `docs/`, `CLAUDE.md` (root), `pyproject.toml`

## NPT Training Overview

The NPT trainer compresses document KV caches via a Q-Former so a frozen LLM (LLaMA 3.1 8B) can predict continuation text from the compressed prefix. Unlike RAG fine-tuning which only supervises answer tokens, NPT supervises ALL continuation tokens.

### Two-stage forward pass

- **Stage A** (no grad): Frozen LLM forward on `[preamble + docs | continuation]` with block-diagonal causal mask. Extracts per-document hidden states and teacher logits.
- **Stage B** (grad): Q-Former compresses hidden states into KV prefix. Frozen LLM forward with compressed prefix. Loss = CE on all continuation tokens + optional KL divergence vs teacher.

### Run command

```bash
python scripts/train_npt.py --gradient_checkpoint_llm --no_wandb
```

### Key args

- `--compression_schedule 2 4 8 16` — ratio progression (each phase gets equal steps)
- `--cross_attn_mode global|chunked` — Q-Former cross-attention strategy
- `--scale N` — multiply Q-Former attn_dim/ffn_dim by N
- `--ce_only` — disable KL loss
- `--kl_weight`, `--kl_top_k` — KL loss tuning
- `--gradient_checkpoint_llm` — save GPU memory in Stage B
- `--offload_stage_a_to_cpu` — move Stage A outputs to CPU between stages
- `--resume_from PATH` — resume from checkpoint

### Metric

Primary metric is eval CE loss (lower is better). Logged every `eval_steps` (default: 4x per epoch).

## Experimentation Rules

1. Only modify the editable files listed above.
2. Each experiment should be a single, focused change. Commit before running.
3. If a run crashes, diagnose from the error. Fix if trivial, skip if fundamental.
4. If eval loss improves, keep the change. If equal or worse, revert.
5. Do not install new packages. Use only what is in `pyproject.toml`.
6. Log results to `autoresearch/results.tsv` (do NOT commit this file).

## Results Tracking

Log each experiment to `autoresearch/results.tsv` (tab-separated):

```
commit	eval_loss	status	description
a1b2c3d	2.345678	keep	baseline
b2c3d4e	2.310000	keep	increase gradient accumulation to 16
c3d4e5f	2.400000	discard	switch to chunked cross-attn
```

Columns: short commit hash, eval loss, `keep`/`discard`/`crash`, description.
