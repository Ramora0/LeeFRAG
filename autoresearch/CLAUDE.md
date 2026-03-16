# Autonomous NPT Research

You are an autonomous researcher running experiments on the NPT (Next-token Prediction) pretraining pipeline for KV cache compression. Your goal is to improve training quality (lower eval loss) by modifying the NPT pipeline and Q-Former architecture.

## File Permissions

### EDITABLE — NPT pipeline files you may modify

These are the only files you are allowed to change:

| File | Purpose |
|------|---------|
| `leefrag/training/npt_trainer.py` | NPT training loop (Stage A/B, loss, LR schedule, eval) |
| `leefrag/data/npt_collator.py` | NPT data collation (continuation building, label construction) |
| `scripts/train_npt_timed.py` | Timed NPT entry point (5-min budget, auto-detects model arch) |
| `scripts/train_npt.py` | Epoch-based NPT entry point (for reference) |
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

## Environment

Always run scripts with the shared venv and HuggingFace cache:

```bash
HF_HOME=/fs/scratch/PAS2836/lees_stuff/hf_cache ../.venv/bin/python scripts/train_npt_timed.py ...
```

## Experiment Strategy

- **Architecture changes** (Q-Former structure, attention, projections): Use `meta-llama/Llama-3.2-1B` (default). Fast iteration, ~5 min per experiment.
- **Hyperparameter tuning** (LR, schedules, loss weights, compression ratios): Use `--model_name ldsjmdy/Tulu3-Block-FT` (LLaMA 3.1 8B). Slower but results transfer directly to production.

## NPT Training Overview

The NPT trainer compresses document KV caches via a Q-Former so a frozen LLM can predict continuation text from the compressed prefix. Unlike RAG fine-tuning which only supervises answer tokens, NPT supervises ALL continuation tokens.

**Default frozen LLM**: `meta-llama/Llama-3.2-1B` (16 layers, 2048 hidden, 8 KV heads, head_dim 64). Architecture is auto-detected from the model.

### Two-stage forward pass

- **Stage A** (no grad): Frozen LLM forward on `[preamble + docs | continuation]` with block-diagonal causal mask. Extracts per-document hidden states and teacher logits.
- **Stage B** (grad): Q-Former compresses hidden states into KV prefix. Frozen LLM forward with compressed prefix. Loss = CE on all continuation tokens + optional KL divergence vs teacher.

### Run command

```bash
HF_HOME=/fs/scratch/PAS2836/lees_stuff/hf_cache ../.venv/bin/python scripts/train_npt_timed.py --no_wandb > run.log 2>&1
```

Training runs for a **fixed 5-minute time budget**. Evaluation runs after training completes and is NOT counted against the budget. Extract results:

```bash
grep "^eval_ce_loss:\|^peak_vram_mb:\|^total_steps:" run.log
```

### Key args

- `--model_name MODEL` — HuggingFace model (default: `meta-llama/Llama-3.2-1B`)
- `--time_budget SECONDS` — wall-clock training budget (default: 300)
- `--compression_schedule 2 4 8 16` — ratio progression (each phase gets equal time)
- `--cross_attn_mode global|chunked` — Q-Former cross-attention strategy
- `--scale N` — multiply Q-Former attn_dim/ffn_dim by N
- `--ce_only` — disable KL loss
- `--kl_weight`, `--kl_top_k` — KL loss tuning
- `--gradient_checkpoint_llm` — save GPU memory in Stage B
- `--offload_stage_a_to_cpu` — move Stage A outputs to CPU between stages
- `--eval_samples N` — cap eval set size (default: 200)

### Output format

The script prints a machine-readable summary after eval:

```
---
eval_ce_loss:       2.345678
eval_kl_loss:       0.123456
eval_total_loss:    2.469134
eval_perplexity:    10.43
training_seconds:   300.1
total_seconds:      325.9
peak_vram_mb:       4500.2
total_steps:        1234
model:              meta-llama/Llama-3.2-1B
qformer_params_M:   1.2
```

### Metric

Primary metric is `eval_ce_loss` (lower is better). This is CE loss on all continuation tokens at the final compression ratio.

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
commit	eval_ce_loss	memory_gb	status	description
a1b2c3d	2.345678	4.4	keep	baseline
b2c3d4e	2.310000	4.5	keep	increase gradient accumulation to 16
c3d4e5f	2.400000	4.4	discard	switch to chunked cross-attn
```

Columns: short commit hash, eval_ce_loss, peak memory in GB (peak_vram_mb / 1024), `keep`/`discard`/`crash`, description.
