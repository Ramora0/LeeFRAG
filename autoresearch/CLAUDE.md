# Autonomous NPT Research

You are an autonomous researcher running experiments on the NPT (Next-token Prediction) pretraining pipeline for KV cache compression. Your goal is to improve training quality (lower eval loss) by modifying the NPT pipeline and Q-Former architecture.

## Setup

1. **Read the in-scope files** listed in "File Permissions" below for full context.
2. **Verify environment**: Check which venv exists (`../.a100`, `../.v100`, or `../.h100`).
3. **Initialize results.tsv**: If `autoresearch/results.tsv` does not exist, copy `autoresearch/base.tsv` into `autoresearch/results.tsv`.
4. **Confirm and go**: Confirm setup looks good, then kick off the experimentation.

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

Always run scripts with the GPU-specific venv and HuggingFace cache. The venv directory depends on the machine: `../.a100`, `../.v100`, or `../.h100`. Check which exists before running.

```bash
HF_HOME=/fs/scratch/PAS2836/lees_stuff/hf_cache ../.a100/bin/python scripts/train_npt_timed.py --no_wandb
```

## Experiment Strategy

**Focus on architecture, not hyperparameters.** Your primary job is to explore structural changes to the Q-Former and the training pipeline — layer design, attention mechanisms, projection strategies, cross-attention patterns, positional encodings, etc. Do not spend time tuning learning rates, weight decay, or other hyperparameters unless you have strong reason to believe they are essential to making a specific architectural change work, or you are genuinely uncertain whether a poor result is due to the architecture or a bad hyperparameter fit.

Since we want quick iteration for architecture experimentation, use `meta-llama/Llama-3.2-1B` (default). Fast iteration, ~5 min per experiment.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A tiny eval_ce_loss improvement that adds 20 lines of hacky code? Probably not worth it. A tiny improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful eval_ce_loss gains, but it should not blow up dramatically.

## NPT Training Overview

The NPT trainer compresses document KV caches via a Q-Former so a frozen LLM can predict continuation text from the compressed prefix. Unlike RAG fine-tuning which only supervises answer tokens, NPT supervises ALL continuation tokens.

**Default frozen LLM**: `meta-llama/Llama-3.2-1B` (16 layers, 2048 hidden, 8 KV heads, head_dim 64). Architecture is auto-detected from the model.

### Two-stage forward pass

- **Stage A** (no grad): Frozen LLM forward on `[preamble + docs | continuation]` with block-diagonal causal mask. Extracts per-document hidden states and teacher logits.
- **Stage B** (grad): Q-Former compresses hidden states into KV prefix. Frozen LLM forward with compressed prefix. Loss = CE on all continuation tokens + optional KL divergence vs teacher.

### Key args

- `--model_name MODEL` — HuggingFace model (default: `meta-llama/Llama-3.2-1B`)
- `--time_budget SECONDS` — wall-clock training budget (default: 300)
- `--compression_schedule 2 4 8 16` — ratio progression (each phase gets equal time, default: 16)
- `--cross_attn_mode global|chunked` — Q-Former cross-attention strategy
- `--scale N` — multiply Q-Former attn_dim/ffn_dim by N
- `--ce_only` — disable KL loss
- `--kl_weight`, `--kl_top_k` — KL loss tuning
- `--gradient_checkpoint_llm` — save GPU memory in Stage B
- `--offload_stage_a_to_cpu` — move Stage A outputs to CPU between stages
- `--max_documents N` — cap total documents before train/eval split (default: 500)

### Output format

The script prints a machine-readable summary after eval:

```
---
eval_ce_loss:       2.345678
training_seconds:   300.1
total_seconds:      325.9
peak_vram_mb:       4500.2
total_steps:        1234
model:              meta-llama/Llama-3.2-1B
qformer_params_M:   3.7
```

### Metric

Primary metric is `eval_ce_loss` (lower is better). This is CE loss on all continuation tokens at the final compression ratio.

## The Experiment Loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit you're on.
2. Modify editable files with an experimental idea — a single, focused change.
3. `git commit` (commit before running).
4. Run the experiment:
   ```bash
   HF_HOME=/fs/scratch/PAS2836/lees_stuff/hf_cache ../.a100/bin/python scripts/train_npt_timed.py --no_wandb
   ```
5. Read the results from the output. The script prints a machine-readable summary at the end.
6. If the run crashed, read the error from the output.
   - If it's something dumb (typo, missing import): fix and re-run.
   - If the idea is fundamentally broken: log as crash and move on.
7. Record the results in `autoresearch/results.tsv` (do NOT commit this file).
8. If eval_ce_loss improved (lower), keep the change — advance the branch.
9. If eval_ce_loss is equal or worse, `git reset --hard` back to where you started.

**Timeout**: Each experiment should take ~5 minutes (+ startup/eval overhead). If a run exceeds 10 minutes, kill it and treat as failure (discard and revert).

**Running out of ideas**: If you feel stuck, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. Do not fall back to hyperparameter tuning as a crutch.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. The loop runs until the human interrupts you, period.

## Results Tracking

Log each experiment to `autoresearch/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

```
commit	eval_ce_loss	total_steps	device	status	description
a1b2c3d	2.345678	250	h100	keep	baseline
b2c3d4e	2.310000	240	h100	keep	add residual connection in cross-attention
c3d4e5f	2.400000	260	h100	discard	switch to chunked cross-attn
d4e5f6g	0.000000	0	h100	crash	double model width (OOM)
```

Columns: short commit hash, eval_ce_loss, total_steps, device (h100/a100/v100), `keep`/`discard`/`crash`/`baseline`, short description.
