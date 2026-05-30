# leefrag-v2 — Per-Layer Differentiable KV Selection

Keep a fraction of the **real** KV (no synthesis): the frozen base LLM
(`Tulu3-Block-FT`, LLaMA 3.1 8B) encodes documents as isolated causal chunks; a
per-layer, query-agnostic **selector** gates which chunk KV tokens the Q+A
tokens may read (different tokens per layer, target fraction π annealed
2×→4×→8×); the LLM is co-adapted (**DoRA + RMSNorm tuning + selector**) to answer
from the pruned cache. Single forward, end-to-end, A100 40GB.

## How it works

Each LLaMA layer's attention is **monkeypatched** (`model/patch.py`) into two paths:

- **Path 1 — chunk self-attention:** block-causal-isolated, fused **FlexAttention**
  (`model/flex_masks.py`). Independent of Q+A and the gate.
- **Path 2 — Q+A attention:** softmax over the **full** key set `[chunks | Q+A]`,
  then the per-layer keep-gate multiplies the **post-softmax** weights on the
  chunk columns (no renormalization — "zeroed out"). This is what lets every
  chunk key receive gradient even when gated to ~0.

The gate is a **Gumbel-sigmoid straight-through** sample (`model/selector.py`);
noise is pre-sampled into the forward context so gradient-checkpoint recompute
replays it. The **budget loss** (`training/budget.py`) is a binomial NLL of the
sampled keep-count K against the prior rate π (`-log Binom(K; D, π)`, minimized at
K=Dπ); it re-samples the selector on captured, detached chunk hidden states with
the same stored noise, so it stays correct under checkpointing and trains only
the selector. Positions are **contiguous** (HF default). The KL anchor uses an
**offline precomputed teacher** (`scripts/precompute_teacher.py`).

## Layout

```
leefrag_v2/
  config.py                 V2ModelConfig / SelectorConfig / V2TrainingConfig
  loader.py                 load base + DoRA + unfreeze norms + selector + patch
  data/adapter.py           reuse v1 dataset/collator; build_blocks(); indexed dataset
  model/selector.py         LayerSelector + gumbel_sigmoid_ste + topk_keep_gate
  model/flex_masks.py       block-causal-isolated mask_mod / BlockMask / dense fallback
  model/gated_attention.py  patched attention + reference_dense_attention (parity)
  model/patch.py            ForwardContext + install/uninstall + new_context
  model/peft_setup.py       DoRA config (heavy q/o/FFN, light k/v) + norm/optim groups
  training/{budget,losses,trainer}.py
scripts/  precompute_teacher, milestone1_parity, milestone2_oracle, milestone3_learned, eval
tests/    test_gate_grad, test_integration_cpu (CPU);  test_flex_mask, test_parity (GPU)
slurms/   a100.slurm (generic runner) + run_v2.sh (example invocations)
```

## Setup (GPU server)

```bash
uv pip install -e .            # repo root (leefrag, the reused data pipeline)
uv pip install -e leefrag-v2   # leefrag_v2 + peft>=0.13 (DoRA)
python -c "import peft; print(peft.__version__)"   # verify DoRA support
```

## Run order (de-risk milestones)

```bash
# 1. Correctness — parity (factorized == dense), flex mask, gate gradient.
python leefrag-v2/scripts/milestone1_parity.py

# 2. Offline teacher logits (before KL training).
python leefrag-v2/scripts/precompute_teacher.py --dataset rag_v1 --split train

# 3. Oracle ceiling — does the model tolerate hard eviction at each π?
python leefrag-v2/scripts/milestone2_oracle.py --keep_rate 0.25 --epochs 1

# 4. Learned selector + budget loss (+ offline KL). Main entry.
python leefrag-v2/scripts/milestone3_learned.py --epochs 4 --use_kl_teacher

# Eval: CE/ppl at each π vs the full-context baseline.
python leefrag-v2/scripts/eval.py --checkpoint outputs_v2/learned/checkpoint-XXXX/checkpoint.pt
```

On SLURM: see `slurms/run_v2.sh` (push your branch, pass its name).

## Validation status

- **Passing on CPU:** post-softmax STE gate gradient (reaches every chunk key),
  no-renorm semantics, factorized-vs-dense parity, full forward+backward through a
  tiny patched `LlamaForCausalLM`.
- **Pending on GPU (A100):** `test_flex_mask` / `test_parity` (FlexAttention needs
  CUDA), then milestones 2–3 at scale. Memory budget ~25GB of 40GB; 4-bit QDoRA is
  the fallback if profiling shows pressure.
