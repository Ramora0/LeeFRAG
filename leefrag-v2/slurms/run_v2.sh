#!/bin/bash
# Example sbatch invocations for the leefrag-v2 milestones (A100 40GB).
# Edit BRANCH to the branch you pushed v2 on. Run lines individually, in order.
BRANCH="${1:-reconstruction}"
SLURM="leefrag-v2/slurms/a100.slurm"

# Milestone 1 — correctness (parity + flex + gate gradient). Fast.
sbatch --time=0:30:00 "$SLURM" leefrag-v2/scripts/milestone1_parity.py "$BRANCH"

# Offline teacher logits (one-time, before KL training).
sbatch --time=4:00:00 "$SLURM" leefrag-v2/scripts/precompute_teacher.py "$BRANCH" \
    --dataset rag_v1 --split train --top_k 128 --out outputs_v2/teacher

# Milestone 2 — oracle keep-set ceiling at each compression.
for kr in 0.5 0.25 0.125; do
  sbatch --time=4:00:00 "$SLURM" leefrag-v2/scripts/milestone2_oracle.py "$BRANCH" \
      --keep_rate "$kr" --epochs 1 --output_dir "outputs_v2/oracle_${kr}"
done

# Milestone 3 — learned selector + budget loss (+ offline KL).
sbatch --time=8:00:00 "$SLURM" leefrag-v2/scripts/milestone3_learned.py "$BRANCH" \
    --epochs 4 --use_kl_teacher --teacher_dir outputs_v2/teacher
