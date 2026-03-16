#!/bin/bash
#SBATCH --job-name=leefrag
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --account=PAS2836
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err

set -euo pipefail

# CLI arguments
# Usage: sbatch slurms/h100.slurm [branch] [train.py args...]
# Example: sbatch slurms/h100.slurm main --num_epochs 2 --layer_adapter_rank 8
BRANCH="${1:-main}"
shift || true
EXTRA_ARGS="$@"

# Clone to local node storage
CODE_LOCAL="$TMPDIR/LeeFRAG-$SLURM_JOB_ID"

echo "================================================"
echo "Cloning branch '$BRANCH' to local storage..."
echo "Job ID: $SLURM_JOB_ID"
echo "================================================"

REMOTE_URL=$(git -C "$HOME/personal/LeeFRAG" remote get-url origin)

git clone --branch "$BRANCH" --single-branch --depth 1 \
    "$REMOTE_URL" "$CODE_LOCAL"

echo "Clone complete. Branch: $(cd "$CODE_LOCAL" && git branch --show-current)"
echo "Commit: $(cd "$CODE_LOCAL" && git rev-parse --short HEAD)"

# Activate python env
source "$HOME/personal/LeeFRAG/.h100/bin/activate"

cd "$CODE_LOCAL"

echo "================================================"
echo "Starting training..."
echo "Args: $EXTRA_ARGS"
echo "================================================"

python scripts/train.py --gradient_checkpoint_llm $EXTRA_ARGS

echo "================================================"
echo "Training complete. Cleaning up..."
echo "================================================"
rm -rf "$CODE_LOCAL"
