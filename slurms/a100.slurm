#!/bin/bash
#SBATCH --job-name=leefrag
#SBATCH --partition=nextgen
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --account=PAS2836
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err

set -euo pipefail

# CLI arguments
# Usage: sbatch slurms/a100.slurm <script> <branch> [extra args...]
# Example: sbatch slurms/a100.slurm scripts/train.py main --num_epochs 2
# Example: sbatch slurms/a100.slurm scripts/train_reconstruction.py reconstruction --bypass
SCRIPT="${1:?Usage: sbatch slurms/a100.slurm <script> <branch> [extra args...]}"
shift
BRANCH="${1:-main}"
shift || true
EXTRA_ARGS="$@"

# Clone to local node storage
CODE_LOCAL="$TMPDIR/LeeFRAG-$SLURM_JOB_ID"

echo "================================================"
echo "Cloning branch '$BRANCH' to local storage..."
echo "Job ID: $SLURM_JOB_ID"
echo "Script: $SCRIPT"
echo "================================================"

REMOTE_URL=$(git -C "$HOME/personal/LeeFRAG" remote get-url origin)

git clone --branch "$BRANCH" --single-branch --depth 1 \
    "$REMOTE_URL" "$CODE_LOCAL"

echo "Clone complete. Branch: $(cd "$CODE_LOCAL" && git branch --show-current)"
echo "Commit: $(cd "$CODE_LOCAL" && git rev-parse --short HEAD)"

# Activate python env
module purge
module load cuda/12.4.1

source "$HOME/personal/LeeFRAG/.a100/bin/activate"

cd "$CODE_LOCAL"

echo "================================================"
echo "Starting training..."
echo "Args: $EXTRA_ARGS"
echo "================================================"

python "$SCRIPT" --gradient_checkpoint_llm $EXTRA_ARGS

echo "================================================"
echo "Training complete. Cleaning up..."
echo "================================================"
rm -rf "$CODE_LOCAL"
