#!/bin/bash
#SBATCH --job-name=leefrag-v2
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

# Generic v2 runner (no v1 flags).
# Usage: sbatch leefrag-v2/slurms/a100.slurm <script> <branch> [extra args...]
# Example:
#   sbatch leefrag-v2/slurms/a100.slurm leefrag-v2/scripts/milestone1_parity.py reconstruction
#   sbatch leefrag-v2/slurms/a100.slurm leefrag-v2/scripts/milestone3_learned.py reconstruction --epochs 4
SCRIPT="${1:?Usage: sbatch a100.slurm <script> <branch> [extra args...]}"
shift
BRANCH="${1:-reconstruction}"
shift || true
EXTRA_ARGS="$@"

CODE_LOCAL="$TMPDIR/LeeFRAG-$SLURM_JOB_ID"
REMOTE_URL=$(git -C "$HOME/personal/LeeFRAG" remote get-url origin)

echo "Job $SLURM_JOB_ID | branch=$BRANCH | script=$SCRIPT"
git clone --branch "$BRANCH" --single-branch --depth 1 "$REMOTE_URL" "$CODE_LOCAL"
echo "Commit: $(cd "$CODE_LOCAL" && git rev-parse --short HEAD)"

module purge
module load cuda/12.4.1
source "$HOME/personal/LeeFRAG/.a100/bin/activate"

# peft is required for DoRA; install once into the env if missing:
#   uv pip install "peft>=0.13"
python -c "import peft; print('peft', peft.__version__)"

cd "$CODE_LOCAL"
echo "Args: $EXTRA_ARGS"
python "$SCRIPT" $EXTRA_ARGS

rm -rf "$CODE_LOCAL"
