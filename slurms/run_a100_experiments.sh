#!/bin/bash
# QA training on latest reconstruction qformer
sbatch --time=2:00:00 slurms/a100.slurm scripts/train.py reconstruction \
    --init_qformer_from "$HOME/personal/LeeFRAG/outputs_reconstruction/checkpoint-1500/checkpoint.pt"

# Reconstruction with identity (CE floor baseline)
sbatch --time=2:00:00 slurms/a100.slurm scripts/train_reconstruction.py reconstruction \
    --bypass
