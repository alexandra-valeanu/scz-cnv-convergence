#!/bin/bash
#SBATCH --job-name=cnv-full-disorder
#SBATCH --partition=defq
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Submit from the repository root. NetColoc only, seed 42; reuse valid propagation and analysis caches.
set -euo pipefail
exec bash scripts/run_netcoloc_notebook.sh cnv_disorder_overlap
