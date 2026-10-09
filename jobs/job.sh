#!/bin/bash
#SBATCH --job-name=cnv-publication
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]}")/..}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
read -r -a networks <<< "${NETWORKS:-PCNET2.2 PCNET2.2_brain STRING_high_conf STRING_high_conf_brain}"
exec .venv/bin/python -u scripts/publication_pipeline.py --networks "${networks[@]}"
