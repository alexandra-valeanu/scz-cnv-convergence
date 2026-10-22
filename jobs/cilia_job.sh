#!/bin/bash
#SBATCH --job-name=cnv-cilia
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Extra analysis: CNV/SCHEMA/GWAS x cilia gene sets, figures and results index
# -> results/<network>/5_cilia
#
# Submit from the repository root:
#   sbatch jobs/cilia_job.sh
#   NETWORK=PCNET2.2 CILIA_SETS=wp4803_ciliopathies sbatch jobs/cilia_job.sh
#
# Parameters (all optional):
#   NETWORK     default PCNET2.2_brain
#   CILIA_SETS  scgsv2_first_order and/or wp4803_ciliopathies (default: both);
#               the figures compare both sets, so they are only drawn when both run

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f jobs/setup_venv.sh ]] || { echo "Submit from the repository root (jobs/setup_venv.sh not found)" >&2; exit 1; }
source jobs/setup_venv.sh
NETWORK=${NETWORK:-PCNET2.2_brain}

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
unset CNV_CONFIG CNV_NULL_METHOD CNV_NETWORK
echo "Commit $(git rev-parse --short HEAD), started $(date)"

set -x
if [[ -n "${CILIA_SETS:-}" ]]; then
    .venv/bin/python -u scripts/cilia_colocalization.py --network "$NETWORK" --sets $CILIA_SETS
else
    .venv/bin/python -u scripts/cilia_colocalization.py --network "$NETWORK"
    .venv/bin/python -u scripts/plot_cilia_colocalisation.py --network "$NETWORK"
fi
.venv/bin/python -u scripts/index_results.py
