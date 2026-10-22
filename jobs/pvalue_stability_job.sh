#!/bin/bash
#SBATCH --job-name=pvalue-stability
#SBATCH --partition=defq
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Stability: pair-test p-values across RNG seeds
# -> results/<network>/6_robustness/pvalue_stability
# Reuses the NPS references cached by nps_stability_job.sh, so run that first:
#   sbatch --dependency=afterok:<nps job id> jobs/pvalue_stability_job.sh
#
# Submit from the repository root:
#   NETWORKS="PCNET2.2_brain STRING_high_conf_brain" sbatch jobs/pvalue_stability_job.sh
#
# Parameters (all optional):
#   NETWORKS      default "PCNET2.2 STRING_high_conf"
#   SEEDS         RNG seeds, at least two (default "1 42 50")
#   MIN_SEEDS     default 3
#   TARGET_SCOPE  default | all (default: default)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f jobs/setup_venv.sh ]] || { echo "Submit from the repository root (jobs/setup_venv.sh not found)" >&2; exit 1; }
source jobs/setup_venv.sh
NETWORKS=${NETWORKS:-PCNET2.2 STRING_high_conf}
SEEDS=${SEEDS:-1 42 50}
read -r -a seeds <<< "$SEEDS"
(( ${#seeds[@]} >= 2 )) || { echo "Need at least two SEEDS to compare seeds, got: $SEEDS" >&2; exit 1; }

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
unset CNV_CONFIG CNV_NULL_METHOD CNV_NETWORK
echo "Commit $(git rev-parse --short HEAD), started $(date)"

set -x
.venv/bin/python -u scripts/pvalue_stability.py --networks $NETWORKS --seeds $SEEDS \
    --min-seeds "${MIN_SEEDS:-3}" --target-scope "${TARGET_SCOPE:-default}" --workers 1
.venv/bin/python -u scripts/index_results.py
