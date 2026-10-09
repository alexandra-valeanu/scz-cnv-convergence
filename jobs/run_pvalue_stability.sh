#!/bin/bash
#SBATCH --job-name=pvalue-stability
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-pvalue-%x-%A_%a.out
#SBATCH --error=logs/slurm-pvalue-%x-%A_%a.err

# Submit from the repository root. NPS references for seeds 1/50 are shared with
# run_nps_stability.sh through the propagation cache, so running that job first avoids recomputation:
set -euo pipefail

source .venv/bin/activate

# Example: NETWORKS="PCNET2.2_brain STRING_high_conf_brain" sbatch jobs/run_pvalue_stability.sh
read -r -a networks <<< "${NETWORKS:-PCNET2.2 STRING_high_conf}"
# Fixed RNG seeds, identical to run_nps_stability.sh; override with PVALUE_SEEDS="..." if needed.
read -r -a seeds <<< "${PVALUE_SEEDS:-1 42 50}"
if (( ${#seeds[@]} < 2 )); then
    echo "Need at least two RNG seeds to assess seed agreement, got: ${seeds[*]}" >&2
    exit 1
fi
echo "P-value stability: network=${networks[*]} seeds=${seeds[*]}"

.venv/bin/python -u scripts/pvalue_stability.py \
    --networks "${networks[@]}" --min-seeds "${MIN_SEEDS:-3}" --target-scope "${TARGET_SCOPE:-default}" --workers 1 \
    --seeds "${seeds[@]}"
