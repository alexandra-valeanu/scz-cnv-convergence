#!/bin/bash
#SBATCH --job-name=nps-stability
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-nps-%x-%A_%a.out
#SBATCH --error=logs/slurm-nps-%x-%A_%a.err

# Submit from the repository root.
set -euo pipefail

source .venv/bin/activate

# Example: NETWORKS="PCNET2.2_brain STRING_high_conf_brain" sbatch jobs/run_nps_stability.sh
read -r -a networks <<< "${NETWORKS:-PCNET2.2 STRING_high_conf}"
# Fixed RNG seeds for the Monte Carlo reference; override with NPS_SEEDS="..." if needed.
read -r -a seeds <<< "${NPS_SEEDS:-1 42 50}"
if (( ${#seeds[@]} < 2 )); then
    echo "Need at least two RNG seeds to assess seed agreement, got: ${seeds[*]}" >&2
    exit 1
fi
echo "NPS stability: network=${networks[*]} seeds=${seeds[*]} reps=1000 5000 10000"

.venv/bin/python -u scripts/nps_stability.py \
    --networks "${networks[@]}" --min-seeds "${MIN_SEEDS:-3}" --target-scope "${TARGET_SCOPE:-default}" --workers 1 \
    --reps 1000 5000 10000 --seeds "${seeds[@]}"
