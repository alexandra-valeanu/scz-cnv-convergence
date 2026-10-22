#!/bin/bash
#SBATCH --job-name=cnv-pipeline
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Main pipeline for one network: seed propagation notebooks, seed sets, pair network,
# landscape then CNV Reactome profiles -> results/<network>/0_inputs ... 4_cnv_reactome_profiles
#
# Submit from the repository root:
#   NETWORK=PCNET2.2_brain sbatch jobs/pipeline_job.sh
#   NETWORK=PCNET2.2 FROM_STAGE=landscape sbatch jobs/pipeline_job.sh
#   NETWORK=PCNET2.2 STAGES="notebooks index" NOTEBOOKS=cnv_convergence sbatch --time=12:00:00 jobs/pipeline_job.sh
#
# Parameters (all optional except NETWORK):
#   NETWORK     PCNET2.2 | PCNET2.2_brain | STRING_high_conf | STRING_high_conf_brain
#   STAGES      run only these stages: notebooks seed_sets pair_network landscape cnv_reactome index
#   FROM_STAGE  first stage to run (default: notebooks)
#   TO_STAGE    last stage to run (default: index)
#   NOTEBOOKS   cnv_convergence and/or cnv_disorder_overlap (default: both)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f jobs/setup_venv.sh ]] || { echo "Submit from the repository root (jobs/setup_venv.sh not found)" >&2; exit 1; }
source jobs/setup_venv.sh
[[ -n "${NETWORK:-}" ]] || { echo "Set NETWORK, e.g. NETWORK=PCNET2.2_brain sbatch jobs/pipeline_job.sh" >&2; exit 1; }

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
unset CNV_CONFIG CNV_NULL_METHOD CNV_NETWORK
echo "Commit $(git rev-parse --short HEAD), started $(date)"

args=(--network "$NETWORK")
[[ -n "${RESULTS_ROOT:-}" ]] && args+=(--results-root "$RESULTS_ROOT")
[[ -n "${RESUME:-}" ]] && args+=(--resume "$RESUME")
[[ -n "${CNV_PAIR_RUN:-}" ]] && args+=(--cnv-pair-run "$CNV_PAIR_RUN")
[[ -n "${STAGES:-}" ]] && args+=(--stages $STAGES)
[[ -n "${FROM_STAGE:-}" ]] && args+=(--from-stage "$FROM_STAGE")
[[ -n "${TO_STAGE:-}" ]] && args+=(--to-stage "$TO_STAGE")
[[ -n "${NOTEBOOKS:-}" ]] && args+=(--notebooks $NOTEBOOKS)

set -x
.venv/bin/python -u scripts/run_full_results.py "${args[@]}"
