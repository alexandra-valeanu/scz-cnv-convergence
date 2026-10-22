#!/bin/bash
# Submit from any working directory; Slurm opens logs before the job script runs.
# Usage: NETWORK=PCNET2.2_brain bash jobs/submit.sh pipeline [sbatch options]
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
job=${1:-}
case "$job" in
    pipeline|cilia|nps_stability|pvalue_stability) shift ;;
    *) echo "Usage: $0 {pipeline|cilia|nps_stability|pvalue_stability} [sbatch options]" >&2; exit 2 ;;
esac
mkdir -p "$root/logs"
cd "$root"
exec sbatch "$@" --chdir="$root" \
    --output="$root/logs/slurm-%x-%j.out" \
    --error="$root/logs/slurm-%x-%j.err" "$root/jobs/${job}_job.sh"
