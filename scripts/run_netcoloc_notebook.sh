#!/bin/bash
# Execute one optional inspection notebook using the prepared environment.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
case "${1:-}" in
    cnv_brain_convergence|cnv_brain_disorder_overlap) network=PCNET2.2_brain ;;
    cnv_disorder_overlap) network=PCNET2.2 ;;
    *) echo 'Expected a configured notebook name' >&2; exit 1 ;;
esac
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset CNV_CONFIG CNV_NULL_METHOD
exec .venv/bin/python -u scripts/run_network_notebooks.py --networks "$network" --notebooks "$1"
