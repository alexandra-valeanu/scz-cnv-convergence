# Cluster jobs

| Job | What it does | Results |
| --- | --- | --- |
| `pipeline_job.sh` | Main pipeline: seed propagation notebooks, seed sets, pair network, landscape, CNV Reactome profiles | `0_inputs` … `4_cnv_reactome_profiles` |
| `cilia_job.sh` | Extra analysis: CNV/SCHEMA/GWAS × cilia gene sets | `5_cilia` |
| `nps_stability_job.sh` | Stability of NPS across reference sizes and seeds | `6_robustness/nps_stability` |
| `pvalue_stability_job.sh` | Stability of p-values across seeds | `6_robustness/pvalue_stability` |

Submit from the repository root. Parameters are environment variables, listed at
the top of each job; resources can be changed with `sbatch` flags.

Alternatively, use `submit.sh` from any directory. It sets the repository working
directory and creates the log directory **before** Slurm opens its output files:

```bash
cd /home/alexandra.valeanu/cnv_convergence_pipeline/jobs
NETWORK=PCNET2.2_brain bash submit.sh pipeline
NETWORK=PCNET2.2 bash submit.sh pipeline --time=48:00:00
```

Direct `sbatch pipeline_job.sh` from inside `jobs/` is not supported; use the
launcher above or submit `jobs/pipeline_job.sh` from the repository root.

Each job first sources `setup_venv.sh`, which creates `.venv` (uv with `uv.lock`,
else `python3 -m venv` + pip) and installs orsum. It only reinstalls when
`pyproject.toml`, `uv.lock` or `requirements.txt` change, and uses a lock so jobs
submitted together don't install at the same time. Compute nodes need internet for
the first install; otherwise run `bash -c 'source jobs/setup_venv.sh'` once on the
login node.

The orsum installer is also included in the setup fingerprint. For separate
output trees, the main job accepts `RESULTS_ROOT`; all analysis jobs and the index
also honour an exported `CNV_RESULTS_ROOT` (prefer an absolute path).

```bash
mkdir -p logs

# Main pipeline, one job per network
NETWORK=PCNET2.2_brain sbatch jobs/pipeline_job.sh
NETWORK=PCNET2.2 FROM_STAGE=landscape sbatch jobs/pipeline_job.sh
NETWORK=PCNET2.2 STAGES="notebooks index" sbatch --time=12:00:00 jobs/pipeline_job.sh

# Cilia
NETWORK=PCNET2.2_brain sbatch jobs/cilia_job.sh

# Stability (p-value job reuses the NPS cache, so chain it)
nps=$(NETWORKS=PCNET2.2_brain sbatch --parsable jobs/nps_stability_job.sh)
NETWORKS=PCNET2.2_brain sbatch --dependency=afterok:$nps jobs/pvalue_stability_job.sh
```

Logs go to `logs/slurm-<job name>-<job id>.out`; each starts with the git commit
and shows every command it runs.
