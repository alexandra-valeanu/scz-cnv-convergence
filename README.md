# CNV network convergence

Network propagation and colocalization of schizophrenia-associated CNV gene
sets with SCHEMA rare-variant genes, schizophrenia GWAS genes and GWAS gene
sets for other diseases.

See [methods](docs/methods.md) for the statistical details,
[schizophrenia landscape](docs/schizophrenia_landscape.md) for that workflow stage,
and [results](results/README.md) for the output layout.

## Networks

| Identifier | Network |
| --- | --- |
| `PCNET2.2` | Full PCNet 2.2 |
| `PCNET2.2_brain` | PCNet 2.2 restricted to brain-expressed genes |
| `STRING_high_conf` | Human STRING v12.0 physical interactions, combined score ≥700 |
| `STRING_high_conf_brain` | The same STRING network restricted to brain-expressed genes |

- **Brain filter:** a gene counts as brain-expressed when it has >1 TPM in at
  least one of 13 HPA brain regions.
- **STRING edges:** only physical interactions are used, not STRING's full
  functional-association network.
- **CNV seeds:** CNVs with only two mapped seed genes are kept, so all nine loci
  are analysed. Seed QC flags these sparse mappings.

All network snapshots and identifier mappings are checksummed.

## Installation

Requires Python 3.11 or newer.

```bash
uv sync --locked --extra test --extra plots
bash scripts/install_orsum.sh .venv
```

Alternatively, install `requirements.txt` into a fresh virtual environment.
NetColoc is pinned to a Git commit and orsum to version 1.8.0.

## Input data

This repository contains the code, the configs and the small processed input
files: the CNV seed sets, coordinates and mapping QC. It does not contain the
raw downloads, which are listed with their SHA-256 checksums in
[input/required_files.tsv](input/required_files.tsv):

- PCNet 2.2 and STRING
- HPA brain expression
- GWAS Catalog
- the SCHEMA and GWAS spreadsheets
- the annotation libraries

Restore these files under `input/` before running the full workflow. The
configs check the checksums and stop if any file differs. See
[input/README.md](input/README.md) for each folder's contents and provenance.

## Running

Run all commands from the repository root:

```bash
# Prepare the annotation libraries once (the MitoCarta download needs internet access)
.venv/bin/python scripts/prepare_publication_annotations.py

# Show the four-network plan without running any computation
.venv/bin/python scripts/publication_pipeline.py --plan

# Run all four networks, or pick one with --networks PCNET2.2_brain
.venv/bin/python scripts/publication_pipeline.py

# Rebuild the results index and verify the existing outputs
.venv/bin/python scripts/index_results.py --verify
```

**Slurm:** run `mkdir -p logs`, then `sbatch jobs/job.sh`. The job requests 4 CPUs
and 40 GB of memory and runs the networks one after another. To run a single
network, set `NETWORKS="PCNET2.2_brain"` before you submit. Dense propagation
kernels can be large, so use compute nodes for full runs.

[configs/publication.json](configs/publication.json) sets the networks, input
configs, enrichment thresholds, Leiden resolution and conditional-null budget.
Outputs go to `results/publication/<network>/publication-<hash>/`. The
pipeline writes `current.json` only after all main stages have finished.
Completed runs are reused only if they pass integrity checks. When choosing a
source run, use `current.json`, not the newest timestamp.

## Analyses

1. **CNV pairs:** every pairwise test, a heatmap of size z-scores, and a circle
   plot of the significant pairs. Edge width shows observed minus expected
   shared genes.
2. **CNV × SCHEMA/GWAS:** the colocalized genes for each pair, with descriptive
   enrichment, separate SynGO, MitoCarta and cell-type panels, and orsum
   summaries.
3. **Schizophrenia landscape:** genes colocalized between SCHEMA and GWAS,
   independent of the CNVs. Leiden modules are named by GO biological process.
4. **CNVs in the landscape:** each CNV's direct propagation reach, seed status,
   stricter joint colocalization, recurrence and conditional-null tests.
5. **CNV × disease:** all tested diseases, with significant pairs joined to
   [dosage-specific literature evidence](input/literature/README.md).

**Limitations:** the CNVs are unsigned because deletion and duplication labels
are not available. Network colocalization is not a clinical risk estimate.
A literature phenotype that has no available gene set is recorded as untested,
not as a negative result.

## Supplementary stability analyses

NPS and p-value stability are separate jobs, and the main workflow does not
rerun them:

```bash
.venv/bin/python scripts/nps_stability.py --networks PCNET2.2 PCNET2.2_brain STRING_high_conf STRING_high_conf_brain --reps 1000 5000 10000 --seeds 1 42 50 --min-seeds 2 --target-scope all
.venv/bin/python scripts/pvalue_stability.py --networks PCNET2.2 PCNET2.2_brain STRING_high_conf STRING_high_conf_brain --seeds 1 42 50 --min-seeds 2 --target-scope all
```

Slurm wrappers: `jobs/run_nps_stability.sh` and
`jobs/run_pvalue_stability.sh`.

**Target scope:** the existing p-value stability results for the brain networks
use the SCHEMA/GWAS target scope, while the main disease panel also includes
non-schizophrenia diseases. Report the scope explicitly.

## Notebooks

The four notebooks in `notebooks/` are optional ways to inspect the results. Run them
with `scripts/run_network_notebooks.py`. Executed copies are saved as
`results/notebooks/executed_<network>_<notebook>_<YYYY-MM-DD>.ipynb`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/cnv_overlap/` | Propagation, statistical tests, nulls, enrichment and provenance |
| `scripts/publication_pipeline.py` | Main workflow |
| `scripts/` | Annotation preparation, analysis helpers and supplementary workflows |
| `jobs/` | Slurm job scripts (submit from the repository root) |
| `configs/` | Analysis settings and source references |
| `input/` | Frozen inputs, provenance records and literature curation |
| `notebooks/` | Optional inspection notebooks (run from the repository root) |
| `tests/` | Statistical, workflow and reporting tests |
| `docs/` | Analysis methods and the schizophrenia-landscape workflow |
| `results/`, `data/cache/` | Outputs and propagation caches (generated locally, not tracked) |

## Tests

```bash
.venv/bin/python -m pytest -q
```
