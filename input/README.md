# Inputs

[required_files.tsv](required_files.tsv) lists every input file with its size,
SHA-256 checksum, role and source. The configs and loaders check these
checksums, so a file that has changed stops the run instead of silently
changing the results.

## How each file is obtained

| How | Files | What to do |
| --- | --- | --- |
| **In the repository** | `processed/`, `literature/`, all `manifest.json` / `sources.json` records | Nothing |
| **Downloaded automatically** | STRING v12.0 (`interactome/STRING_high_conf/`) | Built by `scripts/network_inputs.py` on the first STRING run |
| | HPA brain expression (`expression/HPA_brain/`) | Downloaded by `scripts/brain_inputs.py` on the first brain-network run |
| | SynGO, Allen Brain Atlas, HPA archive and the HPA-derived GMTs (`annotations/`) | `python scripts/fetch_cluster_annotations.py` |
| | MitoCarta pathways and HPA broad brain cell types (`annotations/`) | `python scripts/prepare_publication_annotations.py` (run after the line above) |
| **Downloaded by hand** | PCNet 2.2 (`interactome/pcnet22_*.json`) | Supply locally; the download URL was not recorded |
| | GO BP/CC/MF 2026 and Reactome (`annotations/`) | Download from the URLs in `annotations/sources.json` |
| | GWAS Catalog r2026-09-04 (`gwas_catalog/`) | Only needed to rebuild trait gene sets with `scripts/gwas_catalog.py` |
| | SCHEMA, PGC3 Supplementary Table 12, CNV interval and coordinate tables | Original curation sources; not read by the pipeline |

The download scripts only fetch files that are missing and leave existing files
unchanged. Online resources change over time. A new download will have a
different checksum, which counts as a new snapshot rather than a reproduction.
Use the archived files to reproduce the published results exactly.

## Folders

| Folder | Contents | Read by the pipeline |
| --- | --- | --- |
| `interactome/` | PCNet 2.2 node and edge JSON; `STRING_high_conf/` physical network (combined score ≥700) with its Entrez mapping | Yes |
| `expression/HPA_brain/` | HPA brain-region TPM table and region list, used for the brain-expressed filter | Yes |
| `annotations/` | Enrichment libraries; see [annotations/README.md](annotations/README.md) | Yes |
| `processed/` | Frozen seed sets, coordinates, identifier mapping and trait QC | Yes |
| `literature/` | Dosage-specific CNV-disease literature evidence; see [literature/README.md](literature/README.md) | Yes |
| `gwas_catalog/` | GWAS Catalog studies, associations and trait mappings | Only by `scripts/gwas_catalog.py` |
| `scz_cnv_regions/` | Original SCZ CNV intervals and mapped genes | No (curation source) |
| `control_cnv_regions/` | Original control-region gene table | No (curation source) |
| `rare_scz_variant_genes/` | SCHEMA results spreadsheet | No (curation source) |
| `common_scz_variant_genes/` | PGC3 prioritised genes, Supplementary Table 12 | No (curation source) |
| `coordinates/` | Original GRCh37 gene coordinate table | No (curation source) |
| `deg/` | Differential expression spreadsheet | No (not used by the current analyses) |

## Processed files

- `cnv_convergence_seeds.tsv`: the nine SCZ CNV gene sets.
- `cnv_disorder_overlap_seeds.tsv`: the same CNVs plus nine disorder and trait
  targets (T1D excluded).
- `coordinates.tsv`: curated GRCh37 coordinates. `regions.tsv`: original region
  intervals, documenting the CNV gene content.
- `seeds.tsv`: the original combined snapshot, including control regions. It is
  kept as curation evidence; no notebook or pipeline stage reads it.
- `mapping_snapshot.tsv`, `mapping_conflicts.tsv`, `coordinate_conflicts.tsv`:
  frozen identifier mapping and ambiguity audits.
- `*_associations.tsv`, `trait_qc.tsv`: per-trait GWAS evidence and selection QC.
- `source_manifest.json`: checksums of the curation sources. Its paths show where
  the files were during curation (some under an older `data/` folder); the
  pipeline does not read them.

## Adding or changing inputs

Never replace a checksummed file in place. Put a new GWAS extraction in its own
folder (`gwas_catalog/trait_sets/<snapshot>/`) and a new annotation release in a
new annotation directory, so an existing analysis can't change without notice.

Raw downloads are excluded from Git. Only this guide, the checksum list,
`processed/`, `literature/` and the small provenance records are tracked.
