# Frozen pathway annotation libraries

These GMT files are stored locally for offline over-representation analysis.
`sources.json` records provider URLs, retrieval timestamps, and SHA-256 checksums.

- **GO Biological Process 2026:** 5,954 terms, distributed by Enrichr.
- **Reactome:** 2,868 human pathways, extracted unchanged from the
  [official GMT archive](https://reactome.org/download/current/ReactomePathways.gmt.zip).
  `ReactomePathways.gmt` contains pathway names, stable Reactome IDs in the
  description column, and gene symbols. The archive checksum is also recorded.

Reactome's `current` URL changes with releases. This project uses the downloaded
snapshot, identified by its timestamp and checksum; running a notebook does not
refresh it. The download does not specify a release number in the GMT, so none is
inferred. This replaces the previous Enrichr Reactome Pathways 2024 snapshot.

Only genes in the analysis background contribute; annotation coverage is written
for each pair. No query genes are uploaded. Keep these files with the transferred
project. A new snapshot should use a separate report output directory.

Please cite the Gene Ontology and Reactome resources, and Enrichr for its GO
library, when publishing analyses using these files.

Sources: https://maayanlab.cloud/Enrichr/help and https://reactome.org/download-data

- **GO Molecular Function 2026:** 1,394 terms, distributed by Enrichr;
  downloaded from its `geneSetLibrary` endpoint and frozen in
  `GO_Molecular_Function_2026.gmt`. Retrieval timestamp, URL and checksum are
  recorded in `sources.json`. Used by the shared-network enrichment pipeline.

- **GO Cellular Component 2026:** 510 terms from Enrichr, frozen in
  `GO_Cellular_Component_2026.gmt` with retrieval provenance in `sources.json`.
  Shared-network enrichment now corrects across Reactome, GO BP, GO CC and GO MF;
  the first three are summarized separately by orsum.

## Cluster-context libraries

`scripts/fetch_cluster_annotations.py` freezes these libraries; URLs, retrieval
times and checksums are in `sources.json`. The script leaves existing files unchanged. They are used by
`scripts/shared_network_cluster_context.py`, which annotates the shared core and its
Leiden clusters. BH correction is applied separately within each annotation group.

- **Function:** the GO and Reactome libraries above; Reactome is the only pathway database.
- **Synapse:** SynGO 2024 (134 expert-curated synaptic GO terms), from Enrichr.
- **Cell type:** Allen Brain Atlas 10x scRNA 2021, restricted to the 111 human MTG
  "Human ... up" marker sets, from Enrichr.
- **Tissue, brain region, cell type:** Human Protein Atlas `proteinatlas.tsv.zip`
  (Last-Modified 3 June 2026, archived unchanged). The derived GMTs list HPA-*elevated* genes, meaning
  tissue/region/cell-type enriched, group enriched or enhanced, for each context:
  `HPA_tissue_elevated.gmt` (37 tissues), `HPA_brain_region_elevated.gmt` (13 regions),
  `HPA_brain_cell_type_elevated.gmt` (34 snRNA brain cell types) and
  `HPA_single_cell_type_elevated.gmt` (154 cell types). `HPA_gene_specificity.tsv`
  keeps the per-gene categories.

Please cite SynGO (Koopmans et al., Neuron 2019), the Allen
Brain Atlas, the Human Protein Atlas (Uhlén et al., Science 2015; Karlsson et al.,
Sci Adv 2021) and Enrichr when you publish these analyses.

## Publication libraries

`scripts/prepare_publication_annotations.py` creates these from the files above.

- **MitoCarta 3.0:** `MitoPathways3.0.gmx` is downloaded from the Broad Institute
  and converted unchanged to `MitoCarta3.0.gmt` (149 mitochondrial pathways).
- **HPA broad brain cell types:** `HPA_brain_broad_cell_types.gmt` keeps the eight
  broad brain cell classes from `HPA_single_cell_type_elevated.gmt`.

Please cite MitoCarta3.0 (Rath et al., Nucleic Acids Res 2021) when publishing.
