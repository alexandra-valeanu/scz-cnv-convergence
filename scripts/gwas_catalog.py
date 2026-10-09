"""GWAS Catalog common-variant gene sets, filtered as in CARVA notebook 1A/1B."""

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GWAS_DIR = ROOT / "input" / "gwas_catalog"

RELEASE = "r2026-09-04"
STUDIES_FILE = GWAS_DIR / f"gwas-catalog-v1.0.3.1-studies-{RELEASE}.tsv"
ASSOC_ZIP = GWAS_DIR / f"gwas_catalog_v1.0.2-associations_e116_{RELEASE}_full.zip"

P_THRESH = 5e-8
MIN_GENES_PER_STUDY = 3
MIN_ASSOC_PER_STUDY = 3
MAX_SEEDS = 500

GENOME_WIDE = ("Genome-wide genotyping array",
               "Genome-wide genotyping array, Genome-wide sequencing")

ASSOC_COLS = ["DISEASE/TRAIT", "MAPPED_GENE", "SNP_GENE_IDS", "P-VALUE",
              "MAPPED_TRAIT", "MAPPED_TRAIT_URI", "INTERGENIC", "STUDY ACCESSION"]

# Trait codes read from the release named above. The GWAS Catalog remaps traits
# between releases, so a code that returns no rows is an error, not an empty set.
TRAIT_CODES = {
    "crc":     ("MONDO_0005575", "colorectal cancer"),
    "lung":    ("MONDO_0005138", "lung carcinoma"),
    "crohn":   ("MONDO_0005011", "Crohn disease"),
    "ra":      ("MONDO_0008383", "rheumatoid arthritis"),
    "thyroid": ("MONDO_0005623", "autoimmune thyroid disease"),
    "t2d":     ("MONDO_0005148", "type 2 diabetes mellitus"),
    "obesity": ("MONDO_0011122", "obesity disorder"),
    "cad":     ("MONDO_0005010", "coronary artery disorder"),
}

_CACHE = {}


def trait_code(key):
    return TRAIT_CODES[key][0]


def studies():
    """Study table after the notebook 1A study filters."""
    if "studies" in _CACHE:
        return _CACHE["studies"]

    s = pd.read_csv(STUDIES_FILE, sep="\t", low_memory=False)
    n0 = len(s)
    s = s[s["MAPPED BACKGROUND TRAIT"].isna()]
    # A few studies carry no genotyping technology; 1A assumes a string here.
    s = s[s["GENOTYPING TECHNOLOGY"].notna()]
    tech = s["GENOTYPING TECHNOLOGY"].astype(str).str.split("[", regex=False).str[0].str.strip()
    s = s[tech.isin(GENOME_WIDE)]
    s = s.dropna(subset=["DISEASE/TRAIT", "MAPPED_TRAIT", "MAPPED_TRAIT_URI"])
    s = s[~s["MAPPED_TRAIT_URI"].str.contains(",", regex=False)]
    s = s[s["ASSOCIATION COUNT"] >= MIN_ASSOC_PER_STUDY]
    print(f"GWAS studies: {n0} -> {len(s)} after 1A filters")

    _CACHE["studies"] = s
    return s


def associations():
    """Association table after the notebook 1A and 1B filters."""
    if "assoc" in _CACHE:
        return _CACHE["assoc"]

    with zipfile.ZipFile(ASSOC_ZIP) as z:
        members = [n for n in z.namelist() if n.lower().endswith((".tsv", ".txt")) and not n.startswith("__MACOSX/")]
        if len(members) != 1:
            raise ValueError("Association ZIP must contain exactly one TSV/TXT table")
        with z.open(members[0]) as fh:
            d = pd.read_csv(fh, sep="\t", usecols=ASSOC_COLS, low_memory=False)
    n0 = len(d)

    d["P-VALUE"] = pd.to_numeric(d["P-VALUE"], errors="coerce")
    d = d[(d["P-VALUE"] >= 0) & (d["P-VALUE"] <= P_THRESH)]
    d = d.dropna(subset=["SNP_GENE_IDS", "MAPPED_TRAIT_URI"])
    d = d[d["INTERGENIC"] == 0]
    d = d[~d["SNP_GENE_IDS"].str.contains(r"[,;\s]", regex=True)]
    d = d[~d["MAPPED_TRAIT_URI"].str.contains(",")]
    d = d[d["STUDY ACCESSION"].isin(set(studies()["STUDY ACCESSION"]))]

    per_study = d.groupby("STUDY ACCESSION")["SNP_GENE_IDS"].nunique()
    d = d[d["STUDY ACCESSION"].isin(per_study[per_study >= MIN_GENES_PER_STUDY].index)]
    d["TRAIT_CODE"] = d["MAPPED_TRAIT_URI"].str.split("/").str[-1]
    print(f"GWAS associations: {n0} -> {len(d)} after 1A/1B filters")

    _CACHE["assoc"] = d
    return d


def disease_rows(key):
    """Filtered associations for one disease, best p-value per Ensembl gene."""
    code = trait_code(key)
    rows = associations()
    rows = rows[rows["TRAIT_CODE"] == code]
    if rows.empty:
        raise ValueError(f"no GWAS Catalog rows for {key} ({code}) in {RELEASE}; "
                         "the trait may have been remapped in this release")

    per_gene = (rows.groupby("SNP_GENE_IDS")
                .agg(symbol_reported=("MAPPED_GENE", "first"),
                     p_value=("P-VALUE", "min"),
                     n_studies=("STUDY ACCESSION", "nunique"),
                     study_accessions=("STUDY ACCESSION", lambda v: ";".join(sorted(set(v)))))
                .reset_index()
                .rename(columns={"SNP_GENE_IDS": "ensembl_gene_id"}))
    return per_gene.sort_values(["p_value", "ensembl_gene_id"], kind="stable").reset_index(drop=True)


def trait_studies(key):
    rows = associations()
    accessions = set(rows.loc[rows["TRAIT_CODE"] == trait_code(key), "STUDY ACCESSION"])
    return studies()[studies()["STUDY ACCESSION"].isin(accessions)].copy()


def main(argv=None):
    """Extract frozen trait inputs and create a separate runnable disorder config."""
    import argparse
    import os
    import re
    from cnv_overlap.provenance import file_hash, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trait', action='append', metavar='NAME=ONTOLOGY_ID',
                        help='Repeat for additional traits, e.g. t2d=MONDO_0005148; default: all eight original traits')
    parser.add_argument('--studies', type=Path, default=STUDIES_FILE)
    parser.add_argument('--associations', type=Path, default=ASSOC_ZIP, help='Catalog association ZIP containing one TSV')
    parser.add_argument('--mapping', type=Path, default=ROOT/'input/processed/mapping_snapshot.tsv',
                        help='Frozen ensembl_gene_id/entrez TSV; supply an expanded mapping for new traits as needed')
    parser.add_argument('--base-config', type=Path, default=ROOT/'configs/cnv_disorder_overlap.json')
    parser.add_argument('--output-dir', required=True, type=Path, help='New snapshot directory inside this repository')
    parser.add_argument('--p-threshold', type=float, default=P_THRESH)
    parser.add_argument('--min-genes-per-study', type=int, default=MIN_GENES_PER_STUDY)
    parser.add_argument('--min-associations-per-study', type=int, default=MIN_ASSOC_PER_STUDY)
    parser.add_argument('--max-seeds', type=int, default=MAX_SEEDS)
    args = parser.parse_args(argv)
    if not 0 < args.p_threshold <= 1 or min(args.min_genes_per_study, args.min_associations_per_study, args.max_seeds) < 1:
        parser.error('Threshold must be in (0, 1] and counts must be positive')
    selected = dict(TRAIT_CODES)
    if args.trait:
        selected = {}
        for value in args.trait:
            key, sep, code = value.partition('=')
            if not sep or not re.fullmatch(r'[a-z][a-z0-9_]*', key) or not re.fullmatch(r'[A-Za-z]+_[0-9]+', code):
                parser.error('Use --trait name=ONTOLOGY_123 with a lowercase name and an ontology ID')
            if key in selected:
                parser.error(f'Duplicate trait name: {key}')
            selected[key] = (code, key)
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Output directory must be inside the repository for portable relative input paths')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be new or empty; existing snapshots are not overwritten')
    base_path = args.base_config.resolve()
    spec = json.loads(base_path.read_text())
    base_root = (base_path.parent/spec.get('root', '..')).resolve()
    paths = {k: (base_root/v).resolve() for k, v in spec['inputs'].items()}
    for key, expected in spec['input_sha256'].items():
        if file_hash(paths[key]) != expected:
            raise ValueError(f'Base input changed: {key}')
    if spec['analysis'].get('comparisons') != 'cnv_disorder':
        parser.error('Base config must select cnv_disorder comparisons')
    seeds = pd.read_csv(paths['seeds'], sep='\t')
    if set(selected) & set(seeds.loc[seeds.kind != 'target', 'entity']):
        parser.error('A trait name conflicts with an existing region name')
    mapping = pd.read_csv(args.mapping, sep='\t').dropna(subset=['ensembl_gene_id', 'entrez'])
    numeric = pd.to_numeric(mapping.entrez)
    if (numeric <= 0).any() or (numeric % 1 != 0).any():
        raise ValueError('Mapping Entrez identifiers must be positive integers')
    mapping.entrez = numeric.astype(int)
    counts = mapping.groupby('ensembl_gene_id').entrez.nunique()
    conflicts = mapping[mapping.ensembl_gene_id.isin(counts[counts > 1].index)]
    mapping = mapping[~mapping.ensembl_gene_id.isin(conflicts.ensembl_gene_id)].drop_duplicates('ensembl_gene_id')
    map_id = dict(zip(mapping.ensembl_gene_id, mapping.entrez))
    network = json.loads(paths['network_nodes'].read_text())
    on_network = {int(n['r'].split(':')[-1]) for n in network}

    # The recovered filter functions retain their original importable interface.
    # Configure them for this invocation and clear release-specific cached tables.
    globals().update(STUDIES_FILE=args.studies.resolve(), ASSOC_ZIP=args.associations.resolve(),
                     P_THRESH=args.p_threshold, MIN_GENES_PER_STUDY=args.min_genes_per_study,
                     MIN_ASSOC_PER_STUDY=args.min_associations_per_study, TRAIT_CODES=selected,
                     RELEASE=args.associations.name)
    _CACHE.clear()
    tables, selected_rows, qc = {}, [], []
    for key in selected:
        frame = disease_rows(key)
        frame['entrez'] = frame.ensembl_gene_id.map(map_id).astype('Int64')
        hits = frame.dropna(subset=['entrez']).sort_values(['p_value', 'ensembl_gene_id']).drop_duplicates('entrez')
        hits = hits[hits.entrez.isin(on_network)].head(args.max_seeds)
        if hits.empty:
            raise ValueError(f'{key}: no mapped on-network seeds; supply an expanded frozen mapping')
        tables[key] = frame
        selected_rows.extend({'entity': key, 'kind': 'target', 'entrez': int(g),
                              'source': f'GWAS_Catalog_{trait_code(key)}_intragenic_top{args.max_seeds}'} for g in hits.entrez)
        qc.append({'trait': key, 'code': trait_code(key), 'n_genes_reported': len(frame),
                   'n_unmapped': int(frame.entrez.isna().sum()), 'n_selected': len(hits),
                   'n_studies': len(trait_studies(key))})
    seeds = pd.concat([seeds[~seeds.entity.isin(selected)], pd.DataFrame(selected_rows)], ignore_index=True)
    seeds = seeds.sort_values(['entity', 'entrez']).reset_index(drop=True)
    # Build everything before writing; the primary snapshot/config is never edited.
    output.mkdir(parents=True, exist_ok=True)
    for key, frame in tables.items():
        frame.to_csv(output/f'{key}_associations.tsv', sep='\t', index=False)
    pd.DataFrame(qc).to_csv(output/'trait_qc.tsv', sep='\t', index=False)
    conflicts.to_csv(output/'mapping_conflicts.tsv', sep='\t', index=False)
    mapping.to_csv(output/'mapping_snapshot.tsv', sep='\t', index=False)
    seeds.to_csv(output/'seeds.tsv', sep='\t', index=False)
    source = {'base_config': str(base_path.relative_to(ROOT)), 'base_config_sha256': file_hash(base_path),
              'base_input_sha256': spec['input_sha256'], 'traits': selected,
              'source_files': {str(p): file_hash(p) for p in [args.studies, args.associations, args.mapping]},
              'script_sha256': file_hash(Path(__file__)),
              'filters': {'p_threshold': args.p_threshold, 'min_genes_per_study': args.min_genes_per_study,
                          'min_associations_per_study': args.min_associations_per_study,
                          'max_seeds': args.max_seeds, 'genotyping_technology': GENOME_WIDE,
                          'intragenic_single_gene_single_trait': True},
              'mapping_policy': 'frozen Ensembl to Entrez; conflicting mappings rejected; no live/symbol fallback',
              'output_sha256': {p.name: file_hash(p) for p in sorted(output.glob('*.tsv'))}}
    write_json(output/'source_manifest.json', source)
    paths.update(seeds=output/'seeds.tsv', source_manifest=output/'source_manifest.json')
    spec['root'] = os.path.relpath(ROOT, output)
    spec['inputs'] = {k: str(v.relative_to(ROOT)) for k, v in paths.items()}
    spec['input_sha256'] = {k: file_hash(v) for k, v in paths.items()}
    spec['seed_definition'] = 'Base seed definitions retained for unchanged entities; added/replaced Catalog traits: '+', '.join(selected)+'; exact filters, mapping and base input hashes in source_manifest'
    write_json(output/'analysis.json', spec)
    print(pd.DataFrame(qc).to_string(index=False))
    print(f'Created {output / "analysis.json"}')
    print('Set CNV_CONFIG to this config before running notebooks/cnv_disorder_overlap.ipynb.')


if __name__ == '__main__':
    import json
    main()
