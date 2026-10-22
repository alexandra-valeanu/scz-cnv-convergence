"""Test CNV, SCHEMA, and GWAS colocalization with curated cilia gene sets.

Each cilia set uses the reference run settings and a separate BH family across
all queries. SCHEMA/GWAS use kind="cnv" for pair planning; query_type records
their biological role. Ensembl IDs map through unique source/HPA symbols and
explicit HGNC renames. This is an enrichment-motivated exploratory analysis."""
from cnv_overlap.results import network_root as configured_network_root, resolve_path
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT/'scripts'))

from cnv_overlap.pipeline import AnalysisConfig, run
from cnv_overlap.provenance import atomic_path, file_hash, write_json
from network_inputs import NETWORKS, load_network_inputs
from results_layout import results_dir

SETS = ('scgsv2_first_order', 'wp4803_ciliopathies')
# Schizophrenia gene sets tested alongside the CNVs (entity name -> query_type).
SCZ_SETS = {'schema': 'schema', 'gwaspri': 'gwas'}
SCGS = ROOT/'input/cilium_genes/SCGSv2_supplementary_material.xlsx'
WP4803 = ROOT/'input/cilium_genes/WP4803/WP4803.gpml'
HPA = ROOT/'input/annotations/proteinatlas.tsv.zip'
GPML = '{http://pathvisio.org/GPML/2013a}'
# HGNC renames of SCGSv2 symbols that neither PCNet nor the HPA snapshot resolve.
HGNC_RENAMES = {'TTC26': 'IFT56', 'TTC30A': 'IFT70A', 'TTC30B': 'IFT70B', 'HSPB11': 'IFT25', 'FAM92B': 'CIBAR2'}


def scgs_first_order(path=SCGS):
    """SCGSv2 rows marked 'x' in the First order column."""
    table = pd.read_excel(path, sheet_name='SCGSv2', dtype=str)
    table = table[table['Ensembl ID'].str.match(r'ENSG\d+$', na=False)]
    table = table[table['First order'].str.strip().str.lower().eq('x')]
    return pd.DataFrame({'ensembl': table['Ensembl ID'].str.strip(),
                         'source_symbol': table['Gene Name'].str.strip()}).drop_duplicates()


def wp_gene_products(path=WP4803):
    """Human gene products in a GPML pathway, with their Ensembl Xrefs and labels."""
    rows = []
    for node in ET.parse(path).getroot().iter(f'{GPML}DataNode'):
        xref = node.find(f'{GPML}Xref')
        if node.get('Type') == 'GeneProduct' and xref is not None and xref.get('Database') == 'Ensembl':
            rows.append({'ensembl': xref.get('ID').strip(), 'source_symbol': node.get('TextLabel', '').strip()})
    if not rows:
        raise ValueError(f'No Ensembl gene products in {path}')
    return pd.DataFrame(rows).drop_duplicates()


def map_to_network(genes, symbols):
    """Map Ensembl genes to network Entrez IDs by exact, unique symbols."""
    network = pd.DataFrame({'entrez': list(symbols), 'symbol': list(symbols.values())})
    network = network[network.groupby('symbol').entrez.transform('nunique') == 1]
    by_symbol = dict(zip(network.symbol, network.entrez))
    hpa = pd.read_csv(HPA, sep='\t', usecols=['Gene', 'Ensembl']).dropna().drop_duplicates()
    hpa = hpa[hpa.groupby('Ensembl').Gene.transform('nunique') == 1]
    hpa_symbol = dict(zip(hpa.Ensembl, hpa.Gene))
    rows = []
    for gene in genes.itertuples(index=False):
        entrez, how = by_symbol.get(gene.source_symbol), 'source_symbol'
        if entrez is None and gene.ensembl in hpa_symbol:
            entrez, how = by_symbol.get(hpa_symbol[gene.ensembl]), 'hpa_symbol'
        if entrez is None and gene.source_symbol in HGNC_RENAMES:
            entrez, how = by_symbol.get(HGNC_RENAMES[gene.source_symbol]), 'hgnc_rename'
        rows.append({'ensembl': gene.ensembl, 'source_symbol': gene.source_symbol,
                     'entrez': entrez, 'network_symbol': symbols.get(entrez, ''),
                     'mapping': how if entrez is not None else 'unmapped'})
    return pd.DataFrame(rows)


def write_table(frame, path):
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', default='PCNET2.2_brain', choices=NETWORKS)
    parser.add_argument('--publication-config', type=Path, default=ROOT/'configs/publication.json')
    parser.add_argument('--sets', nargs='+', choices=SETS, default=list(SETS),
                        help='Cilia seed sets to test; each set is an independent run and summary')
    args = parser.parse_args()
    settings = json.loads(args.publication_config.read_text())
    spec, root, inputs, provenance = load_network_inputs(ROOT/settings['disorder_config'], args.network,
                                                         min_seeds=settings['min_seeds'])
    config = AnalysisConfig.from_dict(spec['analysis'])
    A, nodes, symbols, entities, coords, regions = inputs
    cnvs = {n: e for n, e in entities.items() if e['kind'] == 'cnv'}
    missing = set(SCZ_SETS) - entities.keys()
    if missing:
        raise ValueError(f'SCZ gene sets missing from the disorder inputs: {sorted(missing)}')
    queries = {**cnvs, **{n: {**entities[n], 'kind': 'cnv'} for n in SCZ_SETS}}
    query_type = {n: SCZ_SETS.get(n, 'cnv') for n in queries}
    out_root = results_dir(configured_network_root(args.network, ROOT/'results'), 'cilia_colocalisation')
    loaders = {'scgsv2_first_order': (scgs_first_order, SCGS), 'wp4803_ciliopathies': (wp_gene_products, WP4803)}
    for name in args.sets:
        load, source = loaders[name]
        genes = load(source)
        summary = []
        mapping = map_to_network(genes, symbols)
        seeds = sorted({int(g) for g in mapping.entrez.dropna()})
        out_root.mkdir(parents=True, exist_ok=True)

        print(f'{name}: {len(genes)} source genes, {len(seeds)} mapped to network Entrez IDs', flush=True)
        run_path = run(A, nodes, symbols, {**queries, name: {'kind': 'target', 'seeds': seeds}}, coords, regions,
                       config=config, output_root=out_root/name/'pair_tests', cache_dir=root/spec['cache_dir'],
                       provenance={**provenance, 'question_family': name,
                                   'cilia_source': {'path': str(source.resolve()), 'sha256': file_hash(source)},
                                   'hpa_mapping': {'path': str(HPA.relative_to(ROOT)), 'sha256': file_hash(HPA)},
                                   'mapping_policy': 'exact unique network symbol; source symbol, else HPA symbol of Ensembl ID, else HGNC rename',
                                   'hgnc_renames': HGNC_RENAMES,
                                   'scz_query_sets': f'{" and ".join(SCZ_SETS)} relabelled kind cnv to be paired with the cilia target'})
        tests = pd.read_csv(resolve_path(run_path/'pair_tests.tsv'), sep='\t')
        tests = tests[(tests['null'] == 'netcoloc') & (tests.statistic == 'size')]
        qc = pd.read_csv(resolve_path(run_path/'seed_qc.tsv'), sep='\t').set_index('entity')
        overlap = {q: len(set(e['seeds']) & set(seeds)) for q, e in queries.items()}
        candidates = pd.read_csv(resolve_path(run_path/'candidate_genes.tsv'), sep='\t')
        for row in tests.itertuples(index=False):
            summary.append({'cilia_set': name, 'query': row.a, 'query_type': query_type[row.a],
                            'query_seeds': int(qc.loc[row.a, 'retained_seeds']),
                            'cilia_seeds': int(qc.loc[name, 'retained_seeds']), 'shared_seeds': overlap[row.a],
                            'observed': row.observed, 'expected': row.null_mean, 'size_ratio': row.size_ratio,
                            'z': row.z, 'p': row.p, 'q': row.q, 'significant': row.q < config.fdr,
                            'candidate_genes': int(((candidates.a == row.a) & (candidates.b == name)).sum()),
                            'run': str(run_path.resolve())})
        summary = pd.DataFrame(summary).sort_values(['q', 'z'], ascending=[True, False])
        from cnv_overlap.provenance import run_dir
        from cnv_overlap.results import finalize_manifest, verify_run
        identity = {'network': args.network, 'cilia_set': name, 'pair_run': str(run_path.resolve()),
                    'pair_manifest': file_hash(run_path/'manifest.json'), 'code': file_hash(__file__)}
        bundle = run_dir(out_root/name, identity, name)
        if (bundle/'manifest.json').exists():
            verify_run(bundle)
        else:
            bundle.mkdir(parents=True, exist_ok=True)
            write_table(mapping, bundle/'seed_mapping.tsv')
            write_table(summary, bundle/'summary.tsv')
            finalize_manifest(bundle/'manifest.json', {'status': 'complete', 'identity': identity,
                'sources': {'pair_tests': {'path': str(run_path.resolve()), 'manifest_sha256': file_hash(run_path/'manifest.json')}}})
        print(summary.drop(columns='run').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
