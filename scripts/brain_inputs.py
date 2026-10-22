"""Frozen HPA TPM filter and conservative symbol-to-network-Entrez mapping."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cnv_overlap.provenance import atomic_path, file_hash, write_json

HPA_BASE = 'https://www.proteinatlas.org/download/tsv/'
EXPRESSION_FILE = 'rna_brain_region_hpa.tsv.zip'
REGIONS_FILE = 'rna_brain_region_hpa_brain_regions.tsv.zip'


def expressed_genes(expression, regions):
    """Return distinct HPA genes with TPM > 1 in any of the 13 validated regions."""
    required = {'Gene', 'Gene name', 'Brain region', 'TPM'}
    if not required <= set(expression):
        raise ValueError('HPA expression data require Gene, Gene name, Brain region and TPM')
    if len(regions) != 13 or expression['Brain region'].isna().any() or set(expression['Brain region']) != set(regions):
        raise ValueError('Expected expression measurements for exactly the 13 HPA brain regions')
    tpm = pd.to_numeric(expression.TPM, errors='raise')
    if not np.isfinite(tpm).all() or (tpm < 0).any():
        raise ValueError('HPA TPM must be finite and nonnegative')
    return expression.loc[tpm > 1, ['Gene', 'Gene name']].drop_duplicates()


def map_expressed_genes(expression, regions, network_nodes):
    """Exact symbols only; exclude ambiguous HPA and network assignments."""
    expressed = expressed_genes(expression, regions)
    hpa = expression[['Gene', 'Gene name']].dropna().drop_duplicates()
    hpa = hpa[hpa.groupby('Gene name').Gene.transform('nunique') == 1]
    table = network_nodes[['entrez', 'symbol']].dropna().drop_duplicates()
    table = table[table.groupby('symbol').entrez.transform('nunique') == 1]
    table = table[table.groupby('entrez').symbol.transform('nunique') == 1]
    return (expressed.merge(hpa, on=['Gene', 'Gene name'])
            .merge(table, left_on='Gene name', right_on='symbol')
            .sort_values(['entrez', 'Gene']).drop_duplicates('entrez'))


def prepare_brain_universe(root, spec, network):
    """Verify or download HPA inputs; write and return universe and filter-metadata paths."""
    from network_inputs import download_string_file

    root = Path(root)
    source = root/'input/expression/HPA_brain'
    manifest_path = source/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, checksum in manifest['artifacts'].items():
            if file_hash(source/name) != checksum:
                raise ValueError(f'HPA snapshot changed: {name}')
    else:
        for name in (EXPRESSION_FILE, REGIONS_FILE):
            download_string_file(HPA_BASE + name, source/name)
        # Validate before sealing the snapshot.
        expressed_genes(pd.read_csv(source/EXPRESSION_FILE, sep='\t'),
                        pd.read_csv(source/REGIONS_FILE, sep='\t')['Brain region'].tolist())
        manifest = {'source_urls': [HPA_BASE + n for n in (EXPRESSION_FILE, REGIONS_FILE)],
                    'release': 'unversioned HPA download; identity fixed by SHA256, not assumed v24',
                    'artifacts': {n: file_hash(source/n) for n in (EXPRESSION_FILE, REGIONS_FILE)}}
        write_json(manifest_path, manifest)
    expression = pd.read_csv(source/EXPRESSION_FILE, sep='\t')
    regions = pd.read_csv(source/REGIONS_FILE, sep='\t')['Brain region'].tolist()
    if 'adjacency' in spec['inputs']:
        table = pd.read_csv(root/spec['inputs']['nodes'], sep='\t')
    else:
        raw = json.loads((root/spec['inputs']['network_nodes']).read_text())
        table = pd.DataFrame({'entrez': [int(str(n['r']).split(':')[-1]) for n in raw],
                              'symbol': [n['n'] for n in raw]})
    mapped = map_expressed_genes(expression, regions, table)
    if 'gene_universe' in spec['inputs']:
        prior = pd.read_csv(root/spec['inputs']['gene_universe'], sep='\t')
        mapped = mapped[mapped.entrez.isin(prior.entrez)]
    if mapped.empty:
        raise ValueError('No HPA brain-expressed genes map to the selected network')
    directory = root/'data/network_configs'/network
    universe = directory/'brain_gene_universe.tsv'
    with atomic_path(universe) as temporary:
        mapped.to_csv(temporary, sep='\t', index=False)
    metadata = {**manifest, 'criterion': 'TPM > 1 in at least one of 13 brain regions',
                'regions': regions, 'mapping': 'exact unique HPA Gene name to unique network symbol; ambiguous/unmapped excluded',
                'hpa_expressed_genes': int(expressed_genes(expression, regions).Gene.nunique()),
                'network_nodes_before_filter': len(table), 'mapped_expressed_nodes': len(mapped),
                'graph_policy': 'induced graph, then largest connected component before propagation'}
    write_json(directory/'brain_filter.json', metadata)
    return universe, directory/'brain_filter.json'
