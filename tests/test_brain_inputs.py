import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from brain_inputs import expressed_genes, map_expressed_genes
from network_inputs import load_network_inputs
from run_network_notebooks import resolve_workflow


def test_strict_tpm_any_region_and_unambiguous_mapping():
    regions = [f'region{i}' for i in range(13)]
    expression = pd.DataFrame([
        {'Gene': gene, 'Gene name': symbol, 'Brain region': region,
         'TPM': value if i == 12 else 0, 'nTPM': 99}
        for gene, symbol, value in [('E1', 'A', 1), ('E2', 'B', 1.1),
                                     ('E3', 'C', 2), ('E4', 'C', 3), ('E5', 'D', 2)]
        for i, region in enumerate(regions)])
    nodes = pd.DataFrame({'entrez': [1, 2, 3, 4, 5], 'symbol': ['A', 'B', 'C', 'D', 'D']})
    assert map_expressed_genes(expression, regions, nodes).entrez.tolist() == [2]
    with pytest.raises(ValueError, match='13 HPA'):
        expressed_genes(expression, regions[:-1])
    expression.loc[0, 'TPM'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        expressed_genes(expression, regions)


def test_filter_before_lcc_and_seed_qc(tmp_path, monkeypatch):
    # Original LCC is 1--2--3--4--5. Removing gene 3 leaves pairs;
    # the smaller original component 6--7--8 becomes the brain LCC.
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8)]
    matrix = np.zeros((8, 8))
    for a, b in edges:
        matrix[a-1, b-1] = matrix[b-1, a-1] = 1
    sp.save_npz(tmp_path/'adjacency.npz', sp.csr_matrix(matrix))
    pd.DataFrame({'entrez': range(1, 9), 'symbol': list('ABCDEFGH')}).to_csv(tmp_path/'nodes.tsv', sep='\t', index=False)
    pd.DataFrame({'entrez': range(1, 9), 'chrom': '1', 'start': range(1, 9), 'end': range(1, 9)}).to_csv(tmp_path/'coordinates.tsv', sep='\t', index=False)
    seeds = [('cnv', 'cnv', g) for g in (6, 7, 8)] + [('small', 'cnv', g) for g in (1, 2, 3)]
    seeds += [(name, 'target', g) for name in ('schema', 'gwaspri', 'other') for g in (6, 7, 8)]
    pd.DataFrame(seeds, columns=['entity', 'kind', 'entrez']).to_csv(tmp_path/'seeds.tsv', sep='\t', index=False)
    pd.DataFrame({'entity': ['cnv', 'small'], 'chrom': '1', 'start': 1, 'end': 8, 'assembly': 'GRCh37'}).to_csv(tmp_path/'regions.tsv', sep='\t', index=False)
    universe = tmp_path/'brain.tsv'
    pd.DataFrame({'entrez': [1, 2, 4, 5, 6, 7, 8]}).to_csv(universe, sep='\t', index=False)
    manifest = tmp_path/'brain.json'
    manifest.write_text('{}')
    monkeypatch.setattr('brain_inputs.prepare_brain_universe', lambda *args: (universe, manifest))
    spec = {'root': '.', 'inputs': {k: k+ext for k, ext in [('adjacency', '.npz'), ('nodes', '.tsv'), ('seeds', '.tsv'), ('coordinates', '.tsv'), ('regions', '.tsv')]},
            'output_root': 'results/cnv_disorder_overlap'}
    config = tmp_path/'config.json'
    config.write_text(json.dumps(spec))
    spec, _, inputs, provenance = load_network_inputs(config, 'PCNET2.2_brain')
    assert inputs[1].tolist() == [6, 7, 8]
    assert set(inputs[3]) == {'cnv', 'schema', 'gwaspri'}
    assert provenance['excluded_below_min_seeds'] == {'small': 0}
    assert 'brain_filter' in provenance['sha256']
    assert spec['cache_dir'] == 'data/cache/PCNET2.2_brain'
    assert spec['output_root'] == 'results/PCNET2.2_brain/cnv_disorder_overlap'
    # The same source config retains the original network and targets.
    _, _, original, _ = load_network_inputs(config, 'PCNET2.2', min_seeds=0)
    assert original[1].tolist() == [1, 2, 3, 4, 5]
    assert 'other' in original[3]


@pytest.mark.parametrize('network,notebook,expected', [
    ('PCNET2.2', 'cnv_convergence', ('PCNET2.2', 'cnv_convergence')),
    ('PCNET2.2_brain', 'cnv_disorder_overlap', ('PCNET2.2_brain', 'cnv_brain_disorder_overlap')),
    ('STRING_high_conf', 'cnv_brain_convergence', ('STRING_high_conf_brain', 'cnv_brain_convergence')),
])
def test_workflow_selection(network, notebook, expected):
    assert resolve_workflow(network, notebook) == expected
