from cnv_overlap.results import resolve_path
from dataclasses import replace
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pandas as pd
import pytest

from cnv_overlap.pipeline import AnalysisConfig, pair_plan, run
from test_workflow import fixture, settings


def test_comparison_plans_exclude_unrequested_pairs():
    entities = fixture()[3] | {'control': {'kind': 'comparator', 'seeds': [9, 10]}}
    assert pair_plan(entities, 'cnv_cnv') == [('cnv_a', 'cnv_b', 'region_pair')]
    disorder = pair_plan(entities, 'cnv_disorder')
    assert len(disorder) == 4
    assert all(family == 'locus_target' and b.startswith('target') for a, b, family in disorder)


@pytest.mark.parametrize('mode,count', [('cnv_cnv', 1), ('cnv_disorder', 4)])
def test_scoped_runs_only_compute_requested_family(tmp_path, mode, count):
    config = replace(settings(), comparisons=mode)
    out = run(*fixture(), config=config, output_root=tmp_path/'results', cache_dir=tmp_path/'cache', progress=lambda _: None)
    hypotheses = pd.read_csv(resolve_path(out/'hypotheses.tsv'), sep='\t')
    assert len(hypotheses) == count
    assert set(hypotheses.family) == {'region_pair' if mode == 'cnv_cnv' else 'locus_target'}
    qc = pd.read_csv(resolve_path(out/'seed_qc.tsv'), sep='\t')
    assert len(qc) == (2 if mode == 'cnv_cnv' else 4)


@pytest.mark.parametrize('name,mode', [('cnv_convergence', 'cnv_cnv'), ('cnv_disorder_overlap', 'cnv_disorder')])
def test_split_notebook_executes_on_synthetic_inputs(tmp_path, monkeypatch, name, mode):
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root/'tests/fixtures/analysis.json').read_text())
    spec['root'] = str(root)
    spec['output_root'] = str(tmp_path/'results'/name)
    spec['cache_dir'] = str(tmp_path/'cache')
    spec['analysis']['comparisons'] = mode
    path = tmp_path/'analysis.json'
    path.write_text(json.dumps(spec))
    monkeypatch.setenv('CNV_CONFIG', str(path))
    for variable in ('CNV_NETWORK', 'CNV_EXECUTION_DIR', 'CNV_RESULTS_ROOT'):
        monkeypatch.delenv(variable, raising=False)
    existing_reports = set((root/'results/executions').rglob('manifest.json'))
    nb = nbformat.read(root/'notebooks'/f'{name}.ipynb', as_version=4)
    # This fixture has two seeds per entity; production notebooks deliberately require three.
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace('MIN_SEEDS = 3', 'MIN_SEEDS = 2')
    NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(root)}}).execute()
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
    assert list((tmp_path/'results/executions/custom').glob(f'*/reports/{name}/manifest.json'))
    assert set((root/'results/executions').rglob('manifest.json')) == existing_reports


def test_notebooks_share_persistent_cnv_nps(tmp_path, monkeypatch):
    import numpy as np
    from cnv_overlap.propagation import Propagator

    root = Path(__file__).resolve().parents[1]
    configs = [json.loads((root/'configs'/f'{name}.json').read_text())
               for name in ('cnv_convergence', 'cnv_disorder_overlap')]
    assert configs[0]['cache_dir'] == configs[1]['cache_dir']
    assert configs[0]['analysis']['propagation'] == configs[1]['analysis']['propagation']
    for key in ('network_edges', 'network_nodes'):
        assert configs[0]['input_sha256'][key] == configs[1]['input_sha256'][key]
    seed_tables = [pd.read_csv(root/c['inputs']['seeds'], sep='\t') for c in configs]
    cnvs = [set(zip(t.loc[t.kind == 'cnv', 'entity'], t.loc[t.kind == 'cnv', 'entrez'])) for t in seed_tables]
    assert cnvs[0] == cnvs[1]

    A, nodes, _, entities, _, _ = fixture()
    p = Propagator(A, nodes, tmp_path, settings().propagation)
    saved = {name: np.array(p.nps(e['seeds'])) for name, e in entities.items() if e['kind'] == 'cnv'}
    reopened = Propagator(A, nodes, tmp_path, settings().propagation)

    def no_recalculation(*args):
        pytest.fail('The second notebook should read the saved CNV NPS without recomputing it')

    monkeypatch.setattr(reopened, 'reference', no_recalculation)
    monkeypatch.setattr(reopened, 'heat', no_recalculation)
    for name, expected in saved.items():
        np.testing.assert_array_equal(expected, reopened.nps(entities[name]['seeds']))
