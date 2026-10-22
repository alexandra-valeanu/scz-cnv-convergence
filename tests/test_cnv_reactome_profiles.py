"""Validate pathway scoring, joint null reconstruction, resume, and source integrity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cnv_overlap.propagation import Propagator
from cnv_overlap.results import verify_run
from cnv_overlap.statistics import hit_mask
from cnv_reactome_profiles import (pathway_universe, overlay_categories, score_profiles,
                                   pair_coverage, run_profiles)
from test_scz_core import scz_fixture
from test_workflow import settings


def test_scoring_threshold_denominator_and_outside():
    matrix = csr_matrix([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
    z = np.array([[1., 2., -1.], [3., 0., 2.]])
    landscape = np.array([True, False, True])
    scores = score_profiles(z, matrix, landscape, 1.)
    np.testing.assert_allclose(scores[0], [[.5, 0, 0], [.5, 1, 1]])
    np.testing.assert_allclose(scores[1], [[1.5, 0, 1], [1.5, 2, 3]])
    np.testing.assert_allclose(scores[2, :, 0], [1, 0])
    assert np.isnan(scores[2, :, 1:]).all()
    np.testing.assert_allclose(pair_coverage(scores[0]), [.25, 0, 0])


def test_overlay_is_exhaustive_including_both_below_product():
    actual = overlay_categories(np.array([2, 1.1, 2, 0, 1]),
                                np.array([2, 1.1, 0, 2, 1]),
                                np.array([True, False, False, False, False]), 1.)
    assert actual.tolist() == ['landscape', 'both_below_product', 'schema_only', 'gwas_only', 'neither']


def test_pathways_use_full_network_and_reject_duplicates(tmp_path):
    gmt = tmp_path/'terms.gmt'
    gmt.write_text('A\tR-A\tG1\tG1\tG2\tMISSING\nB\tR-B\tG3\n')
    universe, matrix = pathway_universe(gmt, ['G1', 'G2', 'G3'], 2)
    assert universe.network_genes.tolist() == [2, 1]
    assert universe.tested.tolist() == [True, False]
    np.testing.assert_array_equal(matrix.toarray(), [[1, 1, 0]])
    with pytest.raises(ValueError, match='unique'):
        pathway_universe(gmt, ['G1', 'G1'])
    gmt.write_text('A\tR-A\tG1\nB\tR-A\tG2\n')
    with pytest.raises(ValueError, match='duplicate'):
        pathway_universe(gmt, ['G1', 'G2'], 1)


def test_run_reconstructs_null_and_resumes(tmp_path):
    A, nodes, symbols, entities, *_ = scz_fixture()
    # Include shared CNV seeds to test the joint mapping, and target overlap.
    entities['cnv_b']['seeds'] = [2, 3]
    cfg = settings()
    prop = Propagator(A, nodes, tmp_path/'cache', cfg.propagation)
    profiles = {n: prop.nps(e['seeds']) for n, e in entities.items()}
    landscape = hit_mask(profiles['schema'], profiles['gwaspri'], cfg.thresholds)
    annotation = tmp_path/'terms.gmt'
    annotation.write_text('A\tR-A\tG1\tG2\tG3\tG4\nB\tR-B\tG20\tG30\tG31\n')
    kwargs = dict(config=cfg, annotation=annotation, output_root=tmp_path/'results',
                  reps=5, minimum=2, block_size=2, progress=lambda _: None)
    out = run_profiles(prop, profiles, entities, symbols, landscape, **kwargs)
    verify_run(out)
    names = ['cnv_a', 'cnv_b']
    _, matrix = pathway_universe(annotation, [symbols[g] for g in prop.nodes], 2)
    null = []
    checkpoints = sorted((out/'checkpoints').glob('*.npz'))
    for checkpoint in checkpoints:
        with np.load(checkpoint) as block:
            for flat in block['seeds']:
                draws = {n: flat[2*i:2*i+2] for i, n in enumerate(names)}
                assert len(set(draws['cnv_a']) & set(draws['cnv_b'])) == 1
                assert not set(flat) & {1, 2, 3}
                for n in names:
                    assert prop.signature(prop.seed_indices(draws[n])) == prop.signature(prop.seed_indices(entities[n]['seeds']))
                zz = np.stack([prop.nps(draws[n]) for n in names])
                null.append(score_profiles(zz, matrix, landscape, cfg.thresholds.individual))
    null = np.array(null)
    observed = score_profiles(np.stack([profiles[n] for n in names]), matrix, landscape, cfg.thresholds.individual)
    table = pd.read_csv(out/'cnv_pathway_tests.tsv', sep='\t')
    from cnv_overlap.statistics import bh
    for k, endpoint in enumerate(('coverage', 'mean_positive_nps', 'outside_coverage')):
        frame = table[table.endpoint == endpoint].set_index(['cnv', 'pathway'])
        valid = frame.p.notna()
        np.testing.assert_allclose(frame.loc[valid, 'q_bh'], bh(frame.loc[valid, 'p']))
        for i, n in enumerate(names):
            for j, pathway in enumerate(['R-A', 'R-B']):
                row = frame.loc[n, pathway]
                if np.isfinite(observed[k, i, j]):
                    assert row.p == pytest.approx((1 + (null[:, k, i, j] >= observed[k, i, j]-1e-12).sum())/6)
                    assert row.null_mean == pytest.approx(null[:, k, i, j].mean())
    pair_null = np.stack([pair_coverage(draw[0]) for draw in null])
    shared = pd.read_csv(out/'shared_pathway_tests.tsv', sep='\t')
    np.testing.assert_allclose(shared.null_mean, pair_null.mean(axis=0))
    np.testing.assert_allclose(shared.p, (1 + (pair_null >= pair_coverage(observed[0])-1e-12).sum(axis=0))/6)
    overlay = pd.read_csv(out/'scz_overlay.tsv', sep='\t')
    assert (overlay[['landscape', 'both_below_product', 'schema_only', 'gwas_only', 'neither']].sum(axis=1) == overlay.reached_genes).all()
    assert (out/'cnv_pathway_heatmap.png').stat().st_size > 1000
    assert (out/'scz_overlay.pdf').stat().st_size > 1000
    before = {p: p.read_bytes() for p in checkpoints}
    assert run_profiles(prop, profiles, entities, symbols, landscape, **kwargs) == out
    # Simulate interruption after the first checkpoint. Completed checkpoints are reused.
    manifest = json.loads((out/'manifest.json').read_text())
    manifest['status'] = 'running'
    (out/'manifest.json').write_text(json.dumps(manifest))
    checkpoints[-1].unlink()
    assert run_profiles(prop, profiles, entities, symbols, landscape, **kwargs) == out
    for p, data in before.items():
        # npz ZIP timestamps can differ; compare saved sufficient statistics.
        import io
        with np.load(p) as actual, np.load(io.BytesIO(data)) as expected:
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    verify_run(out)
    (out/'cnv_pathway_tests.tsv').write_text('corrupt')
    with pytest.raises(ValueError, match='changed'):
        run_profiles(prop, profiles, entities, symbols, landscape, **kwargs)


def test_pipeline_order_and_legacy_locations():
    from run_full_results import DEFAULT_STAGES
    from cnv_overlap.results import FOLDERS, LEGACY_FOLDERS
    assert 'scz_networks' not in DEFAULT_STAGES
    assert DEFAULT_STAGES.index('landscape') < DEFAULT_STAGES.index('cnv_reactome')
    assert FOLDERS['schizophrenia_landscape'].startswith('3_scz_landscape/')
    assert FOLDERS['cnv_reactome_profiles'] == '4_cnv_reactome_profiles'
    assert LEGACY_FOLDERS['schizophrenia_landscape'] == '4_scz_landscape'


def test_source_run_validation_and_cli_adapter(tmp_path, monkeypatch):
    import cnv_reactome_profiles as workflow
    from dataclasses import asdict
    from cnv_overlap.results import finalize_manifest
    A, nodes, symbols, entities, coords, regions = scz_fixture()
    cfg = settings()
    prop = Propagator(A, nodes, tmp_path/'cache', cfg.propagation)
    profiles = {n: prop.nps(e['seeds']) for n, e in entities.items()}
    landscape = hit_mask(profiles['schema'], profiles['gwaspri'], cfg.thresholds)
    source = tmp_path/'source'
    source.mkdir()
    frame = pd.DataFrame({'entrez': nodes, 'in_scz_landscape': landscape,
                          **{f'nps_{n}': z for n, z in profiles.items()}})
    frame.to_csv(source/'all_gene_membership.tsv', sep='\t', index=False)
    finalize_manifest(source/'manifest.json', {'status': 'complete', 'identity': {
        'settings': {'network': 'PCNET2.2_brain'}, 'config': asdict(cfg), 'entities': entities}})
    monkeypatch.setattr(workflow, 'load_inputs', lambda *args: (
        {'cache_dir': 'cache'}, tmp_path, (A, nodes, symbols, entities, coords, regions)))
    annotation = tmp_path/'terms.gmt'
    annotation.write_text('A\tR-A\tG1\tG2\tG3\n')
    out = workflow.run(source, reps=2, minimum=2, annotation=annotation, output_root=tmp_path/'output')
    manifest = verify_run(out)
    assert manifest['sources']['source']['path'] == str(source)
    entities['cnv_a']['seeds'] = [1, 3]
    with pytest.raises(ValueError, match='Entities differ'):
        workflow.run(source, reps=2, minimum=2, annotation=annotation, output_root=tmp_path/'output')


def test_default_notebooks_do_not_launch_legacy_shared_core():
    root = Path(__file__).resolve().parents[1]
    for name, stop in [('cnv_brain_convergence', 28), ('cnv_convergence', 18)]:
        notebook = json.loads((root/'notebooks'/f'{name}.ipynb').read_text())
        for cell in notebook['cells'][8:stop]:
            if cell['cell_type'] == 'code':
                assert ''.join(cell['source']).startswith('if os.environ.get("CNV_RUN_LEGACY_SHARED_CORE") == "1":')
                compile(''.join(cell['source']), name, 'exec')


def test_previous_landscape_named_selection_is_retained(tmp_path):
    from test_results_layout import pair_run
    from cnv_overlap.results import select_analysis
    old = tmp_path/'4_scz_landscape/landscape'
    pair_run(old, 1)
    selected = pair_run(old, 2)
    assert select_analysis(tmp_path, 'schizophrenia_landscape') == selected
