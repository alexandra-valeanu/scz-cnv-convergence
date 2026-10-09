"""Synthetic tests only: never open or analyse production results."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import scz_heterogeneous_analysis as analysis
from scz_heterogeneous_stats import (adjusted_p, binomial_interval, calibrate, calibration_p,
    count_histogram, inject_pathway_signal, jaccard, recurrence_p, term_scores)
from cnv_overlap.provenance import file_hash, write_json
from cnv_overlap.statistics import bh


def test_exact_scores_match_competitive_fractions_and_aggregate():
    rng = np.random.default_rng(4)
    reach = rng.random((7, 4, 24)) < .3
    members = np.zeros((3, 24), dtype=bool)
    members[0, :5], members[1, 5:16], members[2, 10:23] = True, True, True
    numerator, counts = term_scores(reach, members)
    sizes = members.sum(axis=1)
    score = numerator / (sizes * (24 - sizes))
    for t, inside in enumerate(members):
        np.testing.assert_array_equal(counts[..., t], reach[..., inside].sum(axis=-1))
        direct = reach[..., inside].mean(axis=-1) - reach[..., ~inside].mean(axis=-1)
        np.testing.assert_allclose(score[..., t], direct, atol=1e-15)
        K = reach.sum(axis=1)
        np.testing.assert_allclose(score[..., t].sum(axis=1),
                                   K[:, inside].mean(axis=1) - K[:, ~inside].mean(axis=1), atol=1e-15)


@pytest.mark.parametrize('alpha', [.00001, .05, .1, .5, .95])
def test_integer_support_cutoffs_exact_with_ties_and_positive_gate(alpha):
    rng = np.random.default_rng(20)
    calibration = rng.integers(-10, 11, (100, 3, 5), dtype=np.int32)
    ordered, cutoffs = calibrate(calibration, alpha)
    for value in range(-12, 14):
        scores = np.full((3, 5), value, dtype=np.int32)
        p, ge = calibration_p(ordered, scores)
        manual_ge = (calibration >= value).sum(axis=0)
        np.testing.assert_array_equal(ge, manual_ge)
        np.testing.assert_array_equal(p, (1 + manual_ge) / 101)
        np.testing.assert_array_equal(scores >= cutoffs, (p < alpha) & (scores > 0))


def test_constant_null_and_empirical_resolution():
    ordered, cutoffs = calibrate(np.zeros((99, 2, 3), dtype=np.int32), .05)
    np.testing.assert_array_equal(cutoffs, np.ones((2, 3)))
    p, _ = calibration_p(ordered, np.zeros((2, 3), dtype=np.int32))
    assert (p == 1).all()
    p, _ = calibration_p(ordered, np.ones((2, 3), dtype=np.int32))
    assert (p == .01).all()
    hist = count_histogram(np.zeros((80, 3), dtype=int), 2)
    p, ge = recurrence_p(np.array([0, 1, 2]), hist)
    np.testing.assert_array_equal(ge, [80, 0, 0])
    np.testing.assert_allclose(p, [1, 1 / 81, 1 / 81])


def test_recurrence_tails_and_bh_families():
    rng = np.random.default_rng(22)
    null = rng.integers(0, 5, (100, 6))
    observed = np.array([0, 1, 2, 3, 4, 2])
    p, ge = recurrence_p(observed, count_histogram(null, 4))
    np.testing.assert_array_equal(ge, (null >= observed).sum(axis=0))
    np.testing.assert_allclose(p, (1 + (null >= observed).sum(axis=0)) / 101)
    q, qall = adjusted_p(p, ['a', 'b', 'a', 'b', 'a', 'b'])
    np.testing.assert_allclose(q[::2], bh(p[::2]))
    np.testing.assert_allclose(q[1::2], bh(p[1::2]))
    np.testing.assert_allclose(qall, bh(p))
    lo, hi = binomial_interval([0, 100], 100)
    assert lo[0] == 0 and hi[1] == 1 and 0 < hi[0] < .05 and lo[1] > .95


def test_signal_preserves_total_reach_and_unaffected_cnvs():
    reach = np.zeros((4, 20), dtype=bool)
    reach[:, 10:18] = True
    inside = np.arange(20) < 5
    changed, swaps = inject_pathway_signal(reach, inside, [0, 2], .6, np.random.default_rng(42))
    assert swaps == 6
    np.testing.assert_array_equal(changed.sum(axis=1), reach.sum(axis=1))
    np.testing.assert_array_equal(changed[[1, 3]], reach[[1, 3]])
    assert changed[[0, 2], :5].sum() == 6
    assert not reach[:, :5].any()  # input was not mutated
    assert np.isnan(jaccard([], []))
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_evaluation_chunk_invariance_and_separation(tmp_path):
    rng = np.random.default_rng(24)
    masks = rng.random((140, 4, 24)) < .3
    packed = np.packbits(masks, axis=-1)
    members = np.zeros((3, 24), dtype=bool)
    members[0, :5], members[1, 4:10], members[2, 10:20] = True, True, True
    cfg = dict(calibration_draws=60, evaluation_draws=80, chunk_size=7, support_alpha=.05)
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    first = analysis.evaluate(packed, members, cfg, a)
    second = analysis.evaluate(packed, members, {**cfg, 'chunk_size': 23}, b)
    for x, y in zip(first, second):
        np.testing.assert_array_equal(x, y)
    scores, _ = term_scores(masks[60:], members)
    np.testing.assert_array_equal(first[2], (scores >= first[1]).sum(axis=1))
    # Changing only evaluation masks cannot move calibration cutoffs.
    altered = packed.copy()
    altered[60:] = 0
    third = analysis.evaluate(altered, members, cfg, b)
    np.testing.assert_array_equal(first[0], third[0])
    np.testing.assert_array_equal(first[1], third[1])
    assert not third[2].any()


def seal_fixture(directory, identity, **extra):
    files = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob('*')
             if p.is_file() and p.name != 'manifest.json'}
    write_json(directory / 'manifest.json', dict(status='complete', identity=identity, artifacts=files, **extra))


def synthetic_source(tmp_path, monkeypatch):
    """A complete tiny upstream artifact chain built entirely under tmp_path."""
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('scz_core_null.py', 'scz_heterogeneous_analysis.py', 'scz_heterogeneous_stats.py'):
        (scripts / name).write_bytes((SCRIPTS / name).read_bytes())
    source = tmp_path / 'results/Toy_brain/scz_core_gene_test/analysis-fixture'
    source.mkdir(parents=True)
    (source / 'degree').mkdir()
    null_dir = source.parent / 'null-fixture'
    null_dir.mkdir()
    annotation = tmp_path / 'input/annotations'
    annotation.mkdir(parents=True)
    genes = np.arange(1, 25)
    cnvs = ['c1', 'c2', 'c3', 'c4']
    seeds = {n: [i + 1] for i, n in enumerate(cnvs)}
    rng = np.random.default_rng(43)
    masks = rng.random((140, 4, 24)) < .3
    reach = rng.random((4, 24)) < .3
    reach[:2, :5] = True
    own = np.array([np.isin(genes, seeds[n]) for n in cnvs])
    np.save(null_dir / 'null_reach.npy', np.packbits(masks, axis=-1))
    np.save(null_dir / 'null_own_seed.npy', np.packbits(np.zeros_like(masks), axis=-1))
    np.savez(null_dir / 'observed.npz', reach=reach, own=own)
    write_json(null_dir / 'layout.json', dict(cnvs=cnvs, gsz_entrez=genes.tolist(), reps=140))
    prop = {'graph': 'synthetic-graph', 'config': {'alpha': .5}, 'software': {}}
    null_ident = dict(algorithm='scz_core_null_v1', null_model='degree', excluded=[], windows=None,
        code=file_hash(scripts / 'scz_core_null.py'), seeds=seeds, gsz=genes.tolist(), reps=140,
        propagation=prop, threshold=1., rng_seed=42)
    seal_fixture(null_dir, null_ident)
    gsz = pd.DataFrame(dict(entrez=genes, symbol=[f'G{g}' for g in genes], schema_nps=2., gwas_nps=2.))
    analysis.table(gsz, source / 'g_sz_genes.tsv')
    pair_dirs, pair_hashes = {}, {}
    for key in ('cnv_cnv', 'cnv_scz', 'schema_gwas'):
        directory = tmp_path / key
        directory.mkdir()
        analysis.table(gsz[['entrez']], directory / 'candidate_genes.tsv')
        identity = dict(graph=prop['graph'], config={'propagation': prop['config'],
                         'thresholds': {'individual': 1., 'product': 3.}}, seeds={'schema': [1], 'gwaspri': [2]})
        seal_fixture(directory, identity)
        pair_dirs[key], pair_hashes[key] = str(directory), file_hash(directory / 'manifest.json')
    observed = gsz.copy()
    for j, cnv in enumerate(cnvs):
        observed[f'reached_{cnv}'] = reach[j].astype(int)
    analysis.table(observed, source / 'degree/cnv_reach_matrix.tsv')
    memberships = np.zeros((4, 24), dtype=bool)
    memberships[0, :5] = True
    memberships[1, 4:10] = True
    memberships[2] = memberships[0]  # redundant term across databases
    memberships[3, 12:20] = True
    D, counts = term_scores(reach, memberships)
    n = memberships.sum(axis=1)
    scores = D / (n * (24 - n))
    rows = []
    for t in range(4):
        rows.append(dict(database='GO_BP' if t < 2 else 'Reactome', term_id=f't{t}', term_name=f'Term {t}',
            ontology='', n_genes_in_gsz=int(n[t]), observed_competitive_statistic=float(scores[:, t].sum()),
            competitive_empirical_p=.2, competitive_bh_q=.4,
            observed_independent_cnv_count=int((counts[:, t] > 0).sum()),
            independent_cnv_empirical_p=.3, independent_cnv_bh_q=.5,
            genes=','.join(f'G{g}({reach[:, g - 1].sum()})' for g in genes[memberships[t]])))
    terms = pd.DataFrame(rows)
    hashes, files = {}, {}
    for family, frame in terms.groupby('database'):
        filename = family + '.gmt'
        (annotation / filename).write_text('synthetic annotation fixture\n')
        hashes[family], files[family] = file_hash(annotation / filename), filename
        analysis.table(frame, source / f'degree/{family.lower()}_convergence.tsv')
    rule = dict(families=files, min_genes_in_gsz=5, max_genes_in_gsz=10, min_complement=10)
    for target in ('schema', 'gwas'):
        analysis.table(pd.DataFrame(dict(a=cnvs, p=.01, q=.04)), source / f'cnv_{target}_summary.tsv')
    analysis.table(pd.DataFrame({'a': ['c1'], 'b': ['c2'], 'p': [.01], 'q': [.04]}), source / 'cnv_cnv_summary.tsv')
    analysis.table(pd.DataFrame([dict(observed=1.5, null_mean=1.2, empirical_p=.2)]), source / 'degree/global_gsz_convergence.tsv')
    parent_ident = dict(algorithm='scz_core_analysis_v1', nulls={'degree': file_hash(null_dir / 'manifest.json')},
                       pairwise=pair_hashes, config={'pathways': rule}, annotations=hashes)
    seal_fixture(source, parent_ident, null_dirs={'degree': str(null_dir)}, pairwise_runs=pair_dirs,
                 g_sz_size=24, pathway_terms_tested={'GO_BP': 2, 'Reactome': 2})
    cfg = dict(network='Toy_brain', source_analysis=str(source), calibration_draws=60,
        evaluation_draws=80, support_alpha=.05, fdr=.05, borderline_margin=.01, chunk_size=17,
        diagnostics=dict(seed=42, pseudo_datasets=10, power_replicates=3, affected_cnvs=[2, 3, 4],
                         swap_fractions=[.5, 1.], target_selection='median_size_per_family'),
        reporting=dict(leading_terms_per_family=2, redundancy_gene_jaccard=.5))
    return source, cfg


def test_synthetic_end_to_end_and_cache_integrity(tmp_path, monkeypatch):
    source, cfg = synthetic_source(tmp_path, monkeypatch)
    config_path = tmp_path / 'config.json'
    write_json(config_path, cfg)
    out = analysis.main(['--config', str(config_path)])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['identity']['calibration_draw_range_half_open'] == [0, 60]
    assert manifest['identity']['evaluation_draw_range_half_open'] == [60, 140]
    recurrence = pd.read_csv(out / 'heterogeneous_term_recurrence.tsv', sep='\t')
    cells = pd.read_csv(out / 'cnv_term_calibrated_support.tsv', sep='\t')
    assert len(recurrence) == 4 and len(cells) == 16
    np.testing.assert_allclose(recurrence.heterogeneous_empirical_p,
                               (1 + recurrence.evaluation_exceedances) / 81)
    assert recurrence.heterogeneous_empirical_p.between(0, 1, inclusive='right').all()
    summed = cells.groupby('term_key').supports_term.sum()
    np.testing.assert_array_equal(summed.loc[recurrence.term_key], recurrence.observed_support_count)
    assert cells.schema_pairwise_q.eq(.04).all()
    design = json.loads((out / 'diagnostic_design.json').read_text())
    assert design['diagnostic_reference_draw_range_half_open'] == [60, 130]
    assert design['pseudo_draw_range_half_open'] == [130, 140]
    counts = np.load(out / 'evaluation_recurrence_counts.npy')
    pseudo = pd.read_csv(out / 'null_pseudo_datasets.tsv', sep='\t')
    # Pseudo-observations are excluded together, not scored against themselves.
    for i in range(10):
        p = (1 + (counts[:-10] >= counts[-10 + i]).sum(axis=0)) / 71
        all_row = pseudo[(pseudo.pseudo_draw == 130 + i) & (pseudo.database == 'ALL_TERMS')].iloc[0]
        assert all_row.minimum_p == pytest.approx(p.min())
        assert all_row.minimum_q == pytest.approx(bh(p).min())
    power = pd.read_csv(out / 'power_summary.tsv', sep='\t')
    assert len(power) == 12 and power.replicates.eq(3).all()
    assert len(pd.read_csv(out / 'mechanism_pair_comparisons.tsv', sep='\t')) == 6
    assert (out / 'cnv_pathway_profiles.png').stat().st_size > 1000
    assert analysis.main(['--config', str(config_path)]) == out
    (out / 'summary.md').write_text('tampered')
    with pytest.raises(ValueError, match='Artifact missing or changed'):
        analysis.main(['--config', str(config_path)])


def test_source_provenance_rejects_graph_mismatch(tmp_path, monkeypatch):
    source, cfg = synthetic_source(tmp_path, monkeypatch)
    ctx = analysis.load_inputs(source, cfg)
    assert ctx['membership'].shape == (4, 24)
    manifest_path = source / 'manifest.json'
    parent = json.loads(manifest_path.read_text())
    pair_dir = Path(parent['pairwise_runs']['schema_gwas'])
    pair = json.loads((pair_dir / 'manifest.json').read_text())
    pair['identity']['graph'] = 'different-graph'
    write_json(pair_dir / 'manifest.json', pair)
    parent['identity']['pairwise']['schema_gwas'] = file_hash(pair_dir / 'manifest.json')
    write_json(manifest_path, parent)
    with pytest.raises(ValueError, match='Graph, propagation or threshold mismatch'):
        analysis.load_inputs(source, cfg)


def test_source_provenance_rejects_reordered_layout(tmp_path, monkeypatch):
    source, cfg = synthetic_source(tmp_path, monkeypatch)
    parent = json.loads((source / 'manifest.json').read_text())
    null_dir = Path(parent['null_dirs']['degree'])
    null = json.loads((null_dir / 'manifest.json').read_text())
    layout = json.loads((null_dir / 'layout.json').read_text())
    layout['cnvs'] = layout['cnvs'][::-1]
    write_json(null_dir / 'layout.json', layout)
    null['artifacts']['layout.json'] = file_hash(null_dir / 'layout.json')
    write_json(null_dir / 'manifest.json', null)
    parent['identity']['nulls']['degree'] = file_hash(null_dir / 'manifest.json')
    write_json(source / 'manifest.json', parent)
    with pytest.raises(ValueError, match='layout does not match'):
        analysis.load_inputs(source, cfg)


def test_source_corruption_is_rejected(tmp_path, monkeypatch):
    source, cfg = synthetic_source(tmp_path, monkeypatch)
    (source / 'g_sz_genes.tsv').write_text('corrupted')
    with pytest.raises(ValueError, match='Artifact missing or changed'):
        analysis.load_inputs(source, cfg)


def test_display_groups_require_same_nonempty_supporters(tmp_path):
    # Identical annotations with different supporters must not collapse into one mechanism.
    result = pd.DataFrame(dict(term_key=['a', 'b', 'c'], database=['R'] * 3, term_id=['a', 'b', 'c'],
        term_name=['A', 'B', 'C'], heterogeneous_bh_q=[.01, .02, .03], heterogeneous_empirical_p=[.001] * 3,
        heterogeneous_all_terms_bh_q=[.01, .02, .03], observed_support_count=[2] * 3,
        supporting_cnvs=['c1,c2', 'c3,c4', 'c1,c2'], family_fdr_recurrent=True, all_terms_fdr_recurrent=True))
    support = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=bool)
    ctx = dict(membership=np.array([[1, 1, 0]] * 3, dtype=bool), cnvs=['c1', 'c2', 'c3', 'c4'],
        reach=np.ones((4, 3), dtype=bool), own=np.zeros((4, 3), dtype=bool),
        symbols=np.array(['G1', 'G2', 'G3']), genes=np.arange(3))
    cfg = {'reporting': {'leading_terms_per_family': 3, 'redundancy_gene_jaccard': .5}}
    _, profiles = analysis.mechanism_profiles(ctx, cfg, result, support, tmp_path)
    groups = profiles.set_index('term_key').display_group
    assert groups.a == groups.c and groups.a != groups.b
