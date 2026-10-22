from cnv_overlap.results import resolve_path
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from cnv_overlap.propagation import Propagator
from notebook_analysis import run_convergence
from scz_core_analysis import build_gsz, empirical_p, pathway_tests, run_target_pair
from scz_core_null import core_null, load_null
from test_workflow import fixture, settings


def scz_fixture():
    A, nodes, symbols, entities, coords, regions = fixture()
    entities = {k: v for k, v in entities.items() if v['kind'] == 'cnv'}
    entities |= {'schema': {'kind': 'target', 'seeds': [3, 4, 20]}, 'gwaspri': {'kind': 'target', 'seeds': [4, 30, 31]}}
    return A, nodes, symbols, entities, coords, regions


def test_degree_core_null_reproduces_run_convergence(tmp_path):
    A, nodes, symbols, entities, coords, regions = scz_fixture()
    config = settings()
    cnvs = {k: v for k, v in entities.items() if v['kind'] == 'cnv'}
    inputs = (A, nodes, symbols, cnvs, coords, regions)
    old = run_convergence(inputs, config=config, cache_dir=tmp_path/'cache', output_root=tmp_path/'old',
                          reps=30, progress=lambda _: None)
    counts = np.load(old/'null_counts.npy')
    prop = Propagator(A, nodes, tmp_path/'cache', config.propagation)
    gsz_idx = np.arange(0, len(prop.nodes), 3)
    seeds = {n: sorted(e['seeds']) for n, e in cnvs.items()}
    out = core_null(prop, seeds, gsz_idx, config=config, reps=30, null_model='degree',
                    output_root=tmp_path/'new', progress=lambda _: None)
    null = load_null(out)
    np.testing.assert_array_equal(null['reach'].sum(axis=1), counts[:, gsz_idx])
    # Own-seed masks mark exactly the sampled seeds that fall in G_SZ.
    assert null['own'].sum(axis=2).max() <= max(len(s) for s in seeds.values())
    # Completed runs are reused, not recomputed.
    assert core_null(prop, seeds, gsz_idx, config=config, reps=30, null_model='degree',
                     output_root=tmp_path/'new', progress=lambda _: None) == out


def test_degree_no_scz_never_samples_excluded_genes(tmp_path):
    A, nodes, symbols, entities, *_ = scz_fixture()
    config = settings()
    prop = Propagator(A, nodes, tmp_path/'cache', config.propagation)
    excluded = (3, 4, 20, 30, 31)
    gsz_idx = np.array([prop.pos[g] for g in excluded])
    seeds = {n: sorted(set(e['seeds']) - set(excluded)) for n, e in entities.items() if e['kind'] == 'cnv'}
    out = core_null(prop, seeds, gsz_idx, config=config, reps=20, null_model='degree_no_scz',
                    output_root=tmp_path, excluded=excluded, progress=lambda _: None)
    assert not load_null(out)['own'].any()
    with pytest.raises(ValueError, match='excludes'):
        core_null(prop, seeds, gsz_idx, config=config, reps=20, null_model='degree', output_root=tmp_path,
                  excluded=excluded, progress=lambda _: None)


def test_target_pair_candidates_define_gsz(tmp_path):
    A, nodes, symbols, entities, *_ = scz_fixture()
    config = settings()
    prop = Propagator(A, nodes, tmp_path/'cache', config.propagation)
    targets = {n: entities[n] for n in ('schema', 'gwaspri')}
    out = run_target_pair(prop, symbols, targets, config, output_root=tmp_path/'pair', provenance={}, fdr=.05)
    tests = pd.read_csv(resolve_path(out/'pair_tests.tsv'), sep='\t')
    assert set(tests.family) == {'target_pair'} and len(tests) == 3
    assert (tests.q == tests.p).all() or tests.p.isna().all()
    gsz_idx, zs, seeds = build_gsz(prop, targets, config, out)
    assert set(prop.nodes[gsz_idx]) == set(pd.read_csv(resolve_path(out/'candidate_genes.tsv'), sep='\t').entrez)
    assert run_target_pair(prop, symbols, targets, config, output_root=tmp_path/'pair', provenance={}, fdr=.05) == out


def test_pathway_statistics_match_direct_calculation():
    rng = np.random.default_rng(1)
    gsz = np.arange(100, 120)
    reach_null = rng.random((200, 4, 20)) < .3
    reach_obs = rng.random((4, 20)) < .5
    K_null, K_obs = reach_null.sum(axis=1), reach_obs.sum(axis=0)
    selected = pd.DataFrame({'database': ['R', 'R'], 'term_id': ['t1', 't2'], 'term_name': ['t1', 't2'],
                             'ontology': '', 'n_genes_in_gsz': [5, 8]})
    members = {('R', 't1'): list(gsz[:5]), ('R', 't2'): list(gsz[5:13])}
    out = pathway_tests(selected, members, gsz, {}, K_obs, K_null, reach_obs, reach_null, .05, chunk=37).set_index('term_id')
    inside = np.zeros(20, dtype=bool)
    inside[:5] = True
    T = K_null[:, inside].sum(axis=1)
    C = K_null[:, inside].mean(axis=1) - K_null[:, ~inside].mean(axis=1)
    U = reach_null[:, :, inside].any(axis=2).sum(axis=1)
    c_obs = K_obs[inside].mean() - K_obs[~inside].mean()
    assert out.loc['t1', 'self_contained_empirical_p'] == pytest.approx(empirical_p(T, K_obs[inside].sum()))
    assert out.loc['t1', 'competitive_empirical_p'] == pytest.approx(empirical_p(C, c_obs))
    assert out.loc['t1', 'independent_cnv_empirical_p'] == pytest.approx(empirical_p(U, reach_obs[:, inside].any(axis=1).sum()))


def test_seed_exclusion_only_removes_own_seed_credit():
    reach = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)
    own = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
    # Gene 0 is CNV 0's own seed: only that contribution goes, CNV 1 still counts.
    np.testing.assert_array_equal((reach & ~own).sum(axis=0), [1, 2, 2])
