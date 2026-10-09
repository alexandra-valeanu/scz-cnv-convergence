import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scz_restricted_core_null import sampling_plan, sample_joint, topology, run_restricted
from cnv_overlap.propagation import Propagator
from test_scz_core import scz_fixture
from test_workflow import settings


def test_restricted_joint_matching_and_overlap(tmp_path):
    A, nodes, _, entities, *_ = scz_fixture()
    prop = Propagator(A, nodes, tmp_path, settings().propagation)
    seeds = {'a': [3, 4, 5], 'b': [4, 5, 6], 'schema': [3, 20], 'gwaspri': [30, 31]}
    groups, _ = sampling_plan(prop, seeds, ['a', 'b'])
    excluded = set().union(*map(set, seeds.values()))
    for rep in range(30):
        draws = sample_joint(prop, seeds, ['a', 'b'], groups, np.random.default_rng(rep))
        assert not excluded & set(draws['a'] + draws['b'])
        assert len(set(draws['a']) & set(draws['b'])) == 2
        for n in draws:
            assert prop.signature(prop.seed_indices(draws[n])) == prop.signature(prop.seed_indices(seeds[n]))


def test_infeasible_pool_fails(tmp_path):
    A, nodes, *_ = scz_fixture()
    prop = Propagator(A, nodes, tmp_path, settings().propagation)
    with pytest.raises(ValueError, match='infeasible'):
        sampling_plan(prop, {'a': [int(nodes[0])], 'schema': list(map(int, nodes))}, ['a'])


def test_topology_empty_isolates_and_edges():
    A = sp.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert topology(A, np.array([False]*3)) == (0, 0, 0)
    assert topology(A, np.array([True]*3)) == (3, 2, 1)
    assert topology(A, np.array([True, False, True])) == (2, 1, 0)


def test_full_run_retains_real_overlap_and_reuses(tmp_path):
    inputs = scz_fixture()
    config = settings()
    kwargs = dict(config=config, cache_dir=tmp_path/'cache', output_root=tmp_path/'results',
                  reps=4, thresholds=(1, 2), progress=lambda _: None)
    out = run_restricted(inputs, **kwargs)
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for n, e in inputs[3].items():
        assert manifest['identity']['seeds'][n] == sorted(e['seeds'])
    excluded = set(pd.read_csv(out/'excluded_random_seeds.tsv', sep='\t').entrez)
    with np.load(out/'draws-000000.npz') as data:
        assert not excluded & set(data['seeds'].ravel())
        # Independently reconstruct the first saved draw via full propagation.
        from scz_restricted_core_null import core_counts
        from cnv_overlap.statistics import hit_mask
        prop = Propagator(inputs[0], inputs[1], tmp_path/'cache', config.propagation)
        names = sorted(n for n, e in inputs[3].items() if e['kind'] == 'cnv')
        target_profiles = {t: prop.nps(inputs[3][t]['seeds']) for t in ('schema', 'gwaspri')}
        landscape = hit_mask(target_profiles['schema'], target_profiles['gwaspri'], config.thresholds)
        offset = 0
        for n in names:
            size = len(inputs[3][n]['seeds'])
            target_profiles[n] = prop.nps(data['seeds'][0, offset:offset + size])
            offset += size
        counts = core_counts(target_profiles, names, config.thresholds, landscape)
        saved_masks = np.unpackbits(data['masks'][0], axis=-1, count=int(landscape.sum()))
        np.testing.assert_array_equal(saved_masks, [counts >= 1, counts >= 2])
    results = pd.read_csv(out/'network_tests.tsv', sep='\t')
    assert len(results) == 6
    assert results.p_empirical.between(1/5, 1).all()
    assert (results.q_bh >= results.p_empirical - 1e-12).all()
    assert run_restricted(inputs, **kwargs) == out


def test_selected_kernel_rows_equal_full_propagation(tmp_path):
    from scz_restricted_core_null import landscape_heat
    A, nodes, *_ = scz_fixture()
    prop = Propagator(A, nodes, tmp_path, settings().propagation)
    selected = np.arange(len(nodes)) % 3 == 0
    seeds = [3, 4, 20]
    actual = landscape_heat(np.asarray(prop.H[selected]), prop.seed_indices(seeds))
    np.testing.assert_array_equal(actual, prop.heat(seeds)[selected])
