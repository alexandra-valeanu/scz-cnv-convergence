import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from cnv_cosine_similarity import cosine_matrix
from cnv_overlap.propagation import Propagator
from cnv_overlap.statistics import hit_mask, rng_for
from scz_landscape_pipeline import conditional_null, landscape_membership
from test_scz_core import scz_fixture
from test_workflow import settings


def test_cosine_uses_pair_union_and_actual_signed_scores():
    profiles = {'a': np.array([3., -2., 1000.]), 'b': np.array([-4., 5., 1000.]),
                'c': np.array([1., 1., 1.])}
    masks = {'a': [True, False, False], 'b': [False, True, False], 'c': [False, False, True]}
    matrix, pairs = cosine_matrix(profiles, masks)
    expected = -22 / (np.sqrt(13)*np.sqrt(41))
    assert matrix.loc['a', 'b'] == pytest.approx(expected)
    assert matrix.loc['b', 'a'] == pytest.approx(expected)
    assert pairs.query("a == 'a' and b == 'b'").union_genes.iloc[0] == 2
    assert np.diag(matrix).tolist() == pytest.approx([1, 1, 1])


def test_cosine_empty_and_zero_norm_are_undefined():
    result, _ = cosine_matrix({'a': np.zeros(2), 'b': np.ones(2)})
    assert np.isnan(result.loc['a', 'a']) and np.isnan(result.loc['a', 'b'])
    result, _ = cosine_matrix({'a': np.ones(2)}, {'a': [False, False]})
    assert np.isnan(result.iloc[0, 0])


def test_landscape_requires_schema_gwas_mask_not_only_two_cnv_masks():
    # First gene satisfies both CNV pairs (3*1.1 >= 3), but not SCHEMA x GWAS.
    profiles = {'cnv': np.array([3., 3.]), 'schema': np.array([1.1, 2.]),
                'gwaspri': np.array([1.1, 2.])}
    entities = {n: {'kind': 'cnv' if n == 'cnv' else 'target', 'seeds': [1]} for n in profiles}
    table, _, _ = landscape_membership(np.array([1, 2]), {1: 'A', 2: 'B'}, entities,
                                        profiles, settings(), min_cnvs=1)
    assert table.shared_core_unrestricted.tolist() == [True, True]
    assert table.shared_core_in_landscape.tolist() == [False, True]


def test_matched_seed_null_matches_full_propagation_and_tests_gene_identity(tmp_path):
    from scz_restricted_core_null import sample_joint, sampling_plan
    A, nodes, symbols, entities, *_ = scz_fixture()
    config = settings()
    prop = Propagator(A, nodes, tmp_path/'cache', config.propagation)
    nodes = prop.nodes
    seeds = {n: sorted(e['seeds']) for n, e in entities.items()}
    profiles = {n: np.asarray(prop.nps(s)) for n, s in seeds.items()}
    names = sorted(n for n in entities if entities[n]['kind'] == 'cnv')
    # A fixed sparse set exercises restriction independently of toy landscape size.
    landscape = np.arange(len(nodes)) % 3 == 0
    both = {n: hit_mask(profiles[n], profiles['schema'], config.thresholds)
               & hit_mask(profiles[n], profiles['gwaspri'], config.thresholds) for n in names}
    out = tmp_path/'null'
    conditional_null(prop, profiles, seeds, names, landscape, both, config, (1, 2), 8, out,
                     symbols, progress=lambda *args, **kwargs: None)
    actual = np.load(out/'draws.npz')
    used = {n: seeds[n] for n in (*names, 'schema', 'gwaspri')}
    excluded = set().union(*map(set, used.values()))
    groups, _ = sampling_plan(prop, used, names)
    expected = []
    for rep in range(8):
        draws = sample_joint(prop, used, names, groups, rng_for(config.propagation.seed, 'landscape_matched_seeds', rep))
        assert not excluded & {g for n in names for g in draws[n]}
        masks = []
        for n in names:
            mu, sd, _ = prop.reference(prop.seed_indices(seeds[n]))
            z = (np.log(prop.heat(draws[n]))-mu)/sd
            masks.append((hit_mask(z, profiles['schema'], config.thresholds)
                          & hit_mask(z, profiles['gwaspri'], config.thresholds))[landscape])
        expected.append(np.stack(masks).sum(axis=0))
    np.testing.assert_array_equal(actual['k_null'], expected)
    k_obs = np.stack([both[n][landscape] for n in names]).sum(axis=0)
    cores_obs = np.array([k_obs >= 1, k_obs >= 2])
    shared = (np.asarray(expected)[:, None, :] >= np.array([1, 2])[None, :, None]) & cores_obs[None]
    np.testing.assert_array_equal(actual['overlap_with_observed'], shared.sum(axis=2))
    genes = pd.read_csv(out/'gene_identity_tests.tsv', sep='\t').set_index('entrez')
    k_ge = (np.asarray(expected) >= k_obs).sum(axis=0)
    np.testing.assert_allclose(genes.loc[nodes[landscape], 'p_gene'], (1 + k_ge)/9)
    assert set(pd.read_csv(out/'core_size_tests.tsv', sep='\t').min_cnvs) == {1, 2}
    assert set(pd.read_csv(out/'excluded_random_seeds.tsv', sep='\t').entrez) == excluded
    identity = pd.read_csv(out/'core_gene_identity.tsv', sep='\t')
    assert identity.observed_core_genes.tolist() == cores_obs.sum(axis=1).tolist()
