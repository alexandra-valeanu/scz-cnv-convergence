from cnv_overlap.results import resolve_path
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from cnv_overlap.propagation import Propagator
from cnv_overlap.statistics import hit_mask, rng_for
from scz_landscape_pipeline import conditional_null, landscape_membership
from test_scz_core import scz_fixture
from test_workflow import settings


def test_landscape_requires_schema_gwas_mask_not_only_two_cnv_masks():
    # First gene satisfies both CNV pairs (3*1.1 >= 3), but not SCHEMA x GWAS.
    profiles = {'cnv': np.array([3., 3.]), 'schema': np.array([1.1, 2.]),
                'gwaspri': np.array([1.1, 2.])}
    entities = {n: {'kind': 'cnv' if n == 'cnv' else 'target', 'seeds': [1]} for n in profiles}
    table, _, _ = landscape_membership(np.array([1, 2]), {1: 'A', 2: 'B'}, entities,
                                        profiles, settings(), min_cnvs=1)
    assert table.shared_core_unrestricted.tolist() == [True, True]
    assert table.shared_core_in_landscape.tolist() == [False, True]


@pytest.mark.parametrize('core_reach', ['joint', 'direct'])
@pytest.mark.parametrize('preserve_target_overlap', [False, True])
def test_matched_seed_null_matches_full_propagation_and_tests_gene_identity(tmp_path, preserve_target_overlap, core_reach):
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
                     symbols, progress=lambda *args, **kwargs: None,
                     module_labels=[1] * int(landscape.sum()), min_module_genes=1,
                     preserve_target_overlap=preserve_target_overlap, core_reach=core_reach)
    actual = np.load(resolve_path(out/'draws.npz'))
    used = {n: seeds[n] for n in (*names, 'schema', 'gwaspri')}
    excluded = set().union(*map(set, used.values()))
    groups, _ = sampling_plan(prop, used, names)
    target_seeds = set(seeds['schema']) | set(seeds['gwaspri'])
    fixed = {n: set(seeds[n]) & target_seeds if preserve_target_overlap else set() for n in names}
    random_seeds = {n: sorted(set(seeds[n]) - fixed[n]) for n in names}
    if preserve_target_overlap:
        groups = [(original[~np.isin(original, list(target_seeds))], pool) for original, pool in groups]
    expected = []
    for rep in range(8):
        draws = sample_joint(prop, random_seeds, names, groups, rng_for(config.propagation.seed, 'landscape_matched_seeds', rep))
        assert not excluded & {g for n in names for g in draws[n]}
        draws = {n: sorted(set(draws[n]) | fixed[n]) for n in names}
        masks = []
        for n in names:
            mu, sd, _ = prop.reference(prop.seed_indices(seeds[n]))
            z = (np.log(prop.heat(draws[n]))-mu)/sd
            joint = hit_mask(z, profiles['schema'], config.thresholds) & hit_mask(z, profiles['gwaspri'], config.thresholds)
            masks.append((joint if core_reach == 'joint' else z > config.thresholds.individual)[landscape])
        expected.append(np.stack(masks).sum(axis=0))
    np.testing.assert_array_equal(actual['k_null'], expected)
    observed = both if core_reach == 'joint' else {n: profiles[n] > config.thresholds.individual for n in names}
    k_obs = np.stack([observed[n][landscape] for n in names]).sum(axis=0)
    cores_obs = np.array([k_obs >= 1, k_obs >= 2])
    shared = (np.asarray(expected)[:, None, :] >= np.array([1, 2])[None, :, None]) & cores_obs[None]
    np.testing.assert_array_equal(actual['overlap_with_observed'], shared.sum(axis=2))
    genes = pd.read_csv(resolve_path(out/'gene_identity_tests.tsv'), sep='\t').set_index('entrez')
    k_ge = (np.asarray(expected) >= k_obs).sum(axis=0)
    np.testing.assert_allclose(genes.loc[nodes[landscape], 'p_gene'], (1 + k_ge)/9)
    assert set(pd.read_csv(resolve_path(out/'core_size_tests.tsv'), sep='\t').min_cnvs) == {1, 2}
    assert set(pd.read_csv(resolve_path(out/'excluded_random_seeds.tsv'), sep='\t').entrez) == excluded
    identity = pd.read_csv(resolve_path(out/'core_gene_identity.tsv'), sep='\t')
    assert identity.observed_core_genes.tolist() == cores_obs.sum(axis=1).tolist()

    module_draws = np.load(resolve_path(out/f'module_draws_{core_reach}.npz'))
    # One module is the entire landscape: module counts must match coverage.
    np.testing.assert_array_equal(module_draws['counts'][:, :, 0], actual['cnv_coverage'])
    results = pd.read_csv(resolve_path(out/'module_tests.tsv'), sep='\t')
    assert set(results['mode']) == {'direct', 'joint'}
    assert set(results.endpoint) == {'pair_coverage', 'cnvs_reaching', 'coverage', 'landscape_share'}
