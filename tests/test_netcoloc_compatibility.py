"""Parity checks against the installed, pinned upstream implementation."""
import numpy as np
import networkx as nx
import pytest
from netcoloc import netprop, netprop_zscore

from cnv_overlap import propagation
from cnv_overlap.propagation import PropagationConfig, Propagator, heat_kernel
from cnv_overlap.statistics import Thresholds, hit_mask, scores


@pytest.mark.parametrize('alpha', [.3, .5, .8])
def test_kernel_orientation_against_netcoloc(alpha):
    graph = nx.barabasi_albert_graph(35, 2, seed=7)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=list(graph), dtype=float)
    expected = netprop.get_individual_heats_matrix(graph, alpha=alpha)
    np.testing.assert_allclose(heat_kernel(adjacency, alpha, 'float64'), expected, atol=1e-13)
    np.testing.assert_allclose(heat_kernel(adjacency, alpha, 'float32'), expected, atol=1e-6)


def test_streaming_nps_against_upstream_with_identical_draws(tmp_path, monkeypatch):
    graph = nx.barabasi_albert_graph(35, 2, seed=7)
    nodes = list(graph)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=float)
    config = PropagationConfig(reference_reps=100, minimum_bin_size=10,
                               min_reference_count=20, dtype='float64')
    prop = Propagator(adjacency, nodes, tmp_path, config)
    seeds = [0, 8, 12]
    sampled = []
    original = netprop_zscore.network_propagation

    def record(kernel, nodes, seeds):
        sampled.append(list(seeds))
        return original(kernel, nodes, seeds)

    monkeypatch.setattr(netprop_zscore, 'network_propagation', record)
    state = np.random.get_state()
    try:
        z, heat, references = netprop_zscore.calculate_heat_zscores(
            prop.H, nodes, dict(graph.degree()), seeds, num_reps=100,
            minimum_bin_size=10, random_seed=42)
    finally:
        np.random.set_state(state)
    # Replay upstream's actual draws through our streaming reference accumulator.
    draws = iter([np.array([g for g in draw if g in group], dtype=int)
                  for draw in sampled[1:] for group in prop.bins
                  if any(g in group for g in seeds)])

    class Replay:
        def choice(self, group, size, replace):
            draw = next(draws)
            assert len(draw) == size and not replace and np.isin(draw, group).all()
            return draw

    monkeypatch.setattr(propagation, 'rng_for', lambda *args: Replay())
    mu, sd, count = prop.reference(prop.seed_indices(seeds))
    np.testing.assert_allclose(mu, np.nanmean(np.log(references), axis=0), atol=1e-13)
    np.testing.assert_allclose(sd, np.nanstd(np.log(references), axis=0), atol=1e-13)
    np.testing.assert_array_equal(count, np.isfinite(references).sum(axis=0))
    np.testing.assert_allclose(prop.heat(seeds), heat, atol=1e-14)
    np.testing.assert_allclose(prop.nps(seeds), z, atol=1e-12)


@pytest.mark.parametrize('thresholds', [Thresholds(), Thresholds(1.5, 4), Thresholds(0, 1)])
def test_fast_monte_carlo_selection_matches_package(thresholds):
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=(2, 10000)) * 3
    # Explicit strict individual and inclusive product boundaries, plus negatives.
    a[:5] = [thresholds.individual, -4, 2, 4, 0]
    b[:5] = [10, -4, thresholds.product / 2, -4, 10]
    assert scores(a, b, thresholds)['size'] == hit_mask(a, b, thresholds).sum()


@pytest.mark.parametrize('policy', ['bin', 'remove', 'none'])
@pytest.mark.parametrize('seeds', [([], []), ([0], [0]), ([0, 2], [0, 2, 3])])
def test_size_permutations_match_upstream(tmp_path, monkeypatch, policy, seeds):
    """Replay local draws through upstream, independent of its global RNG."""
    import pandas as pd
    from netcoloc import network_colocalization as upstream
    from cnv_overlap import statistics

    a = np.array([4., 2., 3., -2., 1.2, -1., 0., 2.5])
    b = np.array([3., 1.8, 2., 4., -3., -1., 2., 2.])
    # Canonical orientation is chosen after shared-seed removal.
    keep = ~np.isin(np.arange(len(a)), list(set(seeds[0]) & set(seeds[1]))) if policy == 'remove' else np.ones(len(a), dtype=bool)
    if a[keep].tobytes() > b[keep].tobytes():
        a, b = b, a
    permutations = []
    rng = np.random.default_rng(9)

    class Record:
        def permutation(self, group):
            order = rng.permutation(len(group))
            permutations.append(order)
            return group[order]

    monkeypatch.setattr(statistics, 'rng_for', lambda *args: Record())
    observed, null = statistics.label_null(a, b, np.arange(len(a)), *seeds, reps=30, overlap=policy)
    if policy == 'bin':
        # Upstream shuffles nonshared first; local implementation shared first.
        permutations = [x for i in range(0, len(permutations), 2)
                        for x in (permutations[i+1], permutations[i])]
    replay = iter(permutations)
    original = {}

    def shuffle(values):
        # Upstream shuffles arrays in place repeatedly; replay draws from originals.
        base = original.setdefault(id(values), values.copy())
        values[:] = base[next(replay)]

    monkeypatch.setattr(upstream.rn, 'shuffle', shuffle)
    size, draws = upstream.calculate_expected_overlap(
        pd.Series(a), pd.Series(b), seed1=seeds[0], seed2=seeds[1],
        overlap_control=None if policy == 'none' else policy,
        num_reps=30, z_score_threshold=3, z1_threshold=1, z2_threshold=1)
    assert size == observed['size']
    np.testing.assert_array_equal(draws, null['size'])


def test_normal_tail_matches_netcoloc_and_keeps_empirical_diagnostic():
    from netcoloc.network_colocalization import get_p_from_permutation_results
    from cnv_overlap.statistics import netcoloc_tail, bh
    null = np.array([0, 1, 2, 3, 4, 2, 1, 2, 3, 2])
    result = netcoloc_tail(5, null)
    assert result['p'] == get_p_from_permutation_results(5, null)
    assert result['p_empirical'] == 1 / 11
    constant = netcoloc_tail(5, np.zeros(10))
    assert np.isnan(constant['p']) and constant['status'] == 'constant_null'
    assert constant['p_empirical'] == 1 / 11
    np.testing.assert_allclose(bh([.01, constant['p']]), [.02, np.nan])
