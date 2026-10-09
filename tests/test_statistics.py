import itertools

import numpy as np
import pytest

from cnv_overlap.statistics import Thresholds, bh, block_label_test, hit_mask, label_null, tail


def test_shared_seed_exact_null_and_orientation():
    a, b, nodes = np.array([1., 2., 10.]), np.array([3., 4., 20.]), np.array([101, 102, 103])
    obs, null = label_null(a, b, nodes, [103], [103], reps=99)
    assert obs["signed_mean"] == pytest.approx(211/3)
    assert set(np.round(null["signed_mean"], 8)) == {70., round(211/3, 8)}
    for aa, bb, nn in [(b, a, nodes), (a[::-1], b[::-1], nodes[::-1])]:
        obs2, null2 = label_null(aa, bb, nn, [103], [103], reps=99)
        assert obs2 == obs
        for key in null:
            np.testing.assert_array_equal(null[key], null2[key])


@pytest.mark.parametrize("shared", [[], [1, 2], [1, 2, 3, 4]])
def test_block_distribution_support(shared):
    a = np.array([1., -2., 3., 4.])
    b = np.array([5., 6., -7., 8.])
    nodes = np.arange(1, 5)
    groups = [np.flatnonzero(np.isin(nodes, shared)), np.flatnonzero(~np.isin(nodes, shared))]
    expected = set()
    for left, right in itertools.product(*[itertools.permutations(g) for g in groups]):
        x = a.copy()
        x[groups[0]], x[groups[1]] = a[list(left)], a[list(right)]
        expected.add(float(x@b/4))
    _, null = label_null(a, b, nodes, shared, shared, reps=200)
    assert set(null["signed_mean"]) <= expected


def test_bh_preserves_missing_and_family():
    np.testing.assert_allclose(bh([.01, np.nan, .2]), [.03, np.nan, .3], equal_nan=True)
    np.testing.assert_allclose(bh([.01, .04, .03, .002]), [.02, .04, .04, .008])
    with pytest.raises(ValueError):
        bh([-1])


def test_tails_ties_and_degenerate():
    assert tail(4, [1, 2, 3])["p"] == .25
    assert tail(2, [2, 2, 2])["p"] == 1
    assert tail(2, [2, 2, 2])["status"] == "constant_null"
    assert np.isnan(tail(2, [2, 2, 2])["z"])
    with pytest.raises(ValueError):
        tail(1, [np.nan, 2])


def test_selection_excludes_double_negatives_and_threshold_boundary():
    np.testing.assert_array_equal(hit_mask([-3, 1, 1.5, 2], [-3, 5, 2, 2]), [False, False, True, True])
    with pytest.raises(ValueError):
        Thresholds(product=-1)


def test_remove_all_shared_fails_explicitly():
    with pytest.raises(ValueError, match="nonempty"):
        label_null([1, 2], [2, 3], [1, 2], [1, 2], [1, 2], overlap="remove")


def test_block_requires_exchangeability():
    m = np.ones((4, 4))
    np.fill_diagonal(m, np.nan)
    with pytest.raises(ValueError, match="exchangeability"):
        block_label_test(m, [1, 1, 0, 0])
    stats, null = block_label_test(m, [1, 1, 0, 0], exchangeable=True)
    assert stats["assignments"] == 6 and stats["p_two_sided"] == 1
