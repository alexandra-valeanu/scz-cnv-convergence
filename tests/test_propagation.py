import numpy as np
import pytest
import scipy.sparse as sp

from cnv_overlap.propagation import PropagationConfig, Propagator, degree_bins, heat_kernel
from cnv_overlap.provenance import cached_array, graph_hash


def graph(n=18):
    rng = np.random.default_rng(1)
    A = np.triu((rng.random((n, n)) < .2).astype(float), 1)
    for i in range(n-1):
        A[i, i+1] = 1
    return sp.csr_matrix(A+A.T)


def test_heat_equation_and_float_precision():
    A = graph()
    h = heat_kernel(A, dtype="float64")
    W = A.toarray()/np.asarray(A.sum(0)).ravel()
    np.testing.assert_allclose((np.eye(len(W))-.5*W)@h, .5*np.eye(len(W)), atol=1e-12)
    np.testing.assert_allclose(h.sum(0), 1)
    np.testing.assert_allclose(heat_kernel(A), h, atol=1e-6)
    with pytest.raises(ValueError):
        heat_kernel(A, alpha=1)


def test_cache_invalidates_and_checks_corruption(tmp_path):
    A = graph()
    nodes = np.arange(len(A.toarray()))
    assert graph_hash(A, nodes) != graph_hash(A, nodes[::-1])
    calls = []
    build = lambda: calls.append(1) or np.arange(3.)
    cached_array(tmp_path, "x", {"alpha": .5}, build)
    cached_array(tmp_path, "x", {"alpha": .5}, build)
    assert len(calls) == 1
    next(tmp_path.glob("*.npy")).write_bytes(b"corrupted")
    cached_array(tmp_path, "x", {"alpha": .5}, build)
    cached_array(tmp_path, "x", {"alpha": .4}, build)
    assert len(calls) == 3


def test_profile_reference_unique_sampling_and_reproducibility(tmp_path):
    A = graph()
    c = PropagationConfig(reference_reps=100, minimum_bin_size=6, min_reference_count=20)
    p = Propagator(A, np.arange(18), tmp_path, c)
    z = np.asarray(p.nps([1, 3, 3]))
    np.testing.assert_array_equal(z, p.nps([3, 1]))
    p2 = Propagator(A[::-1, ::-1], np.arange(18)[::-1], tmp_path, c)
    np.testing.assert_array_equal(z, p2.nps([1, 3]))
    assert np.isfinite(z).all()
    with pytest.raises(ValueError):
        p.nps([])
    with pytest.raises(ValueError):
        p.nps([999])


def test_bins_never_split_equal_degrees():
    d = np.array([1, 1, 2, 2, 2, 3, 4])
    bins = degree_bins(d, 3)
    for degree in np.unique(d):
        assert sum(np.isin(np.flatnonzero(d == degree), group).any() for group in bins) == 1
