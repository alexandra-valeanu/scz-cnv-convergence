import numpy as np
import pandas as pd
import scipy.sparse as sp

from cnv_overlap.enrichment import ora
from cnv_overlap.rewiring import RewireConfig, chains, swap


def test_swap_and_chain_resume(tmp_path):
    n = 30
    rng = np.random.default_rng(7)
    a = np.triu((rng.random((n, n)) < .3).astype(float), 1)
    for i in range(n-1):
        a[i, i+1] = 1
    A = sp.csr_matrix(a+a.T)
    S, accepted = swap(A, 3000, np.random.default_rng(2))
    assert accepted > 0
    np.testing.assert_array_equal(A.sum(0), S.sum(0))
    assert S.nnz == A.nnz and not S.diagonal().any() and (S != S.T).nnz == 0
    config = RewireConfig(chains=2, steps=3, attempts_per_edge=.2)
    root, endpoints = chains(A, np.arange(n), tmp_path, config)
    # Delete a middle pass: regeneration must recreate the same continuation.
    (root/"chain-0-step-1.npz").unlink()
    _, resumed = chains(A, np.arange(n), tmp_path, config)
    for a, b in zip(endpoints, resumed):
        assert (a != b).nnz == 0
    assert (endpoints[0] != endpoints[1]).nnz > 0


def test_ora_empty_and_background():
    terms = {"t1": ("one", {"a", "b", "x"}), "t2": ("two", {"c", "d"})}
    result = ora(["a", "x"], ["a", "b", "c", "d"], terms, min_size=1)
    assert (result.query_size == 1).all() and (result.background_size == 4).all()
    assert (ora([], ["a", "b"], terms, min_size=1).q == 1).all()
