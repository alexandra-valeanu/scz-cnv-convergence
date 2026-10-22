"""NetColoc label permutations with CARVA normal-tail inference and empirical diagnostics.

The local array implementation preserves aligned shared blocks and deterministic RNG.
"""
from dataclasses import dataclass
import hashlib
import itertools

import numpy as np
from netcoloc.network_colocalization import calculate_network_overlap, get_p_from_permutation_results
from scipy.stats import beta


@dataclass(frozen=True)
class Thresholds:
    """Individual NPS and joint-product cutoffs for candidate selection."""
    individual: float = 1.0
    product: float = 3.0

    def __post_init__(self):
        """Require finite, nonnegative individual and positive product cutoffs."""
        if not np.isfinite([self.individual, self.product]).all() or self.individual < 0 or self.product <= 0:
            raise ValueError("Thresholds must be finite, individual >= 0, product > 0")


def vectors(a, b):
    """Return aligned finite 1D float arrays, rejecting empty or invalid inputs."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Scores must be nonempty, aligned, finite 1D vectors")
    return a, b


def hit_mask(a, b, thresholds=Thresholds()):
    """Return the Boolean mask of genes meeting NetColoc overlap thresholds."""
    a, b = vectors(a, b)
    hits = calculate_network_overlap(a, b, z_score_threshold=thresholds.product,
        z1_threshold=thresholds.individual, z2_threshold=thresholds.individual)
    mask = np.zeros(len(a), dtype=bool)
    mask[hits] = True
    return mask


def _fast_hit_mask(a, b, thresholds):
    """Array-only equivalent of NetColoc selection for Monte Carlo inner loops."""
    return (a > thresholds.individual) & (b > thresholds.individual) & (a*b >= thresholds.product)


def scores(a, b, thresholds=Thresholds()):
    """Return candidate count, mean positive product, and mean signed product."""
    a, b = vectors(a, b)
    return {"size": int(_fast_hit_mask(a, b, thresholds).sum()),
            "positive_mean": float((np.maximum(a, 0)*np.maximum(b, 0)).mean()),
            "signed_mean": float((a*b).mean())}


def bh(p):
    """Return BH-adjusted p-values, retaining NaNs in the planned family size."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or np.isinf(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("p must be a vector of probabilities or NaN")
    out = np.full(len(p), np.nan)
    keep = np.flatnonzero(np.isfinite(p))
    if not len(keep):
        return out
    order = keep[np.argsort(p[keep], kind="stable")]
    # Missing hypotheses remain in the prespecified family, conservatively.
    q = p[order] * len(p) / np.arange(1, len(order)+1)
    out[order] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return out


def tail(observed, null):
    """Summarize an upper-tail null with a plus-one p-value and 95% tail interval."""
    null = np.asarray(null, dtype=float)
    if null.ndim != 1 or len(null) < 2 or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError("At least two finite null draws and a finite observation are required")
    n = len(null)
    k = int(np.count_nonzero(null >= observed - 1e-12*max(1, abs(observed))))
    sd = float(null.std())
    return {"observed": float(observed), "null_mean": float(null.mean()), "null_sd": sd,
            "excess": float(observed-null.mean()),
            "size_ratio": float(observed/null.mean()) if null.mean() > 0 else np.nan, "p": (k+1)/(n+1), "n_ge": k, "n_draws": n,
            "p_resolution": 1/(n+1),
            "tail_ci_low": 0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            "tail_ci_high": 1.0 if k == n else float(beta.ppf(.975, k+1, n-k)),
            "z": float((observed-null.mean())/sd) if sd > 0 else np.nan,
            "status": "ok" if sd > 0 else "constant_null"}


def netcoloc_tail(observed, null):
    """CARVA/NetColoc one-sided normal tail; retain finite Monte Carlo diagnostics.

    A zero-variance permutation distribution cannot support a normal fit. Such
    hypotheses remain untested (NaN p/q), counted in the planned BH family.
    """
    result = tail(observed, null)
    result["p_empirical"] = result.pop("p")
    result["p"] = (float(get_p_from_permutation_results(observed, null, alternative="greater"))
                   if result["null_sd"] > 0 else np.nan)
    result["p_method"] = "normal_permutation"
    return result


def rng_for(seed, *identity):
    """Create a deterministic NumPy generator from a seed and analysis identity."""
    key = hashlib.sha256(repr((int(seed), identity)).encode()).digest()
    return np.random.default_rng(np.frombuffer(key, dtype="<u4"))


def label_null(a, b, nodes, seeds_a=(), seeds_b=(), *, reps=9999, seed=42,
               overlap="bin", thresholds=Thresholds()):
    """Return observed statistics and null draws from gene-label permutations.

    With overlap="bin", permute shared seeds separately from other nodes.
    Use "remove" to exclude shared seeds or "none" to permute all nodes.
    """
    a, b = vectors(a, b)
    nodes = np.asarray(nodes)
    if len(nodes) != len(a) or len(set(nodes.tolist())) != len(nodes):
        raise ValueError("Node IDs must be unique and aligned")
    if overlap not in ("bin", "remove", "none") or reps < 2:
        raise ValueError("Invalid overlap policy or number of repetitions")
    order = np.argsort(nodes)
    nodes, a, b = nodes[order], a[order], b[order]
    shared = np.isin(nodes, list(set(seeds_a) & set(seeds_b)))
    if overlap == "remove":
        nodes, a, b = nodes[~shared], a[~shared], b[~shared]
        shared = np.zeros(len(nodes), dtype=bool)
        vectors(a, b)
    # Canonical orientation makes seeded Monte Carlo results identical for A,B / B,A.
    if a.tobytes() > b.tobytes():
        a, b = b, a
    groups = [np.flatnonzero(shared), np.flatnonzero(~shared)] if overlap == "bin" else [np.arange(len(a))]
    observed = scores(a, b, thresholds)
    null = {s: np.empty(reps) for s in observed}
    rng = rng_for(seed, "label")
    perm = a.copy()
    for r in range(reps):
        for group in groups:
            perm[group] = a[rng.permutation(group)]
        for s, value in scores(perm, b, thresholds).items():
            null[s][r] = value
    return observed, null


def block_label_test(matrix, groups, *, exchangeable=False):
    """Enumerate region-label assignments and return a two-sided contrast test.

    Require explicit exchangeability and at least two regions in each group.
    Return the test summary and the complete null contrast array.
    """
    if not exchangeable:
        raise ValueError("Region-label exchangeability must be explicitly justified")
    matrix = np.asarray(matrix, dtype=float)
    groups = np.asarray(groups, dtype=bool)
    if matrix.shape != (len(groups), len(groups)) or not 2 <= groups.sum() <= len(groups)-2:
        raise ValueError("Need a square region matrix and at least two regions per group")
    i, j = np.triu_indices(len(groups), 1)
    values = matrix[i, j]
    if not np.isfinite(values).all() or not np.allclose(matrix, matrix.T, equal_nan=True):
        raise ValueError("Region scores must be finite and symmetric off diagonal")
    def contrast(g):
        """Compare within-selected-group scores with all remaining pair scores."""
        within = g[i] & g[j]
        return values[within].mean()-values[~within].mean()
    obs = contrast(groups)
    null = []
    for selected in itertools.combinations(range(len(groups)), int(groups.sum())):
        g = np.zeros(len(groups), dtype=bool)
        g[list(selected)] = True
        null.append(contrast(g))
    null = np.asarray(null)
    # Complete enumeration includes the observed assignment: no plus-one needed.
    return {"observed": float(obs), "p_two_sided": float(np.mean(np.abs(null) >= abs(obs)-1e-12)),
            "assignments": len(null)}, null
