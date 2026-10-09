"""Array-only inference for calibrated CNV pathway recurrence.

Scores use the exact integer numerator D = N*S - n*r of the competitive
fraction difference C = D/[n*(N-n)]. This preserves ties without float
tolerances. A fixed term has the same positive denominator for all CNVs/draws.
No propagation, genomic sampling, or pathway selection occurs here.
"""
import numpy as np
from scipy import sparse
from scipy.stats import beta

from cnv_overlap.statistics import bh


def term_scores(reach, membership):
    """Return D and reached counts, preserving leading dimensions of reach."""
    reach = np.asarray(reach)
    membership = sparse.csr_matrix(membership, dtype=np.int32)
    n_terms, n_genes = membership.shape
    sizes = np.asarray(membership.sum(axis=1)).ravel()
    if (reach.shape[-1] != n_genes or not n_terms or
            np.any(sizes <= 0) or np.any(sizes >= n_genes)):
        raise ValueError('Terms must be nonempty proper subsets of the aligned gene universe')
    if n_genes**2 > np.iinfo(np.int32).max:
        raise ValueError('Gene universe exceeds the int32 score bound')
    if np.any((reach != 0) & (reach != 1)) or np.any(membership.data != 1):
        raise ValueError('Reach and membership must be binary')
    flat = reach.reshape(-1, n_genes).astype(np.int32)
    counts = (membership @ flat.T).T
    numerator = n_genes * counts - flat.sum(axis=1, dtype=np.int32)[:, None] * sizes
    shape = (*reach.shape[:-1], n_terms)
    return numerator.reshape(shape).astype(np.int32), counts.reshape(shape)


def calibrate(scores, alpha):
    """Sorted calibration distributions and inclusive integer support cutoffs.

cutoff is exactly equivalent to p_cal(c) < alpha AND c > 0, including ties.
No evaluation draws or observed data enter this function.
"""
    scores = np.asarray(scores)
    if scores.ndim != 3 or len(scores) < 2 or not 0 < alpha < 1:
        raise ValueError('Need draw x CNV x term calibration scores and 0 < alpha < 1')
    ordered = np.sort(scores, axis=0)
    B = len(ordered)
    allowed = np.flatnonzero((1 + np.arange(B + 1)) / (B + 1) < alpha)
    if not len(allowed):
        cutoffs = np.full(scores.shape[1:], np.iinfo(np.int32).max, dtype=np.int32)
    else:
        cutoffs = np.maximum(1, ordered[B - 1 - allowed[-1]] + 1)
    return ordered, cutoffs


def calibration_p(ordered, scores):
    """One-sided empirical P, with >= ties and +1, for CNV x term scores."""
    scores = np.asarray(scores)
    if scores.shape != ordered.shape[1:]:
        raise ValueError('Calibration and observed CNV/term shapes differ')
    ge = np.empty(scores.shape, dtype=np.int64)
    for j in range(scores.shape[0]):
        for t in range(scores.shape[1]):
            ge[j, t] = len(ordered) - np.searchsorted(ordered[:, j, t], scores[j, t], side='left')
    return (1 + ge) / (len(ordered) + 1), ge


def count_histogram(null_counts, n_cnvs):
    """Count distribution per term; sufficient for exact discrete recurrence tails."""
    null_counts = np.asarray(null_counts)
    if null_counts.ndim != 2 or not len(null_counts) or np.any(null_counts < 0) or np.any(null_counts > n_cnvs):
        raise ValueError('Invalid recurrence count array')
    return np.stack([(null_counts == k).sum(axis=0) for k in range(n_cnvs + 1)])


def recurrence_p(counts, histogram):
    """Empirical tail for one or many vectors of observed recurrence counts."""
    counts = np.asarray(counts)
    if counts.shape[-1] != histogram.shape[1] or np.any(counts < 0) or np.any(counts >= len(histogram)):
        raise ValueError('Observed counts do not match the recurrence histogram')
    tails = np.cumsum(histogram[::-1], axis=0)[::-1]
    ge = tails[counts, np.arange(histogram.shape[1])]
    return (1 + ge) / (1 + histogram.sum(axis=0)), ge


def adjusted_p(p, families):
    """BH across each annotation family and separately across all tested terms."""
    p = np.asarray(p)
    families = np.asarray(families)
    if p.shape != families.shape:
        raise ValueError('P values and family labels must align')
    q = np.empty(p.shape)
    for family in np.unique(families):
        mask = families == family
        q[mask] = bh(p[mask])
    return q, bh(p)


def binomial_interval(k, n, level=.95):
    """Clopper-Pearson interval for the underlying probability, not adjusted P/q."""
    k, n = np.broadcast_arrays(np.asarray(k), np.asarray(n))
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError('Need 0 <= exceedances <= positive draw count')
    a = (1 - level) / 2
    low = np.where(k == 0, 0., beta.ppf(a, np.maximum(k, 1), n - k + 1))
    high = np.where(k == n, 1., beta.ppf(1 - a, k + 1, np.maximum(n - k, 1)))
    return low, high


def inject_pathway_signal(reach, inside, affected, fraction, rng):
    """Synthetic reach-level alternative, preserving every CNV's total reach.

Swap a fraction of possible outside-reached/inside-unreached pairs for the
affected CNVs. This benchmarks the test; it is not a simulated genomic CNV
or a biological effect-size model. Unaffected CNVs are unchanged.
"""
    result = np.asarray(reach, dtype=bool).copy()
    inside = np.asarray(inside, dtype=bool)
    if not 0 <= fraction <= 1 or inside.shape != (result.shape[-1],):
        raise ValueError('Invalid signal fraction or term mask')
    swaps = 0
    for j in affected:
        gain = np.flatnonzero(inside & ~result[j])
        lose = np.flatnonzero(~inside & result[j])
        n = int(np.floor(fraction * min(len(gain), len(lose))))
        if n:
            result[j, rng.choice(gain, size=n, replace=False)] = True
            result[j, rng.choice(lose, size=n, replace=False)] = False
        swaps += n
    return result, swaps


def jaccard(a, b):
    """Empty/empty has no evidence of similarity and is reported as NaN."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float('nan')
