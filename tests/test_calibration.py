"""Monte Carlo sanity checks, not a claim of calibration on real CNV data."""
import numpy as np

from cnv_overlap.statistics import label_null, tail


def test_label_null_exchangeable_scores_and_planted_alignment():
    rng = np.random.default_rng(39)
    null_hits = planted_hits = 0
    n_trials = 100
    for _ in range(n_trials):
        a = rng.normal(size=80)*2
        b = rng.normal(size=80)*2
        obs, null = label_null(a, b, np.arange(80), reps=199, seed=int(rng.integers(1000000)))
        null_hits += tail(obs["size"], null["size"])["p"] < .05
        obs, null = label_null(a, a+.1*b, np.arange(80), reps=199)
        planted_hits += tail(obs["size"], null["size"])["p"] < .05
    assert null_hits <= 12  # Broad sampling tolerance; deterministic seed.
    assert planted_hits >= 90
