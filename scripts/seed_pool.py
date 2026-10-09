"""Run independent per-seed jobs in parallel worker processes.

Each RNG seed derives its own random streams and cache entries, so seeds can run
concurrently without changing any result. Workers are forked: the job (typically a
closure over the network, kernel and gene sets) is inherited rather than pickled, and
only its return value is sent back to the parent.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

_JOB = None


def _call(seed):
    return _JOB(seed)


def default_workers(n_seeds):
    """One worker per seed, capped by the CPUs this process may use (the Slurm allocation)."""
    return max(1, min(n_seeds, len(os.sched_getaffinity(0))))


def map_seeds(job, seeds, workers=None):
    """Return {seed: job(seed)}, running up to `workers` seeds at once (1 runs them in order here)."""
    global _JOB
    seeds = list(seeds)
    workers = default_workers(len(seeds)) if workers is None else workers
    if workers < 1:
        raise ValueError('Need at least one worker')
    if workers == 1 or len(seeds) == 1:
        return {seed: job(seed) for seed in seeds}
    _JOB = job
    try:
        with ProcessPoolExecutor(min(workers, len(seeds)), mp_context=mp.get_context('fork')) as pool:
            return dict(zip(seeds, pool.map(_call, seeds)))
    finally:
        _JOB = None
