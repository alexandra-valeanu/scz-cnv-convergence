"""Conditional inference on fixed, CNV-independent schizophrenia modules.

Inputs are CNV x gene Boolean reach matrices. All statistics are recomputed on
joint matched-seed draws; modules must be frozen before examining CNV results.
"""
import numpy as np
import pandas as pd
from cnv_overlap.statistics import bh, tail


def module_counts(reach, membership):
    return np.asarray(reach, dtype=np.int64) @ np.asarray(membership, dtype=np.int64).T


def module_statistics(counts, totals, sizes):
    """Pair coverage permits different genes in the same module to converge.

    Mean over all distinct CNV pairs of the product of their module fractions.
    A single CNV cannot generate this statistic alone. Shares include unassigned
    landscape genes in their denominator; zero landscape reach gives share zero.
    """
    counts = np.asarray(counts, dtype=float)
    fractions = counts / np.asarray(sizes)
    n = counts.shape[-2]
    if n < 2:
        raise ValueError('Module convergence requires at least two CNVs')
    pair_coverage = ((fractions.sum(axis=-2)**2 - (fractions**2).sum(axis=-2)) / (n*(n-1)))
    shares = np.divide(counts, np.asarray(totals)[..., None],
                       out=np.zeros_like(counts), where=np.asarray(totals)[..., None] > 0)
    return {'coverage': counts, 'landscape_share': shares,
            'pair_coverage': np.maximum(pair_coverage, 0),
            'cnvs_reaching': (counts > 0).sum(axis=-2)}


def module_tests(observed, null, observed_totals, null_totals, sizes, modules, names, mode):
    """Empirical upper tails; BH families explicitly separated by endpoint.

    Holm adjustment is also reported to permit familywise inference without
    assumptions about dependence among overlapping CNV profiles.
    """
    observed, null = np.asarray(observed), np.asarray(null)
    sizes = np.asarray(sizes)
    if (observed.shape != (len(names), len(modules)) or
            null.ndim != 3 or null.shape[1:] != observed.shape or null.shape[0] < 2 or
            sizes.shape != (len(modules),) or np.any(sizes <= 0)):
        raise ValueError('Require aligned CNV/module arrays, positive module sizes and >=2 draws')
    obs = module_statistics(observed, observed_totals, sizes)
    draws = module_statistics(null, null_totals, sizes)
    rows = []
    for endpoint in ('pair_coverage', 'cnvs_reaching', 'coverage', 'landscape_share'):
        for m, module in enumerate(modules):
            cnvs = range(len(names)) if endpoint in ('coverage', 'landscape_share') else [None]
            for c in cnvs:
                key = (c, m) if c is not None else m
                null_values = draws[endpoint][:, c, m] if c is not None else draws[endpoint][:, m]
                rows.append({'mode': mode, 'endpoint': endpoint, 'module': module,
                             'module_genes': int(sizes[m]), 'cnv': names[c] if c is not None else 'all',
                             **tail(obs[endpoint][key], null_values)})
    columns = ['mode', 'endpoint', 'module', 'module_genes', 'cnv', 'observed', 'p', 'q', 'p_holm']
    result = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    for endpoint, indexes in result.groupby('endpoint').groups.items():
        p = result.loc[indexes, 'p'].to_numpy()
        result.loc[indexes, 'q'] = bh(p)
        order = np.argsort(p, kind='stable')
        holm = np.empty(len(p))
        holm[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
        result.loc[indexes, 'p_holm'] = holm
        result.loc[indexes, 'family'] = f'{mode}:{endpoint}'
        result.loc[indexes, 'family_size'] = len(p)
    return result
