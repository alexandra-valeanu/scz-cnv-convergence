"""Test whether pair p-values and FDR calls depend on the RNG seed.

Two designs, each with the same seeds:
- end_to_end: the production pipeline rerun per seed; the seed drives both the NPS
  reference draws and the gene-label permutations.
- permutation_only: NPS fixed at the configured seed; only the permutation seed varies.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cnv_overlap.pipeline import AnalysisConfig, _save_tests, pair_plan, pair_scores, run
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import atomic_path, digest, file_hash, run_dir, seal, valid, write_json
from cnv_overlap.statistics import label_null, netcoloc_tail
from network_inputs import BASE_NETWORKS, NETWORKS, load_network_inputs
from seed_pool import map_seeds

DEFAULT_SEEDS = (1, 42, 50)
KEY = ['a', 'b', 'family', 'statistic']
COMPARED = ['observed', 'null_mean', 'null_sd', 'z', 'p', 'q']
# Both designs at the configured seed must agree to this tolerance (they are normally identical).
RECOMPUTE_TOLERANCE = 1e-9


def permutation_tests(zs, nodes, entities, config, seed, out, nps_hash):
    """Mirror pipeline._execute's permutation step with a given permutation seed."""
    records = []
    for a, b, family in pair_plan(entities, config.comparisons):
        path = out/'nulls'/f'seed{seed}-{digest([a, b])[:16]}.npz'
        pm = {'nps': nps_hash, 'seed': seed, 'a': a, 'b': b, 'reps': config.permutation_reps,
              'overlap': config.overlap, 'thresholds': asdict(config.thresholds)}
        values, _ = pair_scores(zs[a], zs[b], entities[a]['seeds'], entities[b]['seeds'], nodes, config)
        if valid(path, pm):
            with np.load(path) as data:
                null = {s: data[s] for s in values}
        else:
            obs, null = label_null(zs[a], zs[b], nodes, entities[a]['seeds'], entities[b]['seeds'],
                                   reps=config.permutation_reps, seed=seed, overlap=config.overlap,
                                   thresholds=config.thresholds)
            assert all(np.isclose(obs[s], values[s]) for s in values)
            with atomic_path(path) as tmp:
                np.savez_compressed(tmp, **null)
            seal(path, pm)
        for statistic, value in values.items():
            records.append({'a': a, 'b': b, 'family': family, 'statistic': statistic, 'null': 'netcoloc',
                            'primary': statistic == 'size', **netcoloc_tail(value, null[statistic])})
    path = out/f'pair_tests_permutation_seed{seed}.tsv'
    _save_tests(records, path)
    return pd.read_csv(path, sep='\t')


def stability_table(long, seeds):
    rows = []
    for (design, *key), frame in long.groupby(['design'] + KEY, sort=True):
        f = frame.set_index('seed')
        logp = -np.log10(f.p)
        row = dict(design=design, **dict(zip(KEY, key)))
        for s in seeds:
            row[f'p_seed{s}'], row[f'q_seed{s}'] = f.p[s], f.q[s]
        row.update(max_delta_log10_p=float(logp.max()-logp.min()), max_delta_z=float(f.z.max()-f.z.min()),
                   n_significant=int(f.significant.sum()), n_seeds=len(seeds))
        row['concordant'] = row['n_significant'] in (0, len(seeds))
        rows.append(row)
    return pd.DataFrame(rows)


def run_pvalue_stability(inputs, *, config, cache_dir, output_root, seeds=DEFAULT_SEEDS, exclude=('t1d',),
                         provenance=None, progress=print, workers=None):
    seeds = tuple(sorted(set(seeds)))
    if len(seeds) < 2:
        raise ValueError('Need at least two RNG seeds')
    A, nodes, symbols, raw_entities, coords, regions = inputs
    entities = {n: e for n, e in raw_entities.items() if n not in exclude}
    provenance = {**(provenance or {}), 'excluded_targets': sorted(exclude)}
    output_root = Path(output_root)
    # NPS exactly as the production run at the configured seed computes them (same cache entries).
    # Building or loading the shared kernel here means seed workers only read it from the cache.
    prop = Propagator(A, nodes, cache_dir, config.propagation)
    prop.H

    def end_to_end(seed):
        say = lambda message: progress(f'[seed {seed}] {message}')
        say('End-to-end pipeline')
        return run(A, nodes, symbols, entities, coords, regions,
                   config=replace(config, propagation=replace(config.propagation, seed=seed)),
                   output_root=output_root/'end_to_end', cache_dir=cache_dir, provenance=provenance,
                   progress=say)

    runs = map_seeds(end_to_end, seeds, workers)
    base_seed = config.propagation.seed
    identity = dict(algorithm='pvalue_stability_v1', code=file_hash(__file__), seeds=seeds, nps_seed=base_seed,
                    config=asdict(config), end_to_end={str(s): runs[s].name for s in seeds},
                    provenance=provenance)
    out = run_dir(output_root, identity, 'pvalue-stability')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['status'] == 'complete':
            if any(not (out/p).exists() or file_hash(out/p) != h for p, h in previous['artifacts'].items()):
                raise ValueError('P-value stability artifacts changed')
            progress(f'Reusing completed p-value stability report: {out}')
            return out
    write_json(manifest, dict(status='running', identity=identity))

    nodes = prop.nodes
    active = {n for a, b, _ in pair_plan(entities, config.comparisons) for n in (a, b)}
    retained = {n: {**entities[n], 'seeds': sorted(set(map(int, entities[n]['seeds'])) & prop.pos.keys())}
                for n in sorted(active)}
    zs = {n: np.asarray(prop.nps(e['seeds'])).copy() for n, e in retained.items()}
    nps_hash = digest({n: digest(z.tobytes().hex()) for n, z in zs.items()})

    def permutation_only(seed):
        progress(f'[seed {seed}] Permutation-only, NPS seed={base_seed}')
        return permutation_tests(zs, nodes, retained, config, seed, out, nps_hash)

    permuted = map_seeds(permutation_only, seeds, workers)
    frames = []
    for seed in seeds:
        frames.append(pd.read_csv(runs[seed]/'pair_tests.tsv', sep='\t').assign(design='end_to_end', seed=seed))
        frames.append(permuted[seed].assign(design='permutation_only', seed=seed))
    long = pd.concat(frames, ignore_index=True)
    long['significant'] = long.q < config.fdr
    long.to_csv(out/'pvalues_by_seed.tsv', sep='\t', index=False)

    recompute_text = f'The configured seed ({base_seed}) is not among the seeds; no cross-design check.\n'
    if base_seed in seeds:
        e2e, perm = (long[(long.design == d) & (long.seed == base_seed)].set_index(KEY).sort_index()[COMPARED]
                     for d in ('end_to_end', 'permutation_only'))
        diff = (e2e-perm).abs()
        check = diff.max().rename('max_abs_diff').rename_axis('column').reset_index()
        check.to_csv(out/'recompute_check.tsv', sep='\t', index=False)
        if not e2e.index.equals(perm.index) or (diff.max() > RECOMPUTE_TOLERANCE).any():
            raise ValueError('Permutation-only tests at the configured seed disagree with the pipeline run; '
                             'see recompute_check.tsv')
        recompute_text = (f'At seed {base_seed} the permutation-only recomputation reproduces the production '
                          f'pipeline tests for every pair and statistic (maximum absolute difference '
                          f'{float(diff.max().max()):.3g}); see recompute_check.tsv.\n')

    table = stability_table(long, seeds)
    table.to_csv(out/'pvalue_stability.tsv', sep='\t', index=False)
    genes = {s: set(pd.read_csv(runs[s]/'recurrence.tsv', sep='\t').entrez) for s in seeds}
    everywhere, anywhere = set.intersection(*genes.values()), set.union(*genes.values())
    pd.DataFrame([dict(entrez=g, n_seeds_selected=sum(g in genes[s] for s in seeds), n_seeds=len(seeds))
                  for g in sorted(anywhere)], columns=['entrez', 'n_seeds_selected', 'n_seeds']
                 ).to_csv(out/'recurrence_genes_by_seed.tsv', sep='\t', index=False)
    plot_pvalues(long, seeds, out)

    lines = []
    for design, frame in table[table.statistic == 'size'].groupby('design'):
        n_all = int((frame.n_significant == len(seeds)).sum())
        n_any = int((frame.n_significant > 0).sum())
        lines.append(f'- {design}: {len(frame)} pairs; {n_all} significant (q < {config.fdr}) with every seed, '
                     f'{n_any} with at least one; largest z-score difference between seeds '
                     f'{frame.max_delta_z.max():.3g}.\n')
        for row in frame[~frame.concordant].itertuples():
            qs = ', '.join(f'{getattr(row, f"q_seed{s}"):.3g}' for s in seeds)
            lines.append(f'  - discordant: {row.a} × {row.b} ({row.family}); q by seed {seeds}: {qs}\n')
    (out/'summary.md').write_text(
        '# P-value seed stability\n\n'
        f'Seeds: {seeds}. Gene-label permutations per pair: {config.permutation_reps:,}. '
        f'NPS reference draws: {config.propagation.reference_reps:,}. BH families as in the production run.\n\n'
        '- end_to_end reruns the production pipeline with each seed, which changes both the NPS reference '
        'draws and the gene-label permutations.\n'
        f'- permutation_only fixes NPS at seed {base_seed} and changes only the permutation seed, isolating '
        'Monte Carlo error in the significance test. NPS reference variability alone is covered by '
        'nps_stability.py.\n\n'
        '## Primary statistic (overlap size)\n\n' + ''.join(lines) + '\n'
        f'Recurrent genes (from pairs passing FDR, end_to_end): {len(everywhere)} with every seed, '
        f'{len(anywhere)} with at least one; see recurrence_genes_by_seed.tsv.\n\n'
        'pvalue_stability.tsv lists every pair and statistic with p and q per seed; pvalues_by_seed.tsv '
        'holds the full test output. Pairs near the FDR threshold can switch with any Monte Carlo seed; '
        'report their q-values rather than a single call. No automatic stability cutoff is applied.\n\n'
        '## Reproducibility\n\n' + recompute_text)
    write_json(manifest, dict(status='complete', identity=identity,
                              artifacts={p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p != manifest}))
    progress(f'Completed: {out}')
    return out


def plot_pvalues(long, seeds, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary = long[long.statistic == 'size']
    reference = seeds[len(seeds)//2]
    designs = ['end_to_end', 'permutation_only']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, design in zip(axes, designs):
        frame = primary[primary.design == design].pivot_table(index=['a', 'b'], columns='seed', values='p')
        ref = -np.log10(frame[reference])
        for s in seeds:
            if s != reference:
                ax.scatter(ref, -np.log10(frame[s]), alpha=.7, label=f'seed {s}')
        lim = np.nanmax(-np.log10(frame.to_numpy())) * 1.05
        ax.plot([0, lim], [0, lim], color='grey', lw=1)
        ax.set(title=design.replace('_', ' '), xlabel=f'-log10 p, seed {reference}', ylabel='-log10 p, other seed')
        ax.legend()
    fig.suptitle('Primary statistic (overlap size); each point is one pair')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'pvalue_stability.{suffix}', dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/cnv_disorder_overlap.json')
    parser.add_argument('--networks', nargs='+', choices=(*NETWORKS, 'custom'), default=list(BASE_NETWORKS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--exclude-targets', nargs='*', default=['t1d'],
                        help='Gene sets dropped before testing, as in the production notebook')
    parser.add_argument('--min-seeds', type=int, default=3)
    parser.add_argument('--target-scope', choices=['default', 'all'], default='default')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--workers', type=int, help='Seeds run in parallel; default one per seed, capped by available CPUs')
    parser.add_argument('--output-root', help='Reports go below NETWORK/pvalue_stability in this directory')
    parser.add_argument('--cache-dir', help='Override the configured propagation cache')
    args = parser.parse_args(argv)
    for network in args.networks:
        spec, root, inputs, provenance = load_network_inputs(args.config, None if network == 'custom' else network,
            min_seeds=args.min_seeds, target_scope=args.target_scope)
        config = AnalysisConfig.from_dict(spec['analysis'])
        if args.threads is not None:
            config = replace(config, propagation=replace(config.propagation, threads=args.threads))
        run_pvalue_stability(inputs, config=config, cache_dir=args.cache_dir or root/spec['cache_dir'],
                             output_root=Path(args.output_root or root/'results')/network/'pvalue_stability',
                             seeds=args.seeds, exclude=args.exclude_targets, provenance=provenance, workers=args.workers,
                             progress=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
