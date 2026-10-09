"""Compare NPS reference budgets without running significance permutations."""
import argparse
from dataclasses import asdict, replace
from itertools import combinations
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cnv_overlap.pipeline import AnalysisConfig
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import atomic_path, file_hash, run_dir, write_json
from cnv_overlap.statistics import hit_mask
from network_inputs import BASE_NETWORKS, NETWORKS, load_network_inputs
from notebook_analysis import SCZ_TARGETS, scz_overlap_membership
from seed_pool import map_seeds

DEFAULT_SEEDS = (1, 42, 50)
# Cached and freshly rebuilt NPS must agree to this tolerance (they are normally bit-identical).
RECOMPUTE_TOLERANCE = 1e-6


def set_metrics(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return dict(size_a=int(a.sum()), size_b=int(b.sum()),
                gained=int((~a & b).sum()), lost=int((a & ~b).sum()),
                jaccard=float((a & b).sum()/union) if union else 1.)


def selections(nodes, entities, profiles, config, min_cnvs):
    masks = {('individual', n, ''): z > config.thresholds.individual for n, z in profiles.items()}
    cnvs = sorted(n for n, e in entities.items() if e['kind'] == 'cnv')
    targets = sorted(n for n, e in entities.items() if e['kind'] == 'target')
    for a, b in list(combinations(cnvs, 2)) + [(a, b) for a in cnvs for b in targets]:
        mask = hit_mask(profiles[a], profiles[b], config.thresholds)
        if config.overlap == 'remove':
            mask &= ~np.isin(nodes, list(set(entities[a]['seeds']) & set(entities[b]['seeds'])))
        masks['pair', a, b] = mask
    if all(n in targets for n in SCZ_TARGETS):
        table, _, both = scz_overlap_membership(nodes, entities, profiles, config, min_cnvs)
        masks['shared_core', str(min_cnvs), 'schema+gwaspri'] = table.shared_core.to_numpy()
        masks.update({('cnv_both', n, 'schema+gwaspri'): mask for n, mask in both.items()})
    return masks


def recompute_check(prop, retained, profiles, seed, reps):
    """Rebuild every NPS in an empty cache, sharing only the kernel, so reported
    scores cannot depend on stale or foreign cached references."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Propagator(prop.A, prop.nodes, tmp, prop.config)
        fresh._H = prop.H
        for n in sorted(retained):
            z = np.array(fresh.nps(retained[n]))
            rows.append(dict(seed=seed, reps=reps, entity=n, identical=bool(np.array_equal(z, profiles[n])),
                             max_abs_diff=float(np.abs(z-profiles[n]).max())))
    return rows


def seed_consensus(profiles, masks, entities, nodes, symbols, seeds, reps):
    """Summarise agreement across independent RNG seeds at each reference budget."""
    variability, consensus, genes = [], [], []
    for r in reps:
        keys = [(seed, r) for seed in seeds]
        for n in sorted(entities):
            sd = np.stack([profiles[k][n] for k in keys]).std(axis=0, ddof=1)
            # Pure Monte Carlo error scales as 1/sqrt(reps), so this column should be roughly flat across budgets.
            variability.append(dict(reps=r, entity=n, n_seeds=len(seeds), median_sd=float(np.median(sd)),
                                    p95_sd=float(np.quantile(sd, .95)), max_sd=float(sd.max()),
                                    median_sd_x_sqrt_reps=float(np.median(sd)*np.sqrt(r))))
        for key in masks[keys[0]]:
            stack = np.stack([masks[k][key] for k in keys])
            votes = stack.sum(axis=0)
            kind, a, b = key
            union, unanimous = int((votes > 0).sum()), int((votes == len(seeds)).sum())
            consensus.append(dict(reps=r, kind=kind, a=a, b=b, n_seeds=len(seeds),
                                  min_size=int(stack.sum(1).min()), max_size=int(stack.sum(1).max()),
                                  unanimous=unanimous, any_seed=union, discordant=union-unanimous,
                                  consensus_jaccard=unanimous/union if union else 1.))
            if kind in ('shared_core', 'cnv_both'):
                for i in np.flatnonzero(votes):
                    genes.append(dict(reps=r, kind=kind, a=a, b=b, entrez=int(nodes[i]),
                                      symbol=symbols.get(int(nodes[i]), str(nodes[i])),
                                      n_seeds_selected=int(votes[i]), n_seeds=len(seeds)))
    return (pd.DataFrame(variability), pd.DataFrame(consensus),
            pd.DataFrame(genes, columns=['reps', 'kind', 'a', 'b', 'entrez', 'symbol',
                                         'n_seeds_selected', 'n_seeds']))


def run_stability(inputs, *, config, cache_dir, output_root, reps=(1000, 5000, 10000),
                  seeds=DEFAULT_SEEDS, min_cnvs=3, provenance=None, progress=print, workers=None):
    reps, seeds = tuple(sorted(set(reps))), tuple(sorted(set(seeds)))
    if len(reps) < 2 or not seeds or min_cnvs < 1:
        raise ValueError('Need at least two reference budgets, one seed and positive min-cnvs')
    for r in reps:
        for seed in seeds:
            replace(config.propagation, reference_reps=r, seed=seed)
    A, nodes, symbols, raw_entities, *_ = inputs
    entities = {n: e for n, e in raw_entities.items()
                if e['kind'] == 'cnv' or (e['kind'] == 'target' and n != 't1d')}
    if not any(e['kind'] == 'cnv' for e in entities.values()):
        raise ValueError('Need at least one CNV')
    base = Propagator(A, nodes, cache_dir, config.propagation)
    A, nodes = base.A, base.nodes
    retained = {n: sorted(set(e['seeds']) & base.pos.keys()) for n, e in entities.items()}
    for n, genes in retained.items():
        if not genes:
            raise ValueError(f'No on-network seeds for {n}')
    identity = dict(algorithm='nps_stability_v2', code=file_hash(__file__),
                    membership_code=file_hash(Path(__file__).with_name('notebook_analysis.py')),
                    propagation=base.metadata, entities=entities, symbols={str(k): v for k, v in symbols.items()},
                    thresholds=asdict(config.thresholds), overlap=config.overlap,
                    reps=reps, seeds=seeds, min_cnvs=min_cnvs, provenance=provenance or {})
    out = run_dir(output_root, identity, 'nps-stability')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['status'] == 'complete':
            if any(not (out/p).exists() or file_hash(out/p) != h for p, h in previous['artifacts'].items()):
                raise ValueError('Stability artifacts changed')
            progress(f'Reusing completed stability report: {out}')
            return out
    write_json(manifest, dict(status='running', identity=identity))
    pd.DataFrame([dict(entity=n, raw_seeds=len(set(entities[n]['seeds'])), retained_seeds=len(g),
                       dropped_seeds=','.join(map(str, sorted(set(entities[n]['seeds'])-set(g)))))
                  for n, g in retained.items()]).to_csv(out/'seed_qc.tsv', sep='\t', index=False)
    # Rebuild at the production budget so the verified scores are the ones the main analysis uses.
    check_reps = config.propagation.reference_reps if config.propagation.reference_reps in reps else reps[0]
    # Build or load the shared kernel once, so seed workers only read it from the cache.
    base.H

    def one_seed(seed):
        say = lambda message: progress(f'[seed {seed}] {message}')
        profiles, masks, counts, recompute = {}, {}, [], []
        for r in reps:
            say(f'NPS reference draws={r}')
            prop = Propagator(A, nodes, cache_dir, replace(config.propagation, reference_reps=r, seed=seed))
            profiles[r] = {}
            table = pd.DataFrame({'entrez': nodes, 'symbol': [symbols.get(int(g), str(g)) for g in nodes]})
            for n in sorted(entities):
                say(f'  {n}: computing/reusing NPS and reference')
                z = np.asarray(prop.nps(retained[n])).copy()
                profiles[r][n] = z
                table[f'nps_{n}'] = z
                effective = prop.reference(prop.seed_indices(retained[n]))[2]
                table[f'reference_count_{n}'] = effective.astype(int)
                counts.append(dict(seed=seed, reps=r, entity=n, minimum=int(effective.min()),
                                   median=float(np.median(effective)), maximum=int(effective.max())))
            with atomic_path(out/f'nps_seed{seed}_reps{r}.tsv.gz') as tmp:
                table.to_csv(tmp, sep='\t', index=False, compression='gzip')
            masks[r] = selections(nodes, entities, profiles[r], config, min_cnvs)
            membership = pd.DataFrame({'entrez': nodes, 'symbol': table.symbol})
            for (kind, a, b), mask in masks[r].items():
                if kind in ('shared_core', 'cnv_both'):
                    membership[f'{kind}:{a}:{b}'] = mask
            membership.to_csv(out/f'membership_seed{seed}_reps{r}.tsv', sep='\t', index=False)
            if r == check_reps:
                say('  rebuilding NPS in an empty cache to verify reproducibility')
                recompute.extend(recompute_check(prop, retained, profiles[r], seed, r))
            # Release per-profile cache mappings before the next budget.
            del prop
        return profiles, masks, counts, recompute

    profiles, masks, counts, recompute = {}, {}, [], []
    for seed, (seed_profiles, seed_masks, seed_counts, seed_recompute) in map_seeds(one_seed, seeds, workers).items():
        for r in reps:
            profiles[seed, r], masks[seed, r] = seed_profiles[r], seed_masks[r]
        counts.extend(seed_counts)
        recompute.extend(seed_recompute)
    score_rows, set_rows, crossing_rows = [], [], []
    for ka, kb in combinations(profiles, 2):
        if ka[0] != kb[0] and ka[1] != kb[1]:
            continue
        meta = dict(comparison='budget' if ka[0] == kb[0] else 'seed',
                    seed_a=ka[0], reps_a=ka[1], seed_b=kb[0], reps_b=kb[1])
        for n in sorted(entities):
            a, b = profiles[ka][n], profiles[kb][n]
            delta = b-a
            variable = np.ptp(a) > 0 and np.ptp(b) > 0
            score_rows.append(dict(**meta, entity=n, mean_delta=float(delta.mean()),
                rmse=float(np.sqrt(np.mean(delta**2))), median_abs_delta=float(np.median(abs(delta))),
                p95_abs_delta=float(np.quantile(abs(delta), .95)), max_abs_delta=float(abs(delta).max()),
                pearson=float(np.corrcoef(a, b)[0, 1]) if variable else np.nan,
                spearman=float(spearmanr(a, b).statistic) if variable else np.nan))
        for key, a in masks[ka].items():
            b = masks[kb][key]
            kind, name_a, name_b = key
            row = dict(**meta, kind=kind, a=name_a, b=name_b)
            set_rows.append(dict(**row, **set_metrics(a, b)))
            for i in np.flatnonzero(a != b):
                crossing_rows.append(dict(**row, entrez=int(nodes[i]), symbol=symbols.get(int(nodes[i]), str(nodes[i])),
                                          selected_a=bool(a[i]), selected_b=bool(b[i])))
    pd.DataFrame(counts).to_csv(out/'reference_counts.tsv', sep='\t', index=False)
    recompute = pd.DataFrame(recompute)
    recompute.to_csv(out/'recompute_check.tsv', sep='\t', index=False)
    if (recompute.max_abs_diff > RECOMPUTE_TOLERANCE).any():
        raise ValueError('Freshly rebuilt NPS disagree with cached NPS; see recompute_check.tsv')
    consensus_text = 'Only one RNG seed was supplied; seed agreement was not assessed.\n'
    if len(seeds) > 1:
        variability, consensus, genes = seed_consensus(profiles, masks, entities, nodes, symbols, seeds, reps)
        variability.to_csv(out/'seed_variability.tsv', sep='\t', index=False)
        consensus.to_csv(out/'seed_consensus.tsv', sep='\t', index=False)
        genes.to_csv(out/'seed_consensus_membership.tsv.gz', sep='\t', index=False)
        core = consensus[consensus.kind == 'shared_core']
        consensus_text = ''.join(
            f'- {row.reps:,} draws: {row.unanimous} genes in all {row.n_seeds} seeds, '
            f'{row.any_seed} in at least one (sizes {row.min_size}-{row.max_size}).\n'
            for row in core.itertuples()) or 'No shared-core selection (SCHEMA/GWAS targets absent).\n'
    score_table = pd.DataFrame(score_rows)
    score_table.to_csv(out/'score_stability.tsv', sep='\t', index=False)
    pd.DataFrame(set_rows).to_csv(out/'selection_stability.tsv', sep='\t', index=False)
    columns = ['comparison', 'seed_a', 'reps_a', 'seed_b', 'reps_b', 'kind', 'a', 'b',
               'entrez', 'symbol', 'selected_a', 'selected_b']
    pd.DataFrame(crossing_rows, columns=columns).to_csv(out/'threshold_crossings.tsv.gz', sep='\t', index=False)
    plot_stability(score_table, out)
    (out/'summary.md').write_text(
        '# NPS reference stability\n\n'
        f'Reference budgets: {reps}. RNG seeds: {seeds}.\n\n'
        'Compare score_stability.tsv and selection_stability.tsv; threshold_crossings.tsv.gz '
        'identifies every changed membership. NPS and effective unseeded reference counts '
        'are retained for every gene. Shared-core memberships require both SCHEMA and GWAS.\n\n'
        'Budgets within each RNG seed use nested random draws, matching the production '
        'Propagator. The largest budget is a comparator, not ground truth. Seed comparisons '
        'at each budget assess independent Monte Carlo variability when multiple seeds are supplied. '
        'Empty/empty selections have Jaccard 1; inspect set sizes as well as similarity. '
        'No automatic stability cutoff or significance claim is made. '
        'Pair-label permutations, convergence null tests and enrichment are not rerun.\n\n'
        '## Seed agreement\n\n'
        'Each RNG seed is hashed (SHA-256) with the degree-bin signature to derive its reference '
        'stream, so different seeds give independent draws. seed_variability.tsv gives the across-seed '
        'NPS standard deviation per gene set and budget; median_sd_x_sqrt_reps should be roughly constant '
        'if the differences are pure Monte Carlo error. seed_consensus.tsv counts genes selected by all '
        'seeds versus any seed, and seed_consensus_membership.tsv.gz lists every shared-core/CNV-both gene '
        'with the number of seeds selecting it.\n\n'
        'Shared core:\n\n' + consensus_text + '\n'
        '## Reproducibility\n\n'
        f'At {check_reps:,} draws (the configured production budget when included) every NPS profile was rebuilt from scratch in an empty cache for each seed '
        f'and compared with the reported values ({int(recompute.identical.sum())}/{len(recompute)} '
        f'bit-identical; maximum absolute difference {recompute.max_abs_diff.max():.3g}). '
        'recompute_check.tsv holds the details.\n')
    write_json(manifest, dict(status='complete', identity=identity,
                             artifacts={p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p != manifest}))
    progress(f'Completed: {out}')
    return out


def plot_stability(table, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    budget = table[table.comparison == 'budget']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    labels, values = [], []
    for (a, b), frame in budget.groupby(['reps_a', 'reps_b']):
        labels.append(f'{a:,} vs {b:,}')
        values.append(frame)
    for ax, metric, title in zip(axes[:2], ['rmse', 'p95_abs_delta'], ['NPS root mean squared change', '95th percentile absolute NPS change']):
        for i, frame in enumerate(values):
            ax.scatter(np.full(len(frame), i), frame[metric], alpha=.65)
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel='NPS units', title=title)
    seed = table[table.comparison == 'seed']
    ax = axes[2]
    if len(seed):
        groups = list(seed.groupby('reps_a'))
        for i, (_, frame) in enumerate(groups):
            ax.scatter(np.full(len(frame), i), frame.rmse, alpha=.65)
        ax.set(xticks=range(len(groups)), xticklabels=[f'{r:,}' for r, _ in groups],
               xlabel='Reference draws', ylabel='NPS units', title='NPS RMSE between RNG seeds')
    else:
        ax.text(.5, .5, 'Single RNG seed', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    fig.suptitle('Each point is one gene set and RNG seed (pair)')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'nps_stability.{suffix}', dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cnv_disorder_overlap.json', help='Include all CNVs and targets; defaults to the combined disorder snapshot')
    parser.add_argument('--networks', nargs='+', choices=(*NETWORKS, 'custom'), default=list(BASE_NETWORKS))
    parser.add_argument('--reps', nargs='+', type=int, default=[1000, 5000, 10000])
    parser.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument('--min-seeds', type=int, default=3)
    parser.add_argument('--target-scope', choices=['default', 'all'], default='default')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--workers', type=int, help='Seeds run in parallel; default one per seed, capped by available CPUs')
    parser.add_argument('--min-cnvs', type=int, default=3)
    parser.add_argument('--output-root', help='Reports go below NETWORK/nps_stability in this directory')
    parser.add_argument('--cache-dir', help='Override the configured propagation cache, e.g. persistent cluster scratch')
    args = parser.parse_args(argv)
    for network in args.networks:
        spec, root, inputs, provenance = load_network_inputs(args.config, None if network == 'custom' else network,
            min_seeds=args.min_seeds, target_scope=args.target_scope)
        config = AnalysisConfig.from_dict(spec['analysis'])
        if args.threads is not None:
            config = replace(config, propagation=replace(config.propagation, threads=args.threads))
        run_stability(inputs, config=config, cache_dir=args.cache_dir or root/spec['cache_dir'],
                      output_root=Path(args.output_root or root/'results')/network/'nps_stability',
                      reps=args.reps, seeds=args.seeds, min_cnvs=args.min_cnvs, provenance=provenance, workers=args.workers,
                      progress=lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
