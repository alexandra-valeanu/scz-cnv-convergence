"""Restricted degree-matched joint CNV null within a fixed SCHEMA–GWAS landscape.

Real seeds are unchanged. Random seeds exclude ALL input CNV/SCHEMA/GWAS seeds.
An injective, within-degree-bin mapping of the CNV seed union preserves every
between-CNV overlap. References remain the original unrestricted NPS references.
"""
from cnv_overlap.results import results_dir, finalize_manifest, resolve_path
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from cnv_overlap.pipeline import AnalysisConfig
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import atomic_path, file_hash, run_dir, write_json
from cnv_overlap.statistics import bh, hit_mask, rng_for
from scz_shared_network import load_inputs

TARGETS = ('schema', 'gwaspri')
METRICS = ('genes', 'largest_component', 'edges')


def sampling_plan(prop, seeds, names):
    excluded = set().union(*(set(s) for s in seeds.values()))
    union = np.array(sorted(set().union(*(set(seeds[n]) for n in names))), dtype=int)
    cols = prop.seed_indices(union)
    groups = []
    rows = []
    for b, pool in enumerate(prop.bins):
        original = union[prop.bin_of[cols] == b]
        eligible = pool[~np.isin(prop.nodes[pool], list(excluded))]
        rows.append({'degree_bin': b, 'degree_min': int(prop.degrees[pool].min()),
                     'degree_max': int(prop.degrees[pool].max()), 'eligible_genes': len(eligible),
                     'required_unique_cnv_genes': len(original)})
        if len(original) > len(eligible):
            raise ValueError(f'Degree bin {b}: need {len(original)} distinct replacements, '
                             f'only {len(eligible)} eligible; exact matching is infeasible')
        groups.append((original, eligible))
    return groups, pd.DataFrame(rows)


def sample_joint(prop, seeds, names, groups, rng):
    mapping = {}
    for original, eligible in groups:
        replacements = prop.nodes[rng.choice(eligible, size=len(original), replace=False)]
        mapping.update(zip(map(int, original), map(int, replacements)))
    return {n: [mapping[g] for g in seeds[n]] for n in names}


def core_counts(profiles, names, thresholds, landscape):
    # Same CNV must colocalise with BOTH targets; additionally require membership
    # in the fixed SCHEMA–GWAS candidate landscape. No seed removal.
    counts = np.zeros(int(landscape.sum()), dtype=np.int16)
    for name in names:
        counts += (hit_mask(profiles[name], profiles['schema'], thresholds)
                   & hit_mask(profiles[name], profiles['gwaspri'], thresholds))[landscape]
    return counts


def topology(A, mask):
    n = int(mask.sum())
    if not n:
        return (0, 0, 0)
    sub = A[mask][:, mask]
    _, labels = connected_components(sub, directed=False)
    return (n, int(np.bincount(labels).max()), int(sub.nnz // 2))


def landscape_heat(kernel_rows, cols):
    """Exact NetColoc accumulation on selected rows of the FULL-network kernel."""
    heat = np.zeros(kernel_rows.shape[0], dtype=float)
    for col in sorted(cols):
        heat += kernel_rows[:, col]
    return heat / len(cols)


def run_restricted(inputs, *, config, cache_dir, output_root, reps=5000,
                   thresholds=(2, 3), progress=print):
    thresholds = tuple(sorted(set(thresholds)))
    if reps < 2 or not thresholds or any(k < 1 for k in thresholds):
        raise ValueError('Require at least two replicates and positive core thresholds')
    A, nodes, symbols, entities, *_ = inputs
    prop = Propagator(A, nodes, cache_dir, config.propagation)
    names = sorted(n for n, e in entities.items() if e['kind'] == 'cnv')
    if not names or not all(t in entities for t in TARGETS):
        raise ValueError('Require CNVs and both SCZ targets')
    if max(thresholds) > len(names):
        raise ValueError('Core threshold exceeds the number of CNVs')
    seeds = {n: sorted(set(map(int, entities[n]['seeds'])) & prop.pos.keys()) for n in names + list(TARGETS)}
    groups, balance = sampling_plan(prop, seeds, names)
    identity = {'algorithm': 'restricted_joint_core_v1', 'code': file_hash(__file__),
                'propagation': prop.metadata, 'config': asdict(config), 'seeds': seeds,
                'symbols': {str(g): symbols.get(int(g), str(g)) for g in prop.nodes},
                'reps': reps, 'thresholds': thresholds, 'preserve_cnv_seed_overlap': True,
                'observed_seed_removal': False, 'core_rule': 'both_pair_masks_inside_fixed_gsz'}
    out = run_dir(output_root, identity, 'restricted-core', f'reps{reps}')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get('status') == 'complete':
            if any(not (out/f).exists() or file_hash(out/f) != h for f, h in old['artifacts'].items()):
                raise ValueError('Completed restricted-null artifacts changed')
            return out
    write_json(manifest, {'status': 'running', 'identity': identity})
    balance.to_csv(resolve_path(out/'degree_pool_qc.tsv'), sep='\t', index=False)
    scz = set(seeds['schema']) | set(seeds['gwaspri'])
    pd.DataFrame([{'entity': n, 'raw_seeds': len(set(entities[n]['seeds'])),
                   'mapped_seeds': len(seeds[n]), 'retained_scz_overlap': len(set(seeds[n]) & scz)}
                  for n in seeds]).to_csv(resolve_path(out/'seed_qc.tsv'), sep='\t', index=False)
    pd.DataFrame([{'entrez': g, 'symbol': symbols.get(g, str(g)),
                   'seed_of': ','.join(n for n in seeds if g in seeds[n])}
                  for g in sorted(set().union(*map(set, seeds.values()))) ]).to_csv(
                      resolve_path(out/'excluded_random_seeds.tsv'), sep='\t', index=False)
    progress('Loading fixed profiles and unchanged NPS references')
    profiles = {n: np.asarray(prop.nps(s)) for n, s in seeds.items()}
    landscape = hit_mask(profiles['schema'], profiles['gwaspri'], config.thresholds)
    if not landscape.any():
        raise ValueError('Empty SCHEMA–GWAS landscape')
    observed_counts = core_counts(profiles, names, config.thresholds, landscape)
    graph = prop.A[landscape][:, landscape]
    observed_masks = np.array([observed_counts >= k for k in thresholds])
    observed = np.array([topology(graph, m) for m in observed_masks])
    references = {n: prop.reference(prop.seed_indices(seeds[n]))[:2] for n in names}
    # Selecting output rows does not restrict propagation paths to the landscape.
    kernel_rows = np.asarray(prop.H[landscape])
    target_profiles = {t: profiles[t][landscape] for t in TARGETS}
    all_landscape = np.ones(int(landscape.sum()), dtype=bool)
    progress(f'Fixed landscape: {landscape.sum()} genes; {len(names)} CNVs; observed cores: '
             + str(dict(zip(thresholds, observed[:, 0].tolist()))))
    blocks = []
    for start in range(0, reps, 100):
        stop = min(start + 100, reps)
        path = resolve_path(out/f'draws-{start:06d}.npz')
        stamp = path.with_suffix('.json')
        if not (path.exists() and stamp.exists() and json.loads(stamp.read_text())['sha256'] == file_hash(path)):
            metrics, overlaps, packed, sampled = [], [], [], []
            for rep in range(start, stop):
                draws = sample_joint(prop, seeds, names, groups,
                                     rng_for(config.propagation.seed, 'restricted_core', rep))
                random_profiles = dict(target_profiles)
                for n in names:
                    mu, sd = references[n]
                    heat = landscape_heat(kernel_rows, prop.seed_indices(draws[n]))
                    random_profiles[n] = (np.log(heat) - mu[landscape]) / sd[landscape]
                counts = core_counts(random_profiles, names, config.thresholds, all_landscape)
                masks = np.array([counts >= k for k in thresholds])
                metrics.append([topology(graph, m) for m in masks])
                overlaps.append((masks & observed_masks).sum(axis=1))
                packed.append(np.packbits(masks, axis=-1))
                sampled.append([g for n in names for g in draws[n]])
            with atomic_path(path) as tmp:
                np.savez_compressed(tmp, metrics=metrics, overlaps=overlaps, masks=packed, seeds=sampled)
            write_json(stamp, {'sha256': file_hash(path)})
        with np.load(path) as data:
            blocks.append({k: data[k] for k in data.files})
        progress(f'Restricted core null: {stop}/{reps}')
    null = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    rows = []
    for i, k in enumerate(thresholds):
        for j, metric in enumerate(METRICS):
            values = null['metrics'][:, i, j]
            obs = int(observed[i, j])
            rows.append({'min_cnvs': k, 'statistic': metric, 'observed': obs,
                         'null_mean': values.mean(), 'null_low': np.quantile(values, .025),
                         'null_high': np.quantile(values, .975),
                         'fold_vs_null': obs / values.mean() if values.mean() else np.nan,
                         'p_empirical': (1 + (values >= obs).sum()) / (reps + 1), 'reps': reps})
    tests = pd.DataFrame(rows)
    tests['q_bh'] = bh(tests.p_empirical)
    tests.to_csv(resolve_path(out/'network_tests.tsv'), sep='\t', index=False)
    membership = pd.DataFrame({'entrez': prop.nodes[landscape],
                               'symbol': [symbols.get(int(g), str(g)) for g in prop.nodes[landscape]],
                               'observed_cnv_count': observed_counts})
    masks = np.unpackbits(null['masks'], axis=-1, count=int(landscape.sum())).astype(bool)
    for i, k in enumerate(thresholds):
        membership[f'observed_core_min{k}'] = observed_masks[i]
        membership[f'null_core_frequency_min{k}'] = masks[:, i].mean(axis=0)
    membership.to_csv(resolve_path(out/'landscape_gene_membership.tsv'), sep='\t', index=False)
    pd.DataFrame([{'replicate': r, 'min_cnvs': k, 'core_genes': int(null['metrics'][r, i, 0]),
                   'overlap_with_observed': int(null['overlaps'][r, i])}
                  for r in range(reps) for i, k in enumerate(thresholds)]).to_csv(
                      resolve_path(out/'null_core_overlap.tsv'), sep='\t', index=False)
    write_json(out/'draw_layout.json', {'cnv_order': names,
        'seed_counts': [len(seeds[n]) for n in names], 'thresholds': thresholds,
        'metrics': METRICS, 'mask_gene_order': prop.nodes[landscape].tolist()})
    lines = ['# Restricted random-seed shared-core test', '',
             f'{reps} replicates; fixed SCHEMA–GWAS landscape: {landscape.sum()} genes.',
             'Core: landscape genes colocalising with BOTH targets for the same >=k CNVs.',
             'All observed CNV seeds retained, including SCHEMA/GWAS overlaps.',
             'Random pool excludes every mapped CNV, SCHEMA and GWAS seed.',
             'Each draw preserves seed counts, degree-bin histograms and between-CNV seed overlaps.',
             'Propagation and unrestricted degree-matched NPS normalization are unchanged.',
             'BH correction covers all six statistics/threshold combinations by default.',
             'Connectivity statistics include core-size effects; they are not conditional on size.',
             'Gene recovery frequencies and observed-core overlap are descriptive, not gene-level discoveries.',
             'This restricted-background sensitivity analysis is conditional on the fixed SCZ landscape.', '',
             tests.to_string(index=False), '']
    (out/'summary.md').write_text('\n'.join(lines))
    finalize_manifest(manifest, {'status': 'complete', 'identity': identity,
        'landscape_size': int(landscape.sum()),
        'artifacts': {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p != manifest}})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', default='PCNET2.2_brain')
    parser.add_argument('--reps', type=int, default=5000)
    parser.add_argument('--min-cnvs', nargs='+', type=int, default=[2, 3])
    args = parser.parse_args()
    spec, root, inputs = load_inputs(args.network, 3)
    out = run_restricted(inputs, config=AnalysisConfig.from_dict(spec['analysis']),
                         cache_dir=root/spec['cache_dir'],
                         output_root=results_dir(root/spec['results_root'], 'scz_core_gene_test')/'restricted',
                         reps=args.reps, thresholds=args.min_cnvs,
                         progress=lambda s: print(s, flush=True))
    print(out, flush=True)


if __name__ == '__main__':
    main()
