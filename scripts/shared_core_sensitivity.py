"""Threshold sensitivity and a conditional joint seed null for the SCZ core."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cnv_overlap.enrichment import load_gmt, ora
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import atomic_path, file_hash, run_dir, write_json
from cnv_overlap.statistics import bh, hit_mask, rng_for


MODES = ('schema', 'gwaspri', 'either', 'both')


def target_counts(profiles, cnv_names, nodes, seeds, config):
    """Count distinct CNVs; OR/AND operate within each CNV at the same gene.

    For overlap=remove, seeds must describe the current draw, not the real CNV.
    """
    counts = np.zeros((len(MODES), len(nodes)), dtype=np.int16)
    for name in cnv_names:
        masks = []
        for target in MODES[:2]:
            mask = hit_mask(profiles[name], profiles[target], config.thresholds)
            if config.overlap == 'remove':
                mask &= ~np.isin(nodes, list(set(seeds[name]) & set(seeds[target])))
            masks.append(mask)
        a, b = masks
        counts += np.stack([a, b, a | b, a & b])
    return counts


def _write_table(frame, path):
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, sep='\t', index=False)


def run_core_sensitivity(inputs, *, config, cache_dir, output_root,
                         thresholds=(2, 3, 4), primary_min_cnvs=3, reps=None,
                         annotation_dir=None, progress=print):
    """Rebuild every core in every joint degree-matched draw, fixing SCZ profiles.

    Primary definition is BOTH at primary_min_cnvs. All variants receive empirical
    tails and a single BH correction across the declared sensitivity family.
    This tests core size, not individual genes, edges, or conditional enrichment.
    """
    from shared_network_enrichment import LIBRARIES

    reps = config.permutation_reps if reps is None else reps
    thresholds = tuple(thresholds)
    if (not isinstance(reps, int) or isinstance(reps, bool) or reps < 2
            or not thresholds or any(not isinstance(k, int) or isinstance(k, bool) or k < 1
                                     for k in (*thresholds, primary_min_cnvs))):
        raise ValueError('Require integer reps >= 2 and positive integer CNV thresholds')
    thresholds = tuple(sorted(set(thresholds) | {primary_min_cnvs}))
    A, nodes, symbols, raw_entities, *_ = inputs
    entities = {n: e for n, e in sorted(raw_entities.items())
                if e['kind'] == 'cnv' or (n in MODES[:2] and e['kind'] == 'target')}
    names = sorted(n for n, e in entities.items() if e['kind'] == 'cnv')
    if not names or not set(MODES[:2]) <= entities.keys():
        raise ValueError('Core sensitivity requires CNVs, SCHEMA and GWAS targets')
    prop = Propagator(A, nodes, cache_dir, config.propagation)
    nodes = prop.nodes
    seeds = {n: sorted(set(e['seeds']) & prop.pos.keys()) for n, e in entities.items()}
    paths = ({library: Path(annotation_dir)/filename for library, filename in LIBRARIES.items()}
             if annotation_dir is not None else {})
    identity = {'algorithm': 'conditional_joint_core_size_v1', 'code': file_hash(__file__),
                'propagation': prop.metadata, 'config': asdict(config), 'entities': entities,
                'symbols': {str(g): symbols.get(int(g), str(g)) for g in nodes},
                'thresholds': thresholds, 'modes': MODES, 'primary_min_cnvs': primary_min_cnvs,
                'reps': reps, 'gmt': {n: file_hash(p) for n, p in paths.items()}}
    out = run_dir(output_root, identity, 'core-sensitivity', f'min{primary_min_cnvs}cnvs', f'reps{reps}')
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get('status') == 'complete':
            for name, checksum in old['artifacts'].items():
                if not (out/name).exists() or file_hash(out/name) != checksum:
                    raise ValueError(f'Core sensitivity artifact changed: {name}')
            return out
    out.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, {'status': 'running', 'identity': identity})
    progress('Core sensitivity: loading observed profiles and degree-matched references')
    profiles = {n: np.asarray(prop.nps(s)) for n, s in seeds.items()}
    counts = target_counts(profiles, names, nodes, seeds, config)
    variants = [(mode, k) for mode in MODES for k in thresholds]
    masks = np.stack([counts[MODES.index(mode)] >= k for mode, k in variants])
    primary = variants.index(('both', primary_min_cnvs))
    membership = pd.DataFrame({'entrez': nodes, 'symbol': [symbols.get(int(g), str(g)) for g in nodes]})
    for mode, values in zip(MODES, counts):
        membership[f'n_cnvs_{mode}'] = values
    for (mode, k), mask in zip(variants, masks):
        membership[f'{mode}_min_{k}'] = mask
    _write_table(membership, out/'gene_membership.tsv')
    _write_table(pd.DataFrame([{'entity': n, 'raw_seeds': len(set(entities[n]['seeds'])),
        'retained_seeds': len(s), 'dropped_seeds': ','.join(map(str, sorted(set(entities[n]['seeds'])-set(s))))}
        for n, s in seeds.items()]), out/'seed_qc.tsv')
    signatures = {n: prop.signature(prop.seed_indices(seeds[n])) for n in names}
    references = {n: prop.reference(prop.seed_indices(seeds[n]))[:2] for n in names}
    blocks = []
    for start in range(0, reps, 25):
        stop = min(start+25, reps)
        path = out/f'draws-{start:06d}.npz'
        stamp = Path(str(path)+'.json')
        if not (path.exists() and stamp.exists()
                and json.loads(stamp.read_text())['sha256'] == file_hash(path)):
            sizes = []
            for rep in range(start, stop):
                rng = rng_for(config.propagation.seed, 'shared_core_joint', rep)
                draws = {t: seeds[t] for t in MODES[:2]}
                random_profiles = {t: profiles[t] for t in MODES[:2]}
                for name in names:
                    cols = np.concatenate([rng.choice(group, size=k, replace=False)
                                           for group, k in zip(prop.bins, signatures[name]) if k])
                    draws[name] = prop.nodes[cols]
                    mu, sd = references[name]
                    random_profiles[name] = (np.log(prop.heat(draws[name]))-mu)/sd
                random_counts = target_counts(random_profiles, names, nodes, draws, config)
                sizes.append([int((random_counts[MODES.index(mode)] >= k).sum())
                              for mode, k in variants])
            with atomic_path(path) as tmp:
                np.savez_compressed(tmp, sizes=np.asarray(sizes, dtype=np.int64))
            write_json(stamp, {'sha256': file_hash(path)})
        with np.load(path) as block:
            blocks.append(block['sizes'])
        progress(f'Core null: {stop}/{reps} joint draws')
    null = np.concatenate(blocks)
    with atomic_path(out/'null_sizes.npy') as tmp:
        np.save(tmp, null, allow_pickle=False)
    records = []
    for j, ((mode, k), mask) in enumerate(zip(variants, masks)):
        observed = int(mask.sum())
        intersection = int((mask & masks[primary]).sum())
        union = int((mask | masks[primary]).sum())
        records.append({'mode': mode, 'min_cnvs': k, 'is_primary': j == primary,
            'observed_genes': observed, 'null_mean': null[:, j].mean(),
            'null_low': np.quantile(null[:, j], .025), 'null_high': np.quantile(null[:, j], .975),
            'excess': observed-null[:, j].mean(),
            'p_empirical': (1+int((null[:, j] >= observed).sum()))/(reps+1),
            'p_resolution': 1/(reps+1), 'reps': reps, 'intersection_with_primary': intersection,
            'jaccard_with_primary': intersection/union if union else 1.,
            'genes': ','.join(membership.loc[mask, 'symbol'].astype(str))})
    results = pd.DataFrame(records)
    results['q_sensitivity'] = bh(results.p_empirical)
    _write_table(results, out/'core_size_tests.tsv')
    # All variant gene sets and pairwise Jaccard scores allow stability inspection.
    _write_table(pd.DataFrame([{'variant_a': f'{a[0]}_min_{a[1]}', 'variant_b': f'{b[0]}_min_{b[1]}',
        'jaccard': float((ma & mb).sum()/(ma | mb).sum()) if (ma | mb).any() else 1.}
        for a, ma in zip(variants, masks) for b, mb in zip(variants, masks)]), out/'gene_set_similarity.tsv')
    if paths:
        _enrich_variants(membership, masks, variants, seeds, names, config, paths, out)
    _plot(results, out)
    main = results.iloc[primary]
    lines = ['# Shared-core threshold sensitivity and conditional null', '',
        'Primary definition: both SCHEMA and GWAS for the same gene and the same CNVs.',
        f'Minimum CNVs: {primary_min_cnvs}; observed genes: {main.observed_genes}; '
        f'null mean: {main.null_mean:.3f}; empirical P: {main.p_empirical:.6g}; '
        f'BH Q across all {len(variants)} variants: {main.q_sensitivity:.6g}.', '',
        f'{reps} joint degree-matched CNV seed draws; fixed SCHEMA/GWAS profiles; base seed {config.propagation.seed}.',
        'Each draw rebuilds every core using the original individual and product thresholds.',
        'Either = count distinct CNVs qualifying with at least one target; both = count CNVs qualifying with both.',
        'The primary definition was retained after inspection of the original core; this is an exploratory extension.',
        'P/Q concern core size conditional on fixed targets, not individual genes, edges, causality or enrichment.',
        'Independent CNV draws preserve degree profiles but not genomic contiguity or between-CNV seed overlap.',
        'Enrichment, when present, is descriptive after selection; q_all_variants_libraries corrects all variant/library terms.', '']
    (out/'summary.md').write_text('\n'.join(lines))
    write_json(manifest_path, {'status': 'complete', 'identity': identity,
        'null': 'independent degree-matched CNV seed draws, fixed SCHEMA and GWAS profiles',
        'endpoint': 'number of genes meeting exact core definition; no gene-level FDR',
        'analysis_status': 'exploratory extension after original core inspection',
        'artifacts': {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p != manifest_path}})
    return out


def _enrich_variants(membership, masks, variants, seeds, names, config, paths, out):
    definitions = {n: load_gmt(p) for n, p in paths.items()}
    frames = []
    coverage = []
    # Background eligibility uses real seeds and the identical per-CNV Boolean rule.
    eligibility = {t: np.stack([~np.isin(membership.entrez, list(set(seeds[n]) & set(seeds[t])))
                               for n in names]) for t in MODES[:2]}
    for (mode, k), mask in zip(variants, masks):
        eligible = np.ones(len(membership), dtype=bool)
        if config.overlap == 'remove':
            a, b = eligibility['schema'], eligibility['gwaspri']
            allowed = {'schema': a, 'gwaspri': b, 'either': a | b, 'both': a & b}[mode]
            eligible = allowed.sum(axis=0) >= k
        background = set(membership.loc[eligible, 'symbol'].astype(str))
        query = set(membership.loc[mask, 'symbol'].astype(str))
        if not query <= background:
            raise ValueError('Sensitivity query contains ineligible background genes')
        for library, terms in definitions.items():
            frame = ora(query, background, terms, min_size=5).rename(columns={'q': 'q_within_variant_library'})
            frames.append(frame.assign(mode=mode, min_cnvs=k, library=library))
            covered = set().union(*(genes for _, genes in terms.values())) if terms else set()
            coverage.append({'mode': mode, 'min_cnvs': k, 'library': library,
                'query_genes': len(query), 'query_annotated': len(query & covered),
                'background_genes': len(background), 'background_annotated': len(background & covered),
                'eligible_terms': len(frame)})
    table = pd.concat(frames, ignore_index=True)
    table['q_all_variants_libraries'] = bh(table.p)
    table['q_within_variant_all_libraries'] = np.nan
    for _, idx in table.groupby(['mode', 'min_cnvs']).groups.items():
        table.loc[idx, 'q_within_variant_all_libraries'] = bh(table.loc[idx, 'p'])
    _write_table(table, out/'enrichment_all_terms.tsv')
    _write_table(table.loc[table.q_all_variants_libraries < config.fdr], out/'enrichment_significant.tsv')
    _write_table(pd.DataFrame(coverage), out/'annotation_coverage.tsv')


def _plot(results, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for mode, ax in zip(MODES, axes.ravel()):
        frame = results[results['mode'] == mode]
        ax.fill_between(frame.min_cnvs, frame.null_low, frame.null_high, alpha=.2,
                        color='#4477AA', label='Null 2.5–97.5 percentiles')
        ax.plot(frame.min_cnvs, frame.null_mean, '--', color='#4477AA', label='Null mean')
        ax.plot(frame.min_cnvs, frame.observed_genes, 'o-', color='#AA3377', label='Observed')
        ax.set(title={'schema': 'SCHEMA', 'gwaspri': 'GWAS', 'either': 'GWAS or SCHEMA',
                      'both': 'GWAS and SCHEMA'}[mode], xlabel='Minimum distinct CNVs', ylabel='Selected genes')
        ax.set_xticks(frame.min_cnvs)
        ax.legend(fontsize=8)
    fig.suptitle('Core-size sensitivity · fixed SCZ profiles, degree-matched CNV seed null')
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'core_sensitivity.{suffix}', dpi=180)
    plt.close(fig)


def main():
    import argparse
    import matplotlib
    from cnv_overlap.inputs import load_inputs
    from cnv_overlap.pipeline import AnalysisConfig
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cnv_disorder_overlap.json')
    parser.add_argument('--reps', type=int, default=None)
    parser.add_argument('--thresholds', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--skip-enrichment', action='store_true')
    args = parser.parse_args()
    spec, root, inputs, _ = load_inputs(args.config)
    out = run_core_sensitivity(inputs, config=AnalysisConfig.from_dict(spec['analysis']),
        cache_dir=root/spec['cache_dir'], output_root=root/'results/scz_core_sensitivity',
        thresholds=args.thresholds, reps=args.reps,
        annotation_dir=None if args.skip_enrichment else root/'input/annotations')
    print(out)


if __name__ == '__main__':
    main()
