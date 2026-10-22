"""Separate CNV x SCHEMA and CNV x GWAS Reactome tests on the full interactome."""
from cnv_overlap.results import analysis_from_source, resolve_artifact, finalize_manifest, resolve_path
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from cnv_overlap.pipeline import AnalysisConfig
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import file_hash, run_dir, software, write_json
from cnv_overlap.statistics import bh, hit_mask, rng_for, tail
from scz_shared_network import load_inputs
from scz_restricted_core_null import sampling_plan, sample_joint, landscape_heat
from scz_module_inference import module_tests

ROOT = Path(__file__).resolve().parents[1]


def pathway_membership(path, genes, minimum=5):
    """Use all annotated pathways with >=minimum landscape genes; no CNV selection."""
    rows, masks = [], []
    for line in Path(path).read_text().splitlines():
        name, stable_id, *members = line.split('\t')
        mask = genes.symbol.astype(str).isin(set(members)).to_numpy()
        rows.append({'pathway': stable_id, 'name': name, 'annotation_genes': len(set(members)),
                     'landscape_genes': int(mask.sum()), 'tested': bool(mask.sum() >= minimum),
                     'genes': ','.join(genes.loc[mask, 'symbol'].astype(str))})
        if mask.sum() >= minimum:
            masks.append(mask)
    table = pd.DataFrame(rows)
    if table.pathway.duplicated().any():
        raise ValueError('Duplicate Reactome pathway IDs')
    return table, np.asarray(masks, dtype=bool).reshape(-1, len(genes))


def pair_share(counts, totals):
    share = np.divide(counts, totals[..., None], out=np.zeros_like(counts, dtype=float),
                      where=totals[..., None] > 0)
    n = counts.shape[-2]
    return np.maximum(0, (share.sum(axis=-2)**2-(share**2).sum(axis=-2))/(n*(n-1)))


def run(source, reps=5000, minimum=5):
    source = Path(source).resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete landscape source')
    files = ['all_gene_membership.tsv', '01_schema_gwas_landscape/gene_modules.tsv']
    for f in files:
        if file_hash(resolve_artifact(source, f)) != manifest['artifacts'][manifest.get('artifact_paths', {}).get(f, f)]:
            raise ValueError(f'Source artifact changed: {f}')
    settings = manifest['identity']['settings']
    config = AnalysisConfig.from_dict(manifest['identity']['config'])
    annotation = ROOT/'input/annotations/ReactomePathways.gmt'
    identity = {'source': str(source), 'source_manifest': file_hash(source/'manifest.json'),
                'annotation': file_hash(annotation), 'reps': reps, 'min_network_genes': minimum, 'universe': 'full_interactome', 'branches': ['schema', 'gwaspri'],
                'preserve_target_overlap': True, 'software': software(),
                'code': {p.name: file_hash(p) for p in [Path(__file__),
                    ROOT/'scripts/scz_module_inference.py', ROOT/'scripts/scz_reactome_convergence.py', ROOT/'scripts/scz_restricted_core_null.py',
                    ROOT/'scripts/scz_shared_network.py', ROOT/'scripts/network_inputs.py']},
                'package_code': {p.name: file_hash(p) for p in sorted((ROOT/'src/cnv_overlap').glob('*.py'))}}
    out = run_dir(analysis_from_source(source, 'rare_common_pathways'), identity, 'reactome')
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        if old.get('status') == 'complete':
            for f, h in old['artifacts'].items():
                if file_hash(out/f) != h:
                    raise ValueError(f'Artifact changed: {f}')
            return out
    out.mkdir(parents=True, exist_ok=True)
    finalize_manifest(out/'manifest.json', {'status': 'running', 'identity': identity})
    spec, root, inputs = load_inputs(settings['network'], settings.get('min_seeds', 3),
        settings.get('convergence_config'), settings.get('disorder_config'))
    A, nodes, symbols, entities, *_ = inputs
    if entities != manifest['identity']['entities']:
        raise ValueError('Entities differ from frozen source')
    prop = Propagator(A, nodes, root/spec['cache_dir'], config.propagation)
    seeds = {n: sorted(set(e['seeds']) & prop.pos.keys()) for n, e in entities.items()}
    profiles = {n: np.asarray(prop.nps(s)) for n, s in seeds.items()}
    frozen = pd.read_csv(resolve_artifact(source, files[0]), sep='\t').set_index('entrez')
    if set(frozen.index) != set(prop.nodes):
        raise ValueError('Network universe differs')
    frozen = frozen.loc[prop.nodes]
    for n in profiles:
        if not np.allclose(profiles[n], frozen[f'nps_{n}'], rtol=1e-10, atol=1e-10):
            raise ValueError(f'Profile differs: {n}')
    # A pair hit requires target NPS > threshold. Restricting computed rows to
    # this union is exact; pathway eligibility and denominators use the FULL graph.
    mask = ((profiles['schema'] > config.thresholds.individual)
            | (profiles['gwaspri'] > config.thresholds.individual))
    idx = np.flatnonzero(mask)
    genes = frozen.reset_index()
    universe, full_membership = pathway_membership(annotation, genes, minimum)
    universe = universe.rename(columns={'landscape_genes': 'network_genes'})
    universe.to_csv(resolve_path(out/'pathway_universe.tsv'), sep='\t', index=False)
    selected = universe[universe.tested].reset_index(drop=True)
    if selected.empty:
        raise ValueError('No eligible pathways')
    pathway_sizes = full_membership.sum(axis=1)
    selected['intersection_group'] = [file_hash_for_mask(m) for m in full_membership]
    for target in ('schema', 'gwaspri'):
        selected[f'{target}_eligible_genes'] = full_membership[:, profiles[target] > config.thresholds.individual].sum(axis=1)
    selected.to_csv(resolve_path(out/'tested_pathways.tsv'), sep='\t', index=False)
    membership = full_membership[:, idx]
    names = sorted(n for n in entities if entities[n]['kind'] == 'cnv')
    used = {n: seeds[n] for n in [*names, 'schema', 'gwaspri']}
    groups, balance = sampling_plan(prop, used, names)
    targets = set(seeds['schema']) | set(seeds['gwaspri'])
    fixed = {n: set(seeds[n]) & targets for n in names}
    random_seeds = {n: sorted(set(seeds[n])-fixed[n]) for n in names}
    groups = [(original[~np.isin(original, list(targets))], pool) for original, pool in groups]
    balance['required_unique_cnv_genes'] = [len(original) for original, _ in groups]
    balance.to_csv(resolve_path(out/'degree_pool_qc.tsv'), sep='\t', index=False)
    pd.DataFrame([{'cnv': n, 'fixed_target_overlap': len(fixed[n]),
                   'randomized_seeds': len(random_seeds[n]), 'fixed_genes': ','.join(map(str, sorted(fixed[n])))}
                  for n in names]).to_csv(resolve_path(out/'target_overlap_qc.tsv'), sep='\t', index=False)
    excluded = set().union(*map(set, used.values()))
    pd.DataFrame({'entrez': sorted(excluded)}).to_csv(resolve_path(out/'excluded_random_seeds.tsv'), sep='\t', index=False)
    refs = {n: tuple(v[idx] for v in prop.reference(prop.seed_indices(seeds[n]))[:2]) for n in names}
    kernel = np.asarray(prop.H[idx])
    sparse = csr_matrix(membership.astype(np.int32))
    modes = ('schema', 'gwaspri')
    observed = {mode: [] for mode in modes}
    for n in names:
        for target in modes:
            observed[target].append(hit_mask(profiles[n][idx], profiles[target][idx], config.thresholds))
    counts = {mode: np.zeros((reps, len(names), len(selected)), dtype=np.int16) for mode in modes}
    totals = {mode: np.zeros((reps, len(names)), dtype=np.int16) for mode in modes}
    packed = {mode: np.zeros((reps, len(names), (len(idx)+7)//8), dtype=np.uint8) for mode in modes}
    for rep in range(reps):
        draws = sample_joint(prop, random_seeds, names, groups,
                             rng_for(config.propagation.seed, 'landscape_matched_seeds', rep))
        if excluded & set().union(*(set(s) for s in draws.values())):
            raise AssertionError('Forbidden random seed')
        reaches = {mode: [] for mode in modes}
        for n in names:
            draw = sorted(set(draws[n]) | fixed[n])
            mu, sd = refs[n]
            z = (np.log(landscape_heat(kernel, prop.seed_indices(draw))) - mu)/sd
            for target in modes:
                reaches[target].append(hit_mask(z, profiles[target][idx], config.thresholds))
        for mode in modes:
            reach = np.array(reaches[mode], dtype=np.int32)
            counts[mode][rep] = (sparse @ reach.T).T
            totals[mode][rep] = reach.sum(axis=1)
            packed[mode][rep] = np.packbits(reach.astype(bool), axis=-1)
        if (rep+1) % 500 == 0:
            print(f'{settings["network"]}: {rep+1}/{reps}', flush=True)
    frames = []
    for mode in modes:
        obs_reach = np.array(observed[mode], dtype=np.int32)
        obs_counts = (sparse @ obs_reach.T).T
        obs_totals = obs_reach.sum(axis=1)
        table = module_tests(obs_counts, counts[mode], obs_totals, totals[mode], pathway_sizes,
                             selected.pathway.tolist(), names, mode).rename(columns={'module': 'pathway', 'module_genes': 'network_genes'})
        # Across-CNV relative concentration complements absolute coverage; same null draws.
        obs_share = pair_share(obs_counts, obs_totals)
        null_share = pair_share(counts[mode], totals[mode])
        extra = pd.DataFrame([{'mode': mode, 'endpoint': 'pair_branch_share', 'pathway': p,
                              'network_genes': int(pathway_sizes[j]), 'cnv': 'all',
                              **tail(obs_share[j], null_share[:, j])} for j, p in enumerate(selected.pathway)])
        extra['q'] = bh(extra.p)
        order = np.argsort(extra.p.to_numpy(), kind='stable')
        holm = np.empty(len(extra))
        holm[order] = np.minimum(1, np.maximum.accumulate(extra.p.to_numpy()[order]*np.arange(len(extra), 0, -1)))
        extra['p_holm'] = holm
        extra['family'] = f'{mode}:pair_branch_share'
        extra['family_size'] = len(extra)
        frames.append(pd.concat([table, extra], ignore_index=True))
        np.savez_compressed(resolve_path(out/f'pathway_draws_{mode}.npz'), counts=counts[mode], totals=totals[mode],
            observed_counts=obs_counts, observed_totals=obs_totals, pathways=selected.pathway.to_numpy(dtype=str),
            cnvs=names, gene_order=prop.nodes[idx], observed_reach=obs_reach.astype(bool), packed_reach=packed[mode])
    tests = pd.concat(frames, ignore_index=True).merge(selected[['pathway', 'name', 'intersection_group']], on='pathway')
    # BY adds arbitrary-dependence FDR correction for overlapping/hierarchical pathways.
    tests['q_by'] = np.minimum(1, tests.q * tests.family_size.map(lambda n: np.sum(1/np.arange(1, int(n)+1))))
    # Selection from either branch is corrected jointly over both branch families.
    for endpoint, indexes in tests.groupby('endpoint').groups.items():
        tests.loc[indexes, 'q_both_branches'] = bh(tests.loc[indexes, 'p'])
    tests.to_csv(resolve_path(out/'pathway_tests.tsv'), sep='\t', index=False)
    tests.groupby(['mode', 'endpoint']).agg(tests=('p', 'size'), min_p=('p', 'min'), min_q=('q', 'min'),
        bh_discoveries=('q', lambda x: int((x < .05).sum())),
        holm_discoveries=('p_holm', lambda x: int((x < .05).sum())),
        by_discoveries=('q_by', lambda x: int((x < .05).sum()))).to_csv(resolve_path(out/'summary.tsv'), sep='\t')
    comparison = tests[tests.endpoint == 'pair_coverage'].pivot(index='pathway', columns='mode',
        values=['observed', 'null_mean', 'excess', 'p', 'q', 'q_both_branches'])
    comparison.columns = ['_'.join(c) for c in comparison.columns]
    comparison = comparison.reset_index().merge(selected[['pathway', 'name', 'network_genes']], on='pathway')
    a = comparison.q_both_branches_schema < .05
    b = comparison.q_both_branches_gwaspri < .05
    comparison['detection'] = np.select([a & b, a, b], ['both', 'SCHEMA detected', 'GWAS detected'], default='neither')
    comparison.to_csv(resolve_path(out/'branch_comparison.tsv'), sep='\t', index=False)
    (out/'README.md').write_text('# Separate rare/common Reactome convergence\n\n'
        'Full tested interactome universe; same pathways with >=5 network genes in both branches. '
        'CNV x SCHEMA and CNV x GWAS pair masks are evaluated separately. '
        'Target-positive union rows are an exact computational restriction, not the shared landscape. '
        'Pathway coverage uses full network pathway size; branch share uses total pair-hit genes. '
        'Main endpoint: pair_coverage. Pair_branch_share is secondary. '
        'q_both_branches corrects each endpoint across both targets. Branch-specific BH, BY and Holm are also supplied. '
        'Detection in only one branch does not establish a significant between-branch difference or specificity. '
        'Null draws preserve direct overlap with either target and all between-CNV overlaps. '
        'No genomic contiguity matching or direct rare/common contrast is performed.\n')
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'artifacts': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.name != 'manifest.json'}})
    return out


def file_hash_for_mask(mask):
    import hashlib
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()[:16]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--reps', type=int, default=5000)
    args = parser.parse_args()
    if args.reps < 2:
        parser.error('Require >=2 draws')
    print(run(args.source_run, args.reps))
