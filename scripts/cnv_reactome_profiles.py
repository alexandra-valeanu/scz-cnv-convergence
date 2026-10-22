"""CNV-first Reactome coverage, matched-seed calibration, and fixed SCZ overlays."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cnv_overlap.pipeline import AnalysisConfig
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import atomic_path, file_hash, run_dir, write_json
from cnv_overlap.results import (analysis_from_source, finalize_manifest, resolve_artifact,
                                 verify_run)
from cnv_overlap.statistics import bh, hit_mask, rng_for
from scz_restricted_core_null import sampling_plan, sample_joint, landscape_heat
from scz_shared_network import load_inputs

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ('coverage', 'mean_positive_nps', 'outside_coverage')
CATEGORIES = ('landscape', 'both_below_product', 'schema_only', 'gwas_only', 'neither')


def pathway_universe(annotation, symbols, minimum=5):
    """Eligibility uses the full tested network, never CNV/SCZ scores."""
    if minimum < 1 or len(set(symbols)) != len(symbols):
        raise ValueError('Require minimum >= 1 and unique network symbols')
    positions = {s: i for i, s in enumerate(symbols)}
    rows, rr, cc = [], [], []
    tested = 0
    for line in Path(annotation).read_text().splitlines():
        fields = line.split('\t')
        if len(fields) < 3:
            raise ValueError('Malformed Reactome GMT row')
        name, stable_id, *genes = fields
        members = set(genes) & positions.keys()
        eligible = len(members) >= minimum
        rows.append(dict(pathway=stable_id, name=name, annotation_genes=len(set(genes)),
                         network_genes=len(members), tested=eligible))
        if eligible:
            cols = sorted(positions[s] for s in members)
            rr.extend([tested]*len(cols))
            cc.extend(cols)
            tested += 1
    table = pd.DataFrame(rows)
    if table.empty or table.pathway.duplicated().any():
        raise ValueError('Empty annotation or duplicate Reactome IDs')
    if not tested:
        raise ValueError('No eligible Reactome pathways')
    matrix = csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(tested, len(symbols)))
    return table, matrix


def overlay_categories(schema, gwas, landscape, threshold):
    """Keep genes passing both individual cutoffs but failing the product distinct."""
    a, b = schema > threshold, gwas > threshold
    return np.select([landscape, a & b, a, b], CATEGORIES[:4], default='neither')


def score_profiles(z, membership, landscape, threshold):
    """CNV x pathway measures; zero outside denominator is undefined, not zero."""
    z = np.atleast_2d(z)
    reached = (z > threshold).astype(float)
    size = np.asarray(membership.sum(axis=1)).ravel()
    outside = membership.multiply(~landscape).tocsr()
    outside_size = np.asarray(outside.sum(axis=1)).ravel()
    counts = (membership @ reached.T).T
    outside_counts = (outside @ reached.T).T
    return np.stack([counts/size, (membership @ np.maximum(z, 0).T).T/size,
                     np.divide(outside_counts, outside_size,
                               out=np.full_like(outside_counts, np.nan), where=outside_size > 0)])


def pair_coverage(coverage):
    """Mean coverage product across distinct CNVs; different genes can contribute."""
    n = coverage.shape[0]
    if n < 2:
        return np.full(coverage.shape[1], np.nan)
    return np.maximum(0, (coverage.sum(axis=0)**2 - (coverage**2).sum(axis=0))/(n*(n-1)))


def calibration(observed, total, square, ge, reps):
    mean = total/reps
    sd = np.sqrt(np.maximum(0, square/reps - mean**2))
    p = (1 + ge)/(reps + 1)
    return mean, sd, np.where(np.isfinite(observed), p, np.nan)


def save(frame, path):
    frame.to_csv(path, sep='\t', index=False)


def make_figures(tests, overlay, out, top=30):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main = tests[tests.endpoint == 'coverage']
    order = (main.groupby('pathway').agg(p=('p', 'min'), excess=('excess', 'max'))
             .sort_values(['p', 'excess'], ascending=[True, False]).head(top).index)
    # Exact duplicate network memberships are displayed once; all remain tested.
    groups = main.drop_duplicates('pathway').set_index('pathway').membership_group
    order = list(dict((groups[p], p) for p in reversed(order)).values())[::-1]
    values = main.pivot(index='pathway', columns='cnv', values='excess').loc[order]
    q = main.pivot(index='pathway', columns='cnv', values='q_bh').loc[order, values.columns]
    labels = main.drop_duplicates('pathway').set_index('pathway').name
    fig, ax = plt.subplots(figsize=(max(7, len(values.columns)*.8), max(4, len(order)*.3)))
    limit = max(.01, np.abs(values.to_numpy()).max())
    im = ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_yticks(range(len(order)), [labels[p][:75] for p in order], fontsize=8)
    ax.set_xticks(range(len(values.columns)), values.columns, rotation=60, ha='right')
    for i, j in zip(*np.where(q.to_numpy() < .05)):
        ax.text(j, i, '*', ha='center', va='center')
    ax.set_title('CNV–Reactome coverage above matched-seed expectation\n* BH q < 0.05 across all CNV × pathway cells')
    fig.colorbar(im, ax=ax, label='Observed − null mean coverage')
    fig.savefig(out/'cnv_pathway_heatmap.png', dpi=180, bbox_inches='tight')
    fig.savefig(out/'cnv_pathway_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)
    save(pd.DataFrame({'pathway': order, 'name': [labels[p] for p in order]}), out/'displayed_pathways.tsv')
    # Each panel shows the SCZ category of the reached genes, separately per CNV.
    fig, axes = plt.subplots(1, len(values.columns), figsize=(max(9, 2*len(values.columns)), max(4, len(order)*.3)), squeeze=False)
    colors = ['#7b3294', '#c2a5cf', '#2166ac', '#b35806', '#cccccc']
    for ax, cnv in zip(axes[0], values.columns):
        d = overlay[overlay.cnv == cnv].set_index('pathway').loc[order]
        left = np.zeros(len(order))
        for category, color in zip(CATEGORIES, colors):
            fraction = d[category].to_numpy()/np.maximum(1, d.reached_genes.to_numpy())
            ax.barh(range(len(order)), fraction, left=left, color=color, label=category)
            left += fraction
        ax.set_title(cnv, fontsize=8)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_yticks(range(len(order)), [labels[p][:65] for p in order] if ax is axes[0, 0] else [], fontsize=8)
        ax.set_xlabel('Share of reached genes', fontsize=8)
    axes[0, -1].legend(bbox_to_anchor=(1.03, 1), fontsize=8)
    fig.suptitle('SCZ overlay: fixed target proximity categories (blank = no reached genes)')
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'scz_overlay.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_profiles(prop, profiles, entities, symbols, landscape, *, config, annotation,
                 output_root, reps=5000, minimum=5, source=None, block_size=250,
                 progress=print):
    """Stream resumable joint draws; never materialize draw x CNV x pathway arrays."""
    if reps < 2 or block_size < 1:
        raise ValueError('Require reps >= 2 and positive block size')
    names = sorted(n for n, e in entities.items() if e['kind'] == 'cnv')
    if not names:
        raise ValueError('No CNVs')
    symbols = [symbols.get(int(g), str(g)) for g in prop.nodes]
    landscape = np.asarray(landscape, dtype=bool)
    if landscape.shape != (len(prop.nodes),):
        raise ValueError('Landscape must align with network nodes')
    if any(np.asarray(profiles[n]).shape != landscape.shape or not np.isfinite(profiles[n]).all()
           for n in [*names, 'schema', 'gwaspri']):
        raise ValueError('Profiles must be finite and aligned with network nodes')
    seeds = {n: sorted(set(entities[n]['seeds']) & prop.pos.keys()) for n in names}
    # CNV-first null: excludes real CNV seeds only; all seed positions randomized.
    # SCZ genes do not influence sampling or pathway eligibility.
    groups, balance = sampling_plan(prop, seeds, names)
    universe, matrix = pathway_universe(annotation, symbols, minimum)
    selected = universe[universe.tested].reset_index(drop=True)
    selected['membership_group'] = [','.join(map(str, matrix[i].indices)) for i in range(len(selected))]
    import hashlib
    selected['membership_group'] = selected.membership_group.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16])
    source = Path(source).resolve() if source else None
    identity = dict(algorithm='cnv_reactome_profiles_v1', propagation=prop.metadata,
                    config=asdict(config), entities=entities, symbols=symbols,
                    landscape=prop.nodes[landscape].tolist(), annotation_sha256=file_hash(annotation),
                    reps=reps, minimum=minimum, block_size=block_size,
                    null='joint degree-matched CNV seed replacement; exclude CNV seeds only; no fixed SCZ overlap',
                    source=str(source) if source else None,
                    source_manifest=file_hash(source/'manifest.json') if source else None,
                    code={str(p.relative_to(ROOT)): file_hash(p) for p in
                          [Path(__file__), ROOT/'scripts/scz_restricted_core_null.py',
                           *sorted((ROOT/'src/cnv_overlap').glob('*.py'))]})
    out = run_dir(output_root, identity, 'cnv-reactome', f'reps{reps}')
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('status') == 'complete':
        verify_run(out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    write_json(manifest, dict(status='running', identity=identity))
    save(universe, out/'pathway_universe.tsv')
    selected['outside_genes'] = np.asarray(matrix.multiply(~landscape).sum(axis=1)).ravel().astype(int)
    save(selected, out/'tested_pathways.tsv')
    save(balance, out/'degree_pool_qc.tsv')
    save(pd.DataFrame([dict(cnv=n, raw_seeds=len(set(entities[n]['seeds'])), retained_seeds=len(seeds[n]),
                            genes=','.join(map(str, seeds[n]))) for n in names]), out/'seed_qc.tsv')
    threshold = config.thresholds.individual
    z = np.stack([profiles[n] for n in names])
    observed = score_profiles(z, matrix, landscape, threshold)
    pair_obs = pair_coverage(observed[0])
    refs = {n: prop.reference(prop.seed_indices(seeds[n]))[:2] for n in names}
    totals = np.zeros_like(observed)
    squares = np.zeros_like(observed)
    ge = np.zeros_like(observed, dtype=np.int64)
    pair_total, pair_square = np.zeros_like(pair_obs), np.zeros_like(pair_obs)
    pair_ge = np.zeros_like(pair_obs, dtype=np.int64)
    checkpoint_dir = out/'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    kernel = prop.H
    for start in range(0, reps, block_size):
        stop = min(reps, start + block_size)
        path = checkpoint_dir/f'{start:06d}.npz'
        stamp = path.with_suffix('.json')
        if not (path.exists() and stamp.exists() and json.loads(stamp.read_text())['sha256'] == file_hash(path)):
            t, s, g = np.zeros_like(observed), np.zeros_like(observed), np.zeros_like(ge)
            pt, ps, pg = np.zeros_like(pair_obs), np.zeros_like(pair_obs), np.zeros_like(pair_ge)
            saved_seeds = []
            for rep in range(start, stop):
                draws = sample_joint(prop, seeds, names, groups,
                                     rng_for(config.propagation.seed, 'cnv_reactome_joint', rep))
                random_z = []
                for n in names:
                    mu, sd = refs[n]
                    random_z.append((np.log(landscape_heat(kernel, prop.seed_indices(draws[n]))) - mu)/sd)
                scores = score_profiles(np.array(random_z), matrix, landscape, threshold)
                pair = pair_coverage(scores[0])
                t += np.nan_to_num(scores)
                s += np.nan_to_num(scores)**2
                g += scores >= observed - 1e-12
                pt += np.nan_to_num(pair)
                ps += np.nan_to_num(pair)**2
                pg += pair >= pair_obs - 1e-12
                saved_seeds.append([gene for n in names for gene in draws[n]])
            with atomic_path(path) as temporary:
                np.savez_compressed(temporary, total=t, square=s, ge=g,
                                    pair_total=pt, pair_square=ps, pair_ge=pg, seeds=saved_seeds)
            write_json(stamp, {'sha256': file_hash(path), 'start': start, 'stop': stop})
        with np.load(path) as block:
            totals += block['total']; squares += block['square']; ge += block['ge']
            pair_total += block['pair_total']; pair_square += block['pair_square']; pair_ge += block['pair_ge']
        progress(f'CNV Reactome null: {stop}/{reps}')
    mean, sd, p = calibration(observed, totals, squares, ge, reps)
    frames = []
    for k, endpoint in enumerate(ENDPOINTS):
        rows = []
        for i, n in enumerate(names):
            for j, pathway in enumerate(selected.pathway):
                valid = np.isfinite(observed[k, i, j])
                rows.append(dict(endpoint=endpoint, cnv=n, pathway=pathway,
                                 observed=observed[k, i, j], null_mean=mean[k, i, j] if valid else np.nan,
                                 null_sd=sd[k, i, j] if valid else np.nan, p=p[k, i, j],
                                 status='tested' if valid else 'no_outside_genes'))
        frame = pd.DataFrame(rows)
        valid = frame.p.notna()
        frame['q_bh'] = np.nan
        frame.loc[valid, 'q_bh'] = bh(frame.loc[valid, 'p'])
        frame['family_size'] = int(valid.sum())
        frames.append(frame)
    tests = pd.concat(frames, ignore_index=True)
    tests['q_by'] = np.minimum(1, tests.q_bh * tests.family_size.map({n: np.sum(1/np.arange(1, n+1)) for n in tests.family_size.unique()}))
    tests['excess'] = tests.observed - tests.null_mean
    tests['p_resolution'] = 1/(reps+1)
    tests = tests.merge(selected, on='pathway')
    save(tests, out/'cnv_pathway_tests.tsv')
    pm, ps, pp = calibration(pair_obs, pair_total, pair_square, pair_ge, reps)
    shared = selected.copy()
    shared['observed'] = pair_obs; shared['null_mean'] = pm; shared['null_sd'] = ps
    shared['excess'] = pair_obs-pm; shared['p'] = pp; shared['q_bh'] = bh(pp)
    shared['q_by'] = np.minimum(1, shared.q_bh * np.sum(1/np.arange(1, len(shared)+1)))
    shared['cnvs_with_any_reach'] = (observed[0] > 0).sum(axis=0)
    save(shared, out/'shared_pathway_tests.tsv')
    categories = overlay_categories(profiles['schema'], profiles['gwaspri'], landscape, threshold)
    gene_table = pd.DataFrame(dict(entrez=prop.nodes, symbol=symbols, scz_category=categories,
                                   in_scz_landscape=landscape, nps_schema=profiles['schema'],
                                   nps_gwaspri=profiles['gwaspri']))
    for n in [*names, 'schema', 'gwaspri']:
        gene_table[f'seed_{n}'] = np.isin(prop.nodes, entities[n]['seeds'])
    for n in names:
        gene_table[f'nps_{n}'] = profiles[n]
    save(gene_table, out/'gene_profiles.tsv.gz')
    rows, contributions = [], []
    for i, n in enumerate(names):
        for j, pathway in enumerate(selected.pathway):
            idx = matrix[j].indices
            reached = idx[z[i, idx] > threshold]
            rows.append(dict(cnv=n, pathway=pathway, reached_genes=len(reached),
                             **{c: int((categories[reached] == c).sum()) for c in CATEGORIES}))
            contributions.extend(dict(cnv=n, pathway=pathway, entrez=int(prop.nodes[g]), symbol=symbols[g],
                                      nps=z[i, g], scz_category=categories[g],
                                      own_cnv_seed=int(prop.nodes[g]) in seeds[n],
                                      schema_seed=int(prop.nodes[g]) in entities['schema']['seeds'],
                                      gwas_seed=int(prop.nodes[g]) in entities['gwaspri']['seeds']) for g in reached)
    overlay = pd.DataFrame(rows)
    save(overlay, out/'scz_overlay.tsv')
    save(pd.DataFrame(contributions, columns=['cnv', 'pathway', 'entrez', 'symbol', 'nps', 'scz_category',
                                             'own_cnv_seed', 'schema_seed', 'gwas_seed']), out/'contributing_genes.tsv.gz')
    make_figures(tests, overlay, out)
    discoveries = int(((tests.endpoint == 'coverage') & (tests.q_bh < .05)).sum())
    (out/'README.md').write_text(f'''# CNV-first Reactome profiles

{len(universe)} annotated pathways; {len(selected)} eligible with >= {minimum} full-network genes.
{len(names)} CNVs; {reps} joint draws; empirical P resolution {1/(reps+1):.6g}.
Primary coverage: {discoveries} CNV x pathway cells at BH q < 0.05.

Coverage is the fraction of pathway genes with CNV NPS > {threshold}; all own seeds remain eligible.
Mean positive NPS averages max(NPS, 0) over ALL pathway genes, without thresholding.
Outside coverage uses only pathway genes outside the fixed SCZ landscape in both observed and null scores.
Pathways without outside genes have undefined outside coverage and are not tested for that endpoint.
BH corrects all CNV x pathway cells separately for each endpoint. BY is a dependence-robust sensitivity.
Shared-pathway mean pairwise coverage products have their own across-pathway correction family.
These families must not be pooled as independent discoveries. Reactome pathways overlap and are hierarchical.

The joint null preserves each CNV seed count, degree-bin profile and all between-CNV overlaps.
All CNV seeds are randomized; replacement pools exclude the real CNV seed union only.
SCZ seeds do not control sampling, pathway eligibility, or primary scoring. Their profiles stay fixed for the overlay.
The null does not preserve genomic contiguity, dosage direction, or condition on total reached genes.
NPS uses the original unrestricted degree-matched references, including for random seeds.

The heatmap shows up to 30 pathways ranked by minimum primary P then maximum excess; stars use global BH.
Exact duplicate network memberships are shown once but retained as separate tests in complete tables.
The overlay has five exclusive categories: landscape, both individual target cutoffs but below product,
SCHEMA only, GWAS only, and neither. It measures overlap, not pathway-to-landscape proximity or SCZ specificity.
Absent landscape membership does not establish absence of SCZ biology. Scores do not imply activation or suppression.

Checkpoint aggregates and sampled seeds support exact resume and independent reconstruction.
Seed order within each draw is sorted CNV order, with lengths in seed_qc.tsv.
''')
    finalize_manifest(manifest, dict(status='complete', identity=identity, analysis='cnv_reactome_profiles'))
    return out


def run(source, *, reps=5000, minimum=5, annotation=None, output_root=None, block_size=250):
    source = Path(source).resolve()
    manifest = verify_run(source)
    settings = manifest['identity']['settings']
    config = AnalysisConfig.from_dict(manifest['identity']['config'])
    spec, root, inputs = load_inputs(settings['network'], settings.get('min_seeds', 3),
                                    settings.get('convergence_config'), settings.get('disorder_config'))
    A, nodes, symbols, entities, *_ = inputs
    if entities != manifest['identity']['entities']:
        raise ValueError('Entities differ from the frozen landscape source')
    prop = Propagator(A, nodes, root/spec['cache_dir'], config.propagation)
    frozen = pd.read_csv(resolve_artifact(source, 'all_gene_membership.tsv'), sep='\t').set_index('entrez')
    if frozen.index.duplicated().any() or set(frozen.index) != set(prop.nodes):
        raise ValueError('Network universe differs from the landscape source')
    frozen = frozen.loc[prop.nodes]
    profiles = {n: prop.nps(sorted(set(e['seeds']) & prop.pos.keys())) for n, e in entities.items()}
    for n in profiles:
        if not np.allclose(profiles[n], frozen[f'nps_{n}'], rtol=1e-10, atol=1e-10):
            raise ValueError(f'Profile differs from source: {n}')
    landscape = hit_mask(profiles['schema'], profiles['gwaspri'], config.thresholds)
    if config.overlap == 'remove':
        landscape &= ~np.isin(prop.nodes, list(set(entities['schema']['seeds']) & set(entities['gwaspri']['seeds'])))
    if not np.array_equal(landscape, frozen.in_scz_landscape.to_numpy()):
        raise ValueError('Landscape differs from source')
    return run_profiles(prop, profiles, entities, symbols, landscape, config=config,
                        annotation=annotation or ROOT/'input/annotations/ReactomePathways.gmt',
                        output_root=output_root or analysis_from_source(source, 'cnv_reactome_profiles'),
                        reps=reps, minimum=minimum, source=source, block_size=block_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path, help='Completed frozen SCZ landscape run')
    parser.add_argument('--reps', type=int, default=5000)
    parser.add_argument('--minimum', type=int, default=5, help='Minimum full-network genes per pathway')
    parser.add_argument('--annotation', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--block-size', type=int, default=250)
    args = parser.parse_args()
    print(run(args.source_run, reps=args.reps, minimum=args.minimum, annotation=args.annotation,
              output_root=args.output_root, block_size=args.block_size))
