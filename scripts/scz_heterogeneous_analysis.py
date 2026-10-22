"""Calibrated pathway recurrence and descriptive CNV mechanism profiles.

Uses verified reach masks from completed core analyses.
"""
from cnv_overlap.results import analysis_from_source, finalize_manifest, resolve_path
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from cnv_overlap.provenance import digest, file_hash, run_dir, software, write_json
from scz_heterogeneous_stats import (adjusted_p, binomial_interval, calibrate, calibration_p,
    count_histogram, inject_pathway_signal, jaccard, recurrence_p, term_scores)


def log(message):
    print(message, flush=True)


def table(frame, path):
    frame.to_csv(path, sep='\t', index=False)


def resolved(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def verified_manifest(directory, expected_hash=None):
    directory = Path(directory)
    path = directory / 'manifest.json'
    if expected_hash is not None and file_hash(path) != expected_hash:
        raise ValueError(f'Manifest identity changed: {path}')
    data = json.loads(path.read_text())
    if data.get('status') != 'complete' or not data.get('artifacts'):
        raise ValueError(f'Need a complete, manifested source: {directory}')
    for name, expected in data['artifacts'].items():
        artifact = directory / name
        if not artifact.is_file() or file_hash(artifact) != expected:
            raise ValueError(f'Artifact missing or changed: {artifact}')
    return data


def validate_config(cfg, n_cnvs=None):
    for name in ('calibration_draws', 'evaluation_draws', 'chunk_size'):
        if type(cfg[name]) is not int or cfg[name] < 2:
            raise ValueError(f'{name} must be an integer >= 2')
    for name in ('support_alpha', 'fdr'):
        if not 0 < cfg[name] < 1:
            raise ValueError(f'{name} must be between zero and one')
    if not 0 <= cfg['borderline_margin'] < 1:
        raise ValueError('Invalid borderline margin')
    d = cfg['diagnostics']
    if (type(d['pseudo_datasets']) is not int or
            not 2 <= d['pseudo_datasets'] <= cfg['evaluation_draws'] - 2 or
            type(d['power_replicates']) is not int or d['power_replicates'] < 2):
        raise ValueError('Need >= 2 pseudo-datasets, reference draws, and power replicates')
    if d['target_selection'] != 'median_size_per_family':
        raise ValueError('Only sizes-based median_size_per_family target selection is supported')
    if not d['affected_cnvs'] or any(type(k) is not int or k < 2 for k in d['affected_cnvs']):
        raise ValueError('Power scenarios must affect at least two CNVs')
    if n_cnvs is not None and max(d['affected_cnvs']) > n_cnvs:
        raise ValueError('Power scenario requests more CNVs than the input contains')
    if not d['swap_fractions'] or any(not 0 < f <= 1 for f in d['swap_fractions']):
        raise ValueError('Signal fractions must be in (0, 1]')
    if (type(cfg['reporting']['leading_terms_per_family']) is not int or
            cfg['reporting']['leading_terms_per_family'] < 1 or
            not 0 <= cfg['reporting']['redundancy_gene_jaccard'] <= 1):
        raise ValueError('Invalid reporting configuration')


def load_inputs(source, cfg):
    """Validate linked provenance and recover exact frozen term memberships.

The original pathway tables enumerate every member as SYMBOL(K). Reusing those
verified memberships avoids reinterpreting an updated annotation/symbol mapping.
Their annotation hashes are checked against the original GMT files as well.
"""
    source = Path(source)
    if source.parent.parent.name != cfg['network']:
        raise ValueError('Configured network and source analysis directory differ')
    parent = verified_manifest(source)
    if parent['identity']['algorithm'] != 'scz_core_analysis_v1':
        raise ValueError('Unsupported source analysis schema')
    null_dir = resolved(parent['null_dirs']['degree'])
    null = verified_manifest(null_dir, parent['identity']['nulls']['degree'])
    ident = null['identity']
    if (ident['algorithm'] != 'scz_core_null_v1' or ident['null_model'] != 'degree' or
            ident['excluded'] or ident['windows'] is not None):
        raise ValueError('Primary analysis requires the unexcluded degree null')
    if ident['code'] != file_hash(ROOT / 'scripts/scz_core_null.py'):
        raise ValueError('Cached null producer code differs; verify compatibility before reuse')
    layout = json.loads((null_dir / 'layout.json').read_text())
    cnvs = layout['cnvs']
    genes = np.asarray(layout['gsz_entrez'], dtype=np.int64)
    if (cnvs != sorted(ident['seeds']) or len(set(cnvs)) != len(cnvs) or
            len(np.unique(genes)) != len(genes) or genes.tolist() != ident['gsz'] or
            layout['reps'] != ident['reps']):
        raise ValueError('Cached CNV/gene/draw layout does not match its manifest')
    validate_config(cfg, len(cnvs))
    graph = ident['propagation']['graph']
    pair_identities = {}
    for key, directory in parent['pairwise_runs'].items():
        pair = verified_manifest(resolved(directory), parent['identity']['pairwise'][key])['identity']
        if (pair['graph'] != graph or pair['config']['propagation'] != ident['propagation']['config'] or
                pair['config']['thresholds']['individual'] != ident['threshold']):
            raise ValueError(f'Graph, propagation or threshold mismatch: {key}')
        pair_identities[key] = pair
    gsz = pd.read_csv(resolve_path(source / 'g_sz_genes.tsv'), sep='\t')
    if not np.array_equal(gsz.entrez, genes) or parent['g_sz_size'] != len(genes):
        raise ValueError('G_SZ ordering differs from cached masks')
    thresholds = pair_identities['schema_gwas']['config']['thresholds']
    if not ((gsz.schema_nps > thresholds['individual']) & (gsz.gwas_nps > thresholds['individual']) &
            (gsz.schema_nps * gsz.gwas_nps >= thresholds['product'])).all():
        raise ValueError('G_SZ genes fail the original SCHEMA/GWAS thresholds')
    candidate = pd.read_csv(resolve_path(resolved(parent['pairwise_runs']['schema_gwas']) / 'candidate_genes.tsv'), sep='\t')
    if set(candidate.entrez) != set(genes):
        raise ValueError('G_SZ differs from the original target-pair candidate set')
    packed = np.load(null_dir / 'null_reach.npy', mmap_mode='r', allow_pickle=False)
    expected_shape = (layout['reps'], len(cnvs), (len(genes) + 7) // 8)
    own_packed = np.load(null_dir / 'null_own_seed.npy', mmap_mode='r', allow_pickle=False)
    if packed.shape != expected_shape or packed.dtype != np.uint8 or own_packed.shape != expected_shape:
        raise ValueError('Packed mask shape/dtype mismatch')
    if cfg['calibration_draws'] + cfg['evaluation_draws'] > len(packed):
        raise ValueError('Requested split exceeds available cached draws')
    with np.load(resolve_path(null_dir / 'observed.npz'), allow_pickle=False) as obs:
        reach, own = obs['reach'].copy(), obs['own'].copy()
    if reach.shape != (len(cnvs), len(genes)) or own.shape != reach.shape:
        raise ValueError('Observed mask shape mismatch')
    expected_own = np.array([np.isin(genes, ident['seeds'][n]) for n in cnvs])
    if not np.array_equal(own, expected_own):
        raise ValueError('Observed own-seed masks differ from manifested seeds')
    previous_reach = pd.read_csv(resolve_path(source / 'degree/cnv_reach_matrix.tsv'), sep='\t').set_index('entrez')
    if not np.array_equal(previous_reach.loc[genes, [f'reached_{n}' for n in cnvs]].to_numpy().T, reach):
        raise ValueError('Observed masks differ from completed aggregate analysis')
    pathway_rule = parent['identity']['config']['pathways']
    for family, filename in pathway_rule['families'].items():
        if file_hash(ROOT / 'input/annotations' / filename) != parent['identity']['annotations'][family]:
            raise ValueError(f'Annotation changed: {filename}')
    terms = pd.concat([pd.read_csv(resolve_path(source / f'degree/{f.lower()}_convergence.tsv'), sep='\t')
                       for f in sorted(pathway_rule['families'])], ignore_index=True)
    terms = terms.sort_values(['database', 'term_id', 'term_name']).reset_index(drop=True)
    terms['term_key'] = terms.database + '::' + terms.term_id + '::' + terms.term_name
    if terms.term_key.duplicated().any() or not len(terms):
        raise ValueError('Need unique, nonempty frozen terms')
    sizes = terms.n_genes_in_gsz.to_numpy()
    if not ((sizes >= pathway_rule['min_genes_in_gsz']) & (sizes <= pathway_rule['max_genes_in_gsz']) &
            (len(genes) - sizes >= pathway_rule['min_complement'])).all():
        raise ValueError('Frozen term filters do not match the source tables')
    if terms.groupby('database').size().to_dict() != parent['pathway_terms_tested']:
        raise ValueError('Term family sizes differ from the source manifest')
    symbols = gsz.symbol.astype(str).to_numpy()
    positions = {s: list(np.flatnonzero(symbols == s)) for s in set(symbols)}
    membership = np.zeros((len(terms), len(genes)), dtype=bool)
    K = reach.sum(axis=0)
    for t, row in enumerate(terms.itertuples()):
        for entry in row.genes.split(','):
            match = re.fullmatch(r'(.+)\((\d+)\)', entry)
            if match is None or len(positions.get(match[1], [])) != 1:
                raise ValueError(f'Ambiguous frozen member in {row.term_key}: {entry}')
            pos = positions[match[1]][0]
            if int(match[2]) != K[pos] or membership[t, pos]:
                raise ValueError('Frozen membership recurrence or uniqueness mismatch')
            membership[t, pos] = True
    if not np.array_equal(membership.sum(axis=1), sizes):
        raise ValueError('Frozen membership sizes differ from annotated counts')
    return dict(source=source, parent=parent, null_dir=null_dir, null_identity=ident,
                pair_identities=pair_identities, cnvs=cnvs, genes=genes, symbols=symbols,
                gsz=gsz, packed=packed, reach=reach, own=own, terms=terms, membership=membership)


def unpack(packed, n_genes):
    return np.unpackbits(packed, axis=-1, count=n_genes).astype(bool)


def evaluate(packed, membership, cfg, out):
    """Two passes: freeze calibration, then stream held-out evaluation support."""
    C, B, chunk = cfg['calibration_draws'], cfg['evaluation_draws'], cfg['chunk_size']
    T, N = membership.shape
    J = packed.shape[1]
    matrix = sparse.csr_matrix(membership, dtype=np.int32)
    calibration = np.empty((C, J, T), dtype=np.int32)
    for start in range(0, C, chunk):
        stop = min(C, start + chunk)
        calibration[start:stop] = term_scores(unpack(packed[start:stop], N), matrix)[0]
        if start % (chunk * 10) == 0:
            log(f'Calibration scores: {stop}/{C}')
    ordered, cutoffs = calibrate(calibration, cfg['support_alpha'])
    del calibration
    np.savez_compressed(resolve_path(out / 'calibration_distributions.npz'), ordered_numerators=ordered,
                        support_cutoffs=cutoffs)
    dtype = np.uint8 if J <= 255 else np.uint16
    if J > 65535:
        raise ValueError('Too many CNVs for stored recurrence counts')
    counts = np.empty((B, T), dtype=dtype)
    support_totals = np.zeros((J, T), dtype=np.int64)
    for start in range(0, B, chunk):
        stop = min(B, start + chunk)
        scores, _ = term_scores(unpack(packed[C + start:C + stop], N), matrix)
        support = scores >= cutoffs
        counts[start:stop] = support.sum(axis=1)
        support_totals += support.sum(axis=0)
        if start % (chunk * 10) == 0:
            log(f'Evaluation support: {stop}/{B}')
    np.save(out / 'evaluation_recurrence_counts.npy', counts, allow_pickle=False)
    return ordered, cutoffs, counts, support_totals


def observed_tables(ctx, cfg, ordered, cutoffs, null_counts, support_totals, out):
    terms, reach, membership = ctx['terms'], ctx['reach'], ctx['membership']
    J, N = reach.shape
    B = len(null_counts)
    numerator, reached = term_scores(reach, membership)
    sizes = membership.sum(axis=1)
    denominator = sizes * (N - sizes)
    score = numerator / denominator
    pcal, cal_ge = calibration_p(ordered, numerator)
    support = numerator >= cutoffs
    if not np.array_equal(support, (pcal < cfg['support_alpha']) & (score > 0)):
        raise AssertionError('Integer support cutoffs differ from empirical support rule')
    M = support.sum(axis=0)
    p, ge = recurrence_p(M, count_histogram(null_counts, J))
    q, qall = adjusted_p(p, terms.database.to_numpy())
    ci_low, ci_high = binomial_interval(ge, B)
    cal_low, cal_high = binomial_interval(cal_ge, len(ordered))
    # Exact reconstruction independently checks the reused term membership and score formula.
    if not np.allclose(score.sum(axis=0), terms.observed_competitive_statistic, atol=1e-12):
        raise ValueError('Per-CNV scores do not reconstruct the existing aggregate statistic')
    rows = []
    for j, cnv in enumerate(ctx['cnvs']):
        for t, term in enumerate(terms.itertuples()):
            selected = membership[t] & reach[j]
            direct = selected & ctx['own'][j]
            rows.append(dict(cnv=cnv, database=term.database, term_id=term.term_id,
                term_name=term.term_name, term_key=term.term_key, n_genes_in_gsz=int(sizes[t]),
                total_cnv_reach=int(reach[j].sum()), competitive_score=score[j, t],
                competitive_numerator=int(numerator[j, t]), calibration_empirical_p=pcal[j, t],
                calibration_exceedances=int(cal_ge[j, t]), calibration_draws=len(ordered),
                calibration_tail_ci_low=cal_low[j, t], calibration_tail_ci_high=cal_high[j, t],
                support_cutoff_numerator=int(cutoffs[j, t]), supports_term=bool(support[j, t]),
                distance_from_support_p_threshold=cfg['support_alpha'] - pcal[j, t],
                borderline_support=bool(support[j, t] and abs(pcal[j, t] - cfg['support_alpha']) <= cfg['borderline_margin']),
                calibration_ci_crosses_alpha=bool(cal_low[j, t] <= cfg['support_alpha'] <= cal_high[j, t]),
                evaluation_support_frequency=support_totals[j, t] / B,
                n_reached_genes=int(reached[j, t]), reached_fraction=reached[j, t] / sizes[t],
                reached_genes=','.join(ctx['symbols'][selected]),
                reached_entrez=','.join(map(str, ctx['genes'][selected])),
                direct_seed_genes=','.join(ctx['symbols'][direct]),
                direct_seed_entrez=','.join(map(str, ctx['genes'][direct]))))
    cells = pd.DataFrame(rows)
    for target in ('schema', 'gwas'):
        pairs = pd.read_csv(resolve_path(ctx['source'] / f'cnv_{target}_summary.tsv'), sep='\t')
        cells = cells.merge(pairs[['a', 'p', 'q']].rename(columns={'a': 'cnv', 'p': f'{target}_pairwise_p',
                           'q': f'{target}_pairwise_q'}), on='cnv', how='left', validate='many_to_one')
    pairs = pd.read_csv(resolve_path(ctx['source'] / 'cnv_cnv_summary.tsv'), sep='\t')
    neighbors = {cnv: ','.join(sorted(set(pairs.loc[(pairs.a == cnv) & (pairs.q < cfg['fdr']), 'b']) |
                                    set(pairs.loc[(pairs.b == cnv) & (pairs.q < cfg['fdr']), 'a'])))
                 for cnv in ctx['cnvs']}
    cells['significant_cnv_pairwise_neighbors'] = cells.cnv.map(neighbors)
    table(cells, resolve_path(out / 'cnv_term_calibrated_support.tsv'))
    result = terms[['term_key', 'database', 'term_id', 'term_name', 'ontology', 'n_genes_in_gsz']].copy()
    result['observed_support_count'] = M
    result['supporting_cnvs'] = [','.join(np.asarray(ctx['cnvs'])[support[:, t]]) for t in range(len(terms))]
    result['n_positive_cnvs'] = (score > 0).sum(axis=0)
    result['null_mean_support_count'] = null_counts.mean(axis=0)
    result['evaluation_exceedances'], result['evaluation_draws'] = ge, B
    result['heterogeneous_empirical_p'], result['heterogeneous_bh_q'] = p, q
    result['heterogeneous_all_terms_bh_q'] = qall
    result['tail_probability_ci_low'], result['tail_probability_ci_high'] = ci_low, ci_high
    result['recurrently_supported'] = M >= 2
    result['family_fdr_recurrent'] = (M >= 2) & (q < cfg['fdr'])
    result['all_terms_fdr_recurrent'] = (M >= 2) & (qall < cfg['fdr'])
    result['aggregate_competitive_score'] = score.sum(axis=0)
    for old, new in [('competitive_empirical_p', 'existing_aggregate_p'), ('competitive_bh_q', 'existing_aggregate_q'),
                     ('observed_independent_cnv_count', 'u_independent_cnv_count'),
                     ('independent_cnv_empirical_p', 'u_p'), ('independent_cnv_bh_q', 'u_q')]:
        result[new] = terms[old]
    table(result, resolve_path(out / 'heterogeneous_term_recurrence.tsv'))
    diagnostics = result[['term_key', 'database', 'term_id', 'term_name']].copy()
    diagnostics['support_count_full'] = M
    diagnostics['max_single_cnv_support_strength'] = (-np.log10(pcal)).max(axis=0)
    diagnostics['median_support_strength'] = [float(np.median(-np.log10(pcal[support[:, t], t])))
                                             if support[:, t].any() else np.nan for t in range(len(terms))]
    diagnostics['n_borderline_support_calls'] = (support & (abs(pcal - cfg['support_alpha']) <= cfg['borderline_margin'])).sum(axis=0)
    for j, cnv in enumerate(ctx['cnvs']):
        diagnostics[f'support_count_without_{cnv}'] = M - support[j]
        diagnostics[f'aggregate_score_without_{cnv}'] = score.sum(axis=0) - score[j]
    table(diagnostics, resolve_path(out / 'heterogeneous_term_diagnostics.tsv'))
    for name, values in [('support', support.astype(int)), ('competitive_scores', score), ('calibration_p', pcal)]:
        matrix = pd.DataFrame(values, columns=terms.term_key)
        matrix.insert(0, 'cnv', ctx['cnvs'])
        table(matrix, resolve_path(out / f'cnv_term_{name}_matrix.tsv'))
    gene_matrix = pd.DataFrame(reach.astype(int), columns=ctx['genes'].astype(str))
    gene_matrix.insert(0, 'cnv', ctx['cnvs'])
    table(gene_matrix, resolve_path(out / 'cnv_gsz_reach_matrix.tsv'))
    table(ctx['gsz'], resolve_path(out / 'g_sz_genes.tsv'))
    return result, cells, score, pcal, support


def run_diagnostics(ctx, cfg, cutoffs, null_counts, out):
    """Conditional null calibration and reach-level power; no observed-data tuning.

The last L evaluation draws are pseudo-observations, excluded together from the
diagnostic reference distribution. The primary test still uses all B evaluation
draws. Power alternatives resample these held-out masks and preserve total reach.
"""
    d = cfg['diagnostics']
    C, B, L = cfg['calibration_draws'], cfg['evaluation_draws'], d['pseudo_datasets']
    terms, membership = ctx['terms'], ctx['membership']
    families = terms.database.to_numpy()
    J, N = ctx['reach'].shape
    hist = count_histogram(null_counts[:-L], J)
    pseudo_counts = null_counts[-L:]
    pseudo_p, _ = recurrence_p(pseudo_counts, hist)
    rows, term_hits = [], np.zeros(len(terms), dtype=int)
    masks = {f: families == f for f in np.unique(families)}
    for b, p in enumerate(pseudo_p):
        q, qall = adjusted_p(p, families)
        term_hits += p < cfg['fdr']
        for family, mask in {**masks, 'ALL_TERMS': np.ones(len(terms), dtype=bool)}.items():
            adjusted = qall[mask] if family == 'ALL_TERMS' else q[mask]
            rows.append(dict(pseudo_draw=C + B - L + b, database=family,
                minimum_p=float(p[mask].min()), minimum_q=float(adjusted.min()),
                n_p_below_alpha=int((p[mask] < cfg['fdr']).sum()),
                n_bh_discoveries=int((adjusted < cfg['fdr']).sum()),
                n_recurrent_bh_discoveries=int(((adjusted < cfg['fdr']) & (pseudo_counts[b, mask] >= 2)).sum()),
                diagnostic_reference_draws=B - L))
    pseudo = pd.DataFrame(rows)
    table(pseudo, resolve_path(out / 'null_pseudo_datasets.tsv'))
    summaries = []
    for family, frame in pseudo.groupby('database', sort=True):
        events = int((frame.n_bh_discoveries > 0).sum())
        low, high = binomial_interval(events, L)
        summaries.append(dict(database=family, pseudo_datasets=L, any_bh_discovery_events=events,
            complete_null_any_discovery_rate=events / L, rate_ci_low=float(low), rate_ci_high=float(high),
            mean_fraction_uncorrected_p_below_alpha=float(frame.n_p_below_alpha.mean() /
                (len(terms) if family == 'ALL_TERMS' else masks[family].sum())),
            inflation_flag=bool(low > cfg['fdr']),
            scope='conditional_on_fixed_calibration_and_diagnostic_reference'))
    summary = pd.DataFrame(summaries)
    table(summary, resolve_path(out / 'null_calibration_summary.tsv'))
    term_calibration = terms[['term_key', 'database', 'term_id', 'term_name']].copy()
    term_calibration['pseudo_datasets'] = L
    term_calibration['p_below_alpha_count'] = term_hits
    term_calibration['uncorrected_rejection_rate'] = term_hits / L
    term_calibration['rate_ci_low'], term_calibration['rate_ci_high'] = binomial_interval(term_hits, L)
    table(term_calibration, resolve_path(out / 'null_term_calibration.tsv'))
    log(f'Null pseudo-datasets: {L}; reference: {B - L}; calibration checks complete')

    # Select targets by size and stable identity only, before inspecting observed results.
    targets = []
    for family in sorted(masks):
        frame = terms.loc[masks[family]].sort_values(['n_genes_in_gsz', 'term_key'])
        targets.append(int(frame.index[(len(frame) - 1) // 2]))
    rng = np.random.default_rng(d['seed'])
    pseudo_masks = unpack(ctx['packed'][C + B - L:C + B], N)
    power_rows = []
    matrix = sparse.csr_matrix(membership, dtype=np.int32)
    for target in targets:
        term = terms.iloc[target]
        for affected_n in d['affected_cnvs']:
            for fraction in d['swap_fractions']:
                for rep in range(d['power_replicates']):
                    base_idx = int(rng.integers(L))
                    affected = rng.choice(J, size=affected_n, replace=False)
                    changed, swaps = inject_pathway_signal(pseudo_masks[base_idx], membership[target],
                                                          affected, fraction, rng)
                    scores, _ = term_scores(changed, matrix)
                    support = scores >= cutoffs
                    counts = support.sum(axis=0)
                    p, _ = recurrence_p(counts, hist)
                    q, qall = adjusted_p(p, families)
                    power_rows.append(dict(term_key=term.term_key, database=term.database,
                        term_id=term.term_id, term_name=term.term_name, n_genes_in_gsz=term.n_genes_in_gsz,
                        affected_cnvs=affected_n, swap_fraction=fraction, replicate=rep,
                        base_pseudo_draw=C + B - L + base_idx,
                        affected_cnv_names=','.join(np.asarray(ctx['cnvs'])[affected]), n_swapped_genes=swaps,
                        observed_support_count=int(counts[target]),
                        affected_support_count=int(support[affected, target].sum()),
                        empirical_p=float(p[target]), family_bh_q=float(q[target]),
                        all_terms_bh_q=float(qall[target]),
                        family_detection=bool(q[target] < cfg['fdr'] and counts[target] >= 2),
                        all_terms_detection=bool(qall[target] < cfg['fdr'] and counts[target] >= 2)))
                log(f'Power: {term.database} / {term.term_id}, {affected_n} CNVs, swap fraction {fraction}')
    power = pd.DataFrame(power_rows)
    table(power, resolve_path(out / 'power_simulation_replicates.tsv'))
    power_summary = []
    keys = ['term_key', 'database', 'term_id', 'term_name', 'n_genes_in_gsz', 'affected_cnvs', 'swap_fraction']
    for key, frame in power.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(replicates=len(frame), mean_swapped_genes=float(frame.n_swapped_genes.mean()),
                   mean_support_count=float(frame.observed_support_count.mean()),
                   mean_fraction_affected_supported=float((frame.affected_support_count / frame.affected_cnvs).mean()))
        for scope in ('family', 'all_terms'):
            successes = int(frame[f'{scope}_detection'].sum())
            low, high = binomial_interval(successes, len(frame))
            row.update({f'{scope}_detections': successes, f'{scope}_power': successes / len(frame),
                        f'{scope}_power_ci_low': float(low), f'{scope}_power_ci_high': float(high)})
        power_summary.append(row)
    table(pd.DataFrame(power_summary), resolve_path(out / 'power_summary.tsv'))
    write_json(out / 'diagnostic_design.json', {
        'calibration_draw_range_half_open': [0, C],
        'diagnostic_reference_draw_range_half_open': [C, C + B - L],
        'pseudo_draw_range_half_open': [C + B - L, C + B],
        'primary_evaluation_draw_range_half_open': [C, C + B],
        'power_target_keys': terms.loc[targets, 'term_key'].tolist(),
        'power_target_selection': 'lower median size per family, ties ordered by term_key; no observed statistics',
        'power_model': 'reach-level swaps preserving total reach; resampled null pseudo-datasets; not genomic CNVs',
        'interpretation': 'Conditional diagnostic only; does not establish FDR under arbitrary mixtures of effects.',
        'seed': d['seed']})
    return summary


def mechanism_profiles(ctx, cfg, result, support, out):
    """Group tested pathways into descriptive CNV mechanism profiles."""
    leading = []
    for _, frame in result.groupby('database', sort=True):
        ranked = frame.sort_values(['heterogeneous_bh_q', 'heterogeneous_empirical_p',
                                    'observed_support_count', 'term_key'], ascending=[True, True, False, True])
        leading.extend(ranked.index[ranked.family_fdr_recurrent].tolist())
        leading.extend(ranked.index[:cfg['reporting']['leading_terms_per_family']].tolist())
    leading = list(dict.fromkeys(leading))
    genesets = {t: set(np.flatnonzero(ctx['membership'][t])) for t in leading}
    cnvsets = {t: set(np.flatnonzero(support[:, t])) for t in leading}
    reachedsets = {t: set(np.flatnonzero((ctx['reach'] & ctx['membership'][t] &
                                        support[:, t, None]).any(axis=0))) for t in leading}
    representatives, group_for, rows, gene_rows = [], {}, [], []
    for t in leading:
        term = result.loc[t]
        matches = [r for r in representatives if cnvsets[t] and cnvsets[t] == cnvsets[r] and
                   jaccard(genesets[t], genesets[r]) >= cfg['reporting']['redundancy_gene_jaccard']]
        if matches:
            representative = matches[0]
        else:
            representative = t
            representatives.append(t)
        group_for[t] = f'mechanism_display_{representatives.index(representative) + 1:03d}'
        shared = ctx['membership'][t] & (ctx['reach'][support[:, t]].sum(axis=0) >= 2)
        rows.append(dict(term_key=term.term_key, database=term.database, term_id=term.term_id,
            term_name=term.term_name, display_group=group_for[t],
            representative_term_key=result.loc[representative, 'term_key'],
            observed_support_count=int(term.observed_support_count), supporting_cnvs=term.supporting_cnvs,
            family_bh_q=term.heterogeneous_bh_q, all_terms_bh_q=term.heterogeneous_all_terms_bh_q,
            family_fdr_recurrent=bool(term.family_fdr_recurrent),
            n_genes_reached_by_at_least_two_supporters=int(shared.sum()),
            genes_reached_by_at_least_two_supporters=','.join(ctx['symbols'][shared]),
            n_union_genes_reached_by_supporters=len(reachedsets[t]),
            status='descriptive_redundancy_group_not_inferred_causal_mechanism'))
        for j, cnv in enumerate(ctx['cnvs']):
            for g in sorted(genesets[t]):
                gene_rows.append(dict(term_key=term.term_key, cnv=cnv, supports_term=bool(support[j, t]),
                    entrez=int(ctx['genes'][g]), symbol=ctx['symbols'][g],
                    reached=bool(ctx['reach'][j, g]), direct_seed=bool(ctx['own'][j, g])))
    profiles = pd.DataFrame(rows)
    table(profiles, resolve_path(out / 'mechanism_profiles.tsv'))
    table(pd.DataFrame(gene_rows), resolve_path(out / 'leading_term_gene_reach.tsv'))
    comparisons = []
    for i, a in enumerate(leading):
        for b in leading[i + 1:]:
            comparisons.append(dict(term_key_a=result.loc[a, 'term_key'], term_key_b=result.loc[b, 'term_key'],
                term_gene_jaccard=jaccard(genesets[a], genesets[b]),
                supporting_cnv_jaccard=jaccard(cnvsets[a], cnvsets[b]),
                supporter_reached_gene_jaccard=jaccard(reachedsets[a], reachedsets[b]),
                same_nonempty_supporter_set=bool(cnvsets[a] and cnvsets[a] == cnvsets[b]),
                both_have_at_least_two_supporters=bool(len(cnvsets[a]) >= 2 and len(cnvsets[b]) >= 2),
                both_family_fdr_recurrent=bool(result.loc[a, 'family_fdr_recurrent'] and result.loc[b, 'family_fdr_recurrent']),
                both_all_terms_fdr_recurrent=bool(result.loc[a, 'all_terms_fdr_recurrent'] and result.loc[b, 'all_terms_fdr_recurrent']),
                shared_supporters=','.join(ctx['cnvs'][j] for j in sorted(cnvsets[a] & cnvsets[b])),
                supporters_only_a=','.join(ctx['cnvs'][j] for j in sorted(cnvsets[a] - cnvsets[b])),
                supporters_only_b=','.join(ctx['cnvs'][j] for j in sorted(cnvsets[b] - cnvsets[a])),
                same_display_group=group_for[a] == group_for[b], inference='descriptive_only'))
    columns = ['term_key_a', 'term_key_b', 'term_gene_jaccard', 'supporting_cnv_jaccard',
               'supporter_reached_gene_jaccard', 'same_nonempty_supporter_set', 'both_have_at_least_two_supporters',
               'both_family_fdr_recurrent', 'both_all_terms_fdr_recurrent', 'shared_supporters',
               'supporters_only_a', 'supporters_only_b', 'same_display_group', 'inference']
    table(pd.DataFrame(comparisons, columns=columns), resolve_path(out / 'mechanism_pair_comparisons.tsv'))
    return leading, profiles


def figures(ctx, cfg, result, score, pcal, support, null_counts, leading, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # All significant terms remain in the tables. Cap figures explicitly for readability.
    shown = leading[:30]
    labels = [f'{result.loc[t, "database"]}: {result.loc[t, "term_name"][:65]}' for t in shown]
    J = len(ctx['cnvs'])
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(shown) * .55), 11), constrained_layout=True)
    for ax, data, title, cmap in [(axes[0], score[:, shown], 'Competitive score', 'RdBu_r'),
                                  (axes[1], -np.log10(pcal[:, shown]), 'Calibration support strength (-log10 P)', 'viridis')]:
        kwargs = dict(vmin=-max(.01, float(abs(data).max())), vmax=max(.01, float(abs(data).max()))) if cmap == 'RdBu_r' else {}
        im = ax.imshow(data, aspect='auto', cmap=cmap, **kwargs)
        ys, xs = np.where(support[:, shown])
        ax.scatter(xs, ys, marker='o', facecolors='none', edgecolors='black', s=60, linewidths=1.3)
        ax.set_yticks(range(J), ctx['cnvs'])
        ax.set_xticks(range(len(shown)), labels, rotation=65, ha='right', fontsize=7)
        ax.set_title(title + '; circles = intermediate support calls')
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle('Exploratory CNV pathway profiles; displayed terms are ranked, not necessarily significant')
    fig.savefig(resolve_path(out / 'cnv_pathway_profiles.png'), dpi=160)
    plt.close(fig)

    shown_null = leading[:6]
    fig, axes = plt.subplots(len(shown_null), 1, figsize=(9, 2.6 * len(shown_null)), squeeze=False, constrained_layout=True)
    for ax, t in zip(axes[:, 0], shown_null):
        freq = np.bincount(null_counts[:, t].astype(int), minlength=J + 1) / len(null_counts)
        ax.bar(np.arange(J + 1), freq, color='#809ba7')
        ax.axvline(result.loc[t, 'observed_support_count'], color='#a33c37', linewidth=2, label='Observed')
        ax.set_xticks(range(J + 1))
        ax.set_ylabel('Null frequency')
        ax.set_title(f'{result.loc[t, "term_name"][:70]} | P={result.loc[t, "heterogeneous_empirical_p"]:.3g}, '
                     f'family q={result.loc[t, "heterogeneous_bh_q"]:.3g}', fontsize=9)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel('Number of CNV support calls')
    fig.savefig(resolve_path(out / 'pathway_recurrence_nulls.png'), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(4, len(shown) * .32)), constrained_layout=True)
    remaining = support[:, shown].sum(axis=0)[:, None] - support[:, shown].T
    im = ax.imshow(remaining, aspect='auto', vmin=0, vmax=J, cmap='Blues')
    ax.set_yticks(range(len(shown)), labels, fontsize=7)
    ax.set_xticks(range(J), ctx['cnvs'], rotation=45, ha='right')
    ax.set_title('Descriptive remaining support count after omitting each CNV')
    fig.colorbar(im, ax=ax, label='Remaining support count')
    fig.savefig(resolve_path(out / 'leave_one_cnv_out_recurrence.png'), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, kind in zip(axes, ('supporters', 'genes')):
        sets = [set(np.flatnonzero(support[:, t] if kind == 'supporters' else ctx['membership'][t])) for t in shown]
        sim = np.array([[jaccard(a, b) for b in sets] for a in sets])
        im = ax.imshow(sim, vmin=0, vmax=1, cmap='viridis')
        ax.set_title(f'Descriptive Jaccard similarity: {kind}')
        ax.set_xticks(range(len(shown)), np.arange(1, len(shown) + 1), fontsize=7)
        ax.set_yticks(range(len(shown)), np.arange(1, len(shown) + 1), fontsize=7)
        fig.colorbar(im, ax=ax, shrink=.6)
    fig.savefig(resolve_path(out / 'mechanism_overlap.png'), dpi=160)
    plt.close(fig)
    table(result.loc[shown, ['term_key', 'database', 'term_name']].assign(figure_index=np.arange(1, len(shown) + 1)),
          resolve_path(out / 'figure_term_key.tsv'))

    fig, axes = plt.subplots(len(shown_null), 1, figsize=(12, 2.8 * len(shown_null)), squeeze=False, constrained_layout=True)
    for ax, t in zip(axes[:, 0], shown_null):
        mask = ctx['membership'][t]
        im = ax.imshow(ctx['reach'][:, mask], aspect='auto', vmin=0, vmax=1, cmap='Blues')
        y, x = np.where(ctx['own'][:, mask])
        ax.scatter(x, y, marker='x', color='#b33b39', s=25)
        ax.set_yticks(range(J), [n + (' *' if support[j, t] else '') for j, n in enumerate(ctx['cnvs'])], fontsize=7)
        ax.set_xticks(range(mask.sum()), ctx['symbols'][mask], rotation=90, fontsize=6)
        ax.set_title(result.loc[t, 'term_name'][:100], fontsize=9)
    fig.suptitle('Reached pathway genes; red x = direct seed, * = intermediate CNV support')
    fig.savefig(resolve_path(out / 'supporting_genes.png'), dpi=160)
    plt.close(fig)


def report(ctx, cfg, result, calibration_summary, profiles, out):
    global_result = pd.read_csv(resolve_path(ctx['source'] / 'degree/global_gsz_convergence.tsv'), sep='\t').iloc[0]
    lines = ['# Exploratory heterogeneous CNV pathway convergence', '',
        'Question: which pathways receive recurrent CNV-specific network support, and do their '
        'descriptive profiles involve overlapping or distinct CNVs and reached genes?', '',
        f'- Network: {cfg["network"]}; fixed G_SZ: {len(ctx["genes"])} genes; CNVs: {len(ctx["cnvs"])}.',
        f'- Existing aggregate mean K: {global_result.observed:.3f}; null mean {global_result.null_mean:.3f}; '
        f'P = {global_result.empirical_p:.3g}.',
        f'- Cached degree-matched random-gene null: {cfg["calibration_draws"]:,} calibration draws, '
        f'{cfg["evaluation_draws"]:,} held-out evaluation draws; no propagation rerun.',
        f'- Support rule: positive competitive score and calibration empirical P < {cfg["support_alpha"]}.',
        '- Support cells and CNV subsets are descriptive; primary recurrence inference tests the complete joint null.', '',
        '| Annotation family | Terms | Recurrent, family q < FDR | Minimum family q |',
        '|---|---:|---:|---:|']
    for family, frame in result.groupby('database', sort=True):
        lines.append(f'| {family} | {len(frame)} | {int(frame.family_fdr_recurrent.sum())} | {frame.heterogeneous_bh_q.min():.4g} |')
    lines += ['', f'Across all {len(result)} tested terms, {int(result.all_terms_fdr_recurrent.sum())} '
              f'recurrent terms survive combined BH at {cfg["fdr"]}.', '']
    if result.family_fdr_recurrent.any():
        lines += ['The significant support counts are unusual under the complete degree-matched random-gene null. '
                  'They do not establish that every marked CNV has a true effect or that different causal routes exist.']
    else:
        lines += ['No pathway has FDR-significant recurrent support under this specification. '
                  'Displayed leading terms are exploratory candidates. A negative result must be interpreted '
                  'alongside the power simulations, especially for two- or three-CNV alternatives.']
    lines += ['', 'Mechanism comparisons are descriptive. Display groups combine terms only when their nonempty '
        'supporter sets are identical and their G_SZ gene Jaccard exceeds the configured threshold against a '
        'representative term. Grouping never changes inference. Related annotations are not independent replication.', '',
        'Diagnostics use a separate pseudo-observation/reference split within evaluation draws, with the same fixed '
        'calibration. They assess conditional complete-null behavior; they are not a proof of FDR control under '
        'arbitrary mixtures of effects. Power scenarios swap reached genes into size-selected pathways while '
        'preserving each CNV’s total reach. They are synthetic reach-level benchmarks, not simulated genomic CNVs.', '',
        '| Diagnostic family | Null pseudo-datasets with any BH discovery | 95% probability interval |',
        '|---|---:|---:|']
    for row in calibration_summary.itertuples():
        lines.append(f'| {row.database} | {row.any_bh_discovery_events}/{row.pseudo_datasets} | '
                     f'{row.rate_ci_low:.3f}–{row.rate_ci_high:.3f} |')
    if calibration_summary.inflation_flag.any():
        lines += ['', '**Diagnostic warning:** at least one complete-null discovery rate has its lower '
                  'confidence bound above the configured FDR. Investigate before interpreting discoveries.']
    lines += ['', 'The analysis is restricted to pathways represented in the fixed jointly SCHEMA–GWAS-supported '
        'network. It cannot establish schizophrenia specificity, activation/inhibition, causality, or penetrance. '
        'No partial-conjunction or formal between-subset test is performed.', '',
        'Key outputs:', '',
        '- [Recurrence results](heterogeneous_term_recurrence.tsv): family and all-terms q values, tail exceedances and intervals.',
        '- [CNV × term profiles](cnv_term_calibrated_support.tsv): scores, calibration P values, genes, and pairwise context.',
        '- [Mechanism profiles](mechanism_profiles.tsv) and [comparisons](mechanism_pair_comparisons.tsv): descriptive overlap.',
        '- [Profile heatmap](cnv_pathway_profiles.png), [supporting genes](supporting_genes.png), and [null distributions](pathway_recurrence_nulls.png).',
        '- [Power estimates](power_summary.tsv) and [null calibration](null_calibration_summary.tsv).',
        '- [Manifest](manifest.json): source hashes, exact split, configuration, and output hashes.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default=str(ROOT / 'configs/scz_heterogeneous_analysis.json'))
    parser.add_argument('--source-analysis', help='Completed core analysis directory; overrides the config')
    parser.add_argument('--output-root', help='Parent directory for content-addressed output')
    parser.add_argument('--network', help='Expected network identity; overrides the config')
    args = parser.parse_args(argv)
    cfg = json.loads(resolved(args.config).read_text())
    if args.source_analysis:
        cfg['source_analysis'] = args.source_analysis
    if args.network:
        cfg['network'] = args.network
    validate_config(cfg)
    from cnv_overlap.results import select_analysis, network_root
    source = resolved(cfg['source_analysis']) if cfg.get('source_analysis') else select_analysis(network_root(cfg['network'], ROOT/'results'), 'scz_core_gene_test')
    log('Verifying completed core analysis, cached masks, annotation hashes and graph identities')
    ctx = load_inputs(source, cfg)
    C, B = cfg['calibration_draws'], cfg['evaluation_draws']
    identity = dict(algorithm='scz_heterogeneous_v1', config=cfg,
        code={name: file_hash(ROOT / 'scripts' / name) for name in
              ('scz_heterogeneous_analysis.py', 'scz_heterogeneous_stats.py')},
        software=software(), source_manifest=file_hash(source / 'manifest.json'),
        null_manifest=file_hash(ctx['null_dir'] / 'manifest.json'),
        graph=ctx['null_identity']['propagation']['graph'],
        propagation=ctx['null_identity']['propagation'],
        cnv_seed_hashes={n: digest(s) for n, s in ctx['null_identity']['seeds'].items()},
        schizophrenia_seed_hashes={n: digest(s) for n, s in ctx['pair_identities']['schema_gwas']['seeds'].items()},
        gsz_thresholds=ctx['pair_identities']['schema_gwas']['config']['thresholds'],
        gsz_hash=digest(ctx['genes'].tolist()), annotation_hashes=ctx['parent']['identity']['annotations'],
        pathway_filters=ctx['parent']['identity']['config']['pathways'],
        term_membership_hash=digest(ctx['membership'].astype(int).tolist()),
        calibration_draw_range_half_open=[0, C], evaluation_draw_range_half_open=[C, C + B],
        statistic='integer D=N*S-n*r; C=D/[n*(N-n)]; support p_cal<alpha and D>0; recurrence=sum support',
        multiplicity='BH within each annotation family plus BH over all tested terms',
        exploratory=True)
    base = resolved(args.output_root) if args.output_root else analysis_from_source(source, 'scz_pathway_convergence')
    out = run_dir(base, identity, 'heterogeneous')
    if (out / 'manifest.json').exists():
        previous = json.loads((out / 'manifest.json').read_text())
        if previous.get('status') == 'complete':
            verified_manifest(out)
            log(f'Completed analysis reused: {out}')
            return out
    out.mkdir(parents=True, exist_ok=True)
    finalize_manifest(out / 'manifest.json', {'status': 'running', 'identity': identity})
    write_json(out / 'layout.json', {'cnvs': ctx['cnvs'], 'gsz_entrez': ctx['genes'].tolist(),
        'term_keys': ctx['terms'].term_key.tolist(),
        'calibration_distributions': 'ordered_numerators: calibration_draw x CNV x term (sorted along draw axis)',
        'support_cutoffs': 'CNV x term; D >= cutoff includes the positive-score condition',
        'evaluation_recurrence_counts': 'evaluation_draw x term; draw 0 is source draw calibration_draws'})
    ordered, cutoffs, null_counts, totals = evaluate(ctx['packed'], ctx['membership'], cfg, out)
    calibration_summary = run_diagnostics(ctx, cfg, cutoffs, null_counts, out)
    result, cells, score, pcal, support = observed_tables(ctx, cfg, ordered, cutoffs, null_counts, totals, out)
    leading, profiles = mechanism_profiles(ctx, cfg, result, support, out)
    figures(ctx, cfg, result, score, pcal, support, null_counts, leading, out)
    report(ctx, cfg, result, calibration_summary, profiles, out)
    artifacts = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    finalize_manifest(out / 'manifest.json', {'status': 'complete', 'identity': identity,
        'diagnostic_inflation_flag': bool(calibration_summary.inflation_flag.any()),
        'family_fdr_recurrent_terms': int(result.family_fdr_recurrent.sum()),
        'all_terms_fdr_recurrent_terms': int(result.all_terms_fdr_recurrent.sum()), 'artifacts': artifacts})
    log(f'Results: {out}')
    return out


if __name__ == '__main__':
    main()
