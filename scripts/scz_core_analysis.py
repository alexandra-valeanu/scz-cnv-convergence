"""Test CNV recurrence in the fixed SCHEMA–GWAS network and annotate pathways."""
from cnv_overlap.results import finalize_manifest, resolve_path
import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT/'scripts'))

from results_layout import results_dir
from cnv_overlap.pipeline import AnalysisConfig, pair_scores, run
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import digest, file_hash, graph_hash, run_dir, software, write_json
from cnv_overlap.statistics import bh, hit_mask, label_null, netcoloc_tail
from cnv_overlap.enrichment import load_gmt
from network_inputs import load_network_inputs
from notebook_analysis import SCZ_TARGETS, consecutive_windows, scz_overlap_membership
from scz_core_null import core_null, load_null

SCHEMA, GWAS = 'schema', 'gwaspri'


def log(message):
    print(message, flush=True)


def write_table(frame, path):
    frame.to_csv(path, sep='\t', index=False)


def empirical_p(null, observed):
    """(1 + #{null >= observed}) / (B + 1) along the draw axis, tolerant of float ties."""
    null = np.asarray(null, dtype=float)
    observed = np.asarray(observed, dtype=float)
    tol = 1e-9*np.maximum(1, np.abs(observed))
    return (1 + (null >= observed - tol).sum(axis=0)) / (len(null) + 1)


# --------------------------------------------------------------------------- pairwise

def run_target_pair(prop, symbols, entities, config, *, output_root, provenance, fdr):
    """Run SCHEMA × GWAS with the configured NetColoc label null and overlap policy.

    The target pair has its own output directory and BH family."""
    a, b = sorted(SCZ_TARGETS)
    seeds = {n: sorted(set(map(int, entities[n]['seeds'])) & prop.pos.keys()) for n in (a, b)}
    identity = {'algorithm': 'netcoloc_target_pair_v1', 'code': digest(inspect.getsource(run_target_pair)),
                'graph': prop.graph_id, 'config': asdict(config), 'seeds': seeds,
                'inputs': provenance, 'software': software()}
    identity = json.loads(json.dumps(identity))
    out = run_dir(output_root, identity, 'target_pair', f'alpha{config.propagation.alpha}')
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('status') == 'complete':
        old = json.loads(manifest.read_text())
        if any(file_hash(out/f) != h for f, h in old['artifacts'].items()):
            raise ValueError(f'Target-pair artifacts changed in {out}')
        return out
    out.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {'status': 'running', 'identity': identity})
    nodes = prop.nodes
    zs = {n: np.asarray(prop.nps(seeds[n])).copy() for n in (a, b)}
    write_table(pd.DataFrame([{'entity': n, 'kind': 'target', 'raw_seeds': len(set(entities[n]['seeds'])),
                               'retained_seeds': len(seeds[n])} for n in (a, b)]), resolve_path(out/'seed_qc.tsv'))
    write_table(pd.DataFrame([{'a': a, 'b': b, 'family': 'target_pair'}]), resolve_path(out/'hypotheses.tsv'))
    values, keep = pair_scores(zs[a], zs[b], seeds[a], seeds[b], nodes, config)
    observed, null = label_null(zs[a], zs[b], nodes, seeds[a], seeds[b], reps=config.permutation_reps,
                                seed=config.propagation.seed, overlap=config.overlap, thresholds=config.thresholds)
    assert all(np.isclose(observed[s], values[s]) for s in values)
    tests = pd.DataFrame([{'a': a, 'b': b, 'family': 'target_pair', 'statistic': s, 'null': 'netcoloc',
                           'primary': s == 'size', **netcoloc_tail(v, null[s])} for s, v in values.items()])
    tests['q'] = tests.p  # a single planned test per statistic
    write_table(tests, resolve_path(out/'pair_tests.tsv'))
    hit = np.zeros(len(nodes), dtype=bool)
    hit[keep] = hit_mask(zs[a][keep], zs[b][keep], config.thresholds)
    genes = pd.DataFrame([{'a': a, 'b': b, 'entrez': int(nodes[i]), 'symbol': symbols.get(int(nodes[i]), str(nodes[i])),
                           'nps_a': zs[a][i], 'nps_b': zs[b][i], 'seed_a': int(nodes[i]) in seeds[a],
                           'seed_b': int(nodes[i]) in seeds[b], 'status': 'thresholded_candidate_not_gene_FDR'}
                          for i in np.flatnonzero(hit)])
    write_table(genes, resolve_path(out/'candidate_genes.tsv'))
    size = tests[tests.statistic == 'size'].iloc[0]
    (out/'summary.md').write_text(
        f'# SCHEMA x GWAS (rare x common) NetColoc\n\nCandidate set size {int(size.observed)} '
        f'(null mean {size.null_mean:.1f}); NetColoc p = {size.p:.3g}, empirical p = {size.p_empirical:.3g}. '
        f'Gene-label permutations: {config.permutation_reps}; overlap policy: {config.overlap}. '
        f'Significant at q < {fdr}: {bool(size.q < fdr)}.\n')
    finalize_manifest(manifest, {'status': 'complete', 'identity': identity, 'primary_null': 'netcoloc',
        'artifacts': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p != manifest}})
    return out


def pairwise_runs(network, cfg, progress):
    """CNV x CNV and CNV x SCZ with exactly the notebooks' arguments, plus SCHEMA x GWAS."""
    if not network.endswith('_brain'):
        raise ValueError('The SCZ core analysis is defined on brain-filtered networks')
    min_seeds = cfg['min_seeds']
    spec_c, root_c, inputs_c, prov_c = load_network_inputs(ROOT/cfg['convergence_config'], network, min_seeds=min_seeds)
    spec_d, root_d, inputs_d, prov_d = load_network_inputs(ROOT/cfg['disorder_config'], network, min_seeds=min_seeds)
    config_c = AnalysisConfig.from_dict(spec_c['analysis'])
    config_d = AnalysisConfig.from_dict(spec_d['analysis'])
    if graph_hash(inputs_c[0], inputs_c[1]) != graph_hash(inputs_d[0], inputs_d[1]):
        raise ValueError('CNV x CNV and CNV x SCZ inputs use different networks')
    if config_c.propagation != config_d.propagation or config_c.thresholds != config_d.thresholds:
        raise ValueError('Pairwise configs disagree on propagation or thresholds')
    cnvs_c = {n: sorted(e['seeds']) for n, e in inputs_c[3].items() if e['kind'] == 'cnv'}
    cnvs_d = {n: sorted(e['seeds']) for n, e in inputs_d[3].items() if e['kind'] == 'cnv'}
    if cnvs_c != cnvs_d:
        raise ValueError('CNV seed sets differ between the convergence and disorder inputs')
    entities_d = {n: e for n, e in inputs_d[3].items()
                  if e['kind'] == 'cnv' or (e['kind'] == 'target' and n in SCZ_TARGETS)}
    if not all(n in entities_d for n in SCZ_TARGETS):
        raise ValueError('Both SCHEMA and GWAS-prioritized targets are required')
    progress('Pairwise NetColoc: CNV x CNV')
    cnv_cnv = run(*inputs_c, config=config_c, output_root=root_c/spec_c['output_root'],
                  cache_dir=root_c/spec_c['cache_dir'], provenance=prov_c, progress=progress)
    progress('Pairwise NetColoc: CNV x SCHEMA/GWAS')
    A, nodes, symbols, _, coords, regions = inputs_d
    cnv_scz = run(A, nodes, symbols, entities_d, coords, regions, config=config_d,
                  output_root=root_d/spec_d['output_root'], cache_dir=root_d/spec_d['cache_dir'],
                  provenance={**prov_d, 'question_family': 'combined', 'included_targets': list(SCZ_TARGETS),
                              'analysis_status': 'brain-expressed SCHEMA/GWAS comparisons'}, progress=progress)
    prop = Propagator(A, nodes, root_d/spec_d['cache_dir'], config_d.propagation)
    progress('Pairwise NetColoc: SCHEMA x GWAS')
    results_root = root_d/spec_d['results_root']
    target_pair = run_target_pair(prop, symbols, {n: entities_d[n] for n in SCZ_TARGETS}, config_d,
                                  output_root=results_dir(results_root, 'scz_target_pair'), fdr=config_d.fdr,
                                  provenance={**prov_d, 'question_family': 'schema_gwas'})
    for path in (cnv_cnv, cnv_scz):
        if json.loads((path/'manifest.json').read_text())['identity']['graph'] != prop.graph_id:
            raise ValueError(f'{path} was not computed on the current network')
    return {'prop': prop, 'config': config_d, 'inputs': inputs_d, 'entities': entities_d,
            'results_root': results_root, 'runs': {'cnv_cnv': cnv_cnv, 'cnv_scz': cnv_scz, 'schema_gwas': target_pair}}


def pairwise_summaries(runs, out):
    primary = []
    for key, path in runs.items():
        tests = pd.read_csv(resolve_path(path/'pair_tests.tsv'), sep='\t')
        tests = tests[(tests['null'] == 'netcoloc') & (tests.statistic == 'size')].copy()
        tests['size_ratio'] = tests.observed / tests.null_mean
        primary.append(tests.assign(run=str(path)))
    primary = pd.concat(primary, ignore_index=True)
    cols = ['a', 'b', 'family', 'observed', 'null_mean', 'size_ratio', 'excess', 'p', 'q', 'p_empirical', 'status', 'run']
    views = {'cnv_cnv_summary.tsv': primary.family == 'region_pair',
             'cnv_schema_summary.tsv': (primary.family == 'locus_target') & (primary.b == SCHEMA),
             'cnv_gwas_summary.tsv': (primary.family == 'locus_target') & (primary.b == GWAS),
             'schema_gwas_summary.tsv': primary.family == 'target_pair'}
    for name, mask in views.items():
        write_table(primary.loc[mask, cols].sort_values(['q', 'a', 'b']), out/name)
    return primary


def pairwise_annotations(runs, primary, gsz, symbols, fdr):
    """Which pairwise candidate sets contain each G_SZ gene; pairwise evidence is not required."""
    significant = set(zip(primary.a[primary.q < fdr], primary.b[primary.q < fdr]))
    candidates = pd.concat([pd.read_csv(resolve_path(runs[k]/'candidate_genes.tsv'), sep='\t') for k in ('cnv_cnv', 'cnv_scz')])
    candidates = candidates[candidates.entrez.isin(gsz)]
    rows = []
    for g in gsz:
        frame = candidates[candidates.entrez == g]
        row = {'entrez': int(g), 'symbol': symbols.get(int(g), str(g))}
        for label, mask in [('cnv_schema', frame.b == SCHEMA), ('cnv_gwas', frame.b == GWAS),
                            ('cnv_cnv', ~frame.b.isin(SCZ_TARGETS))]:
            pairs = list(zip(frame.a[mask], frame.b[mask]))
            names = [a if label != 'cnv_cnv' else f'{a}|{b}' for a, b in pairs]
            sig = [n for n, p in zip(names, pairs) if p in significant]
            row[f'{label}_candidate_in'] = ','.join(names)
            row[f'{label}_candidate_in_q_lt_{fdr}'] = ','.join(sig)
            row[f'n_{label}_candidate'] = len(names)
            row[f'n_{label}_candidate_q_lt_{fdr}'] = len(sig)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- G_SZ and pathways

def build_gsz(prop, entities, config, target_pair_dir):
    """G_SZ with the pairwise mask helper and overlap policy; must equal the pair's candidates."""
    seeds = {n: sorted(set(map(int, entities[n]['seeds'])) & prop.pos.keys()) for n in SCZ_TARGETS}
    zs = {n: np.asarray(prop.nps(seeds[n])) for n in SCZ_TARGETS}
    keep = np.ones(len(prop.nodes), dtype=bool)
    if config.overlap == 'remove':
        keep &= ~np.isin(prop.nodes, list(set(seeds[SCHEMA]) & set(seeds[GWAS])))
    mask = np.zeros(len(prop.nodes), dtype=bool)
    mask[keep] = hit_mask(zs[SCHEMA][keep], zs[GWAS][keep], config.thresholds)
    pair_genes = set(pd.read_csv(resolve_path(target_pair_dir/'candidate_genes.tsv'), sep='\t').entrez)
    if set(prop.nodes[mask].tolist()) != pair_genes:
        raise ValueError('G_SZ differs from the SCHEMA x GWAS NetColoc candidate set')
    return np.flatnonzero(mask), zs, seeds


def term_overlaps(families, annotation_dir, symbols, nodes, gsz):
    """Intersect every term with G_SZ via unique symbol -> Entrez mapping in the network."""
    table = pd.Series({int(g): symbols.get(int(g)) for g in nodes}).dropna()
    table = table[~table.duplicated(keep=False)]
    to_entrez = {s: g for g, s in table.items()}
    gsz_set = set(map(int, gsz))
    network = set(map(int, nodes))
    rows, members = [], {}
    for family, filename in families.items():
        for key, (description, genes) in load_gmt(Path(annotation_dir)/filename).items():
            entrez = {to_entrez[s] for s in genes if s in to_entrez}
            inside = sorted(entrez & gsz_set)
            match = re.search(r'\((GO:\d+)\)', key)
            term_id = description if family == 'Reactome' else (match.group(1) if match else key)
            ontology = key.rsplit(' ', 1)[-1] if family == 'SynGO' and key.endswith((' BP', ' CC')) else ''
            rows.append({'database': family, 'term_id': term_id, 'term_name': key, 'ontology': ontology,
                         'n_genes_in_network': len(entrez & network), 'n_genes_in_gsz': len(inside)})
            members[(family, key)] = inside
    return pd.DataFrame(rows), members


def size_distribution(terms, n_gsz):
    rows = []
    for family, frame in terms.groupby('database'):
        sizes = frame.n_genes_in_gsz[frame.n_genes_in_gsz > 0]
        row = {'database': family, 'terms_total': len(frame), 'terms_touching_gsz': len(sizes), 'gsz_size': n_gsz}
        row.update({f'q{int(q*100)}': float(sizes.quantile(q)) for q in (.1, .25, .5, .75, .9, .99)} if len(sizes) else {})
        row['max'] = int(sizes.max()) if len(sizes) else 0
        for lo in (3, 5, 10, 15, 20):
            row[f'n_terms_ge_{lo}'] = int((sizes >= lo).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def select_terms(terms, rule, n_gsz):
    lo, hi, comp = rule['min_genes_in_gsz'], rule['max_genes_in_gsz'], rule['min_complement']
    if lo is None or hi is None or comp is None:
        return None
    keep = (terms.n_genes_in_gsz >= lo) & (terms.n_genes_in_gsz <= hi) & (n_gsz - terms.n_genes_in_gsz >= comp)
    return terms[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- statistics

def gene_statistics(K_obs, K_null, prefix=''):
    null_max = K_null.max(axis=1)
    p = empirical_p(K_null, K_obs)
    return {f'{prefix}null_mean_k': K_null.mean(axis=0), f'{prefix}null_sd_k': K_null.std(axis=0),
            f'{prefix}empirical_p': p, f'{prefix}bh_q': bh(p),
            f'{prefix}p_maxT_gsz': (1 + (null_max[:, None] >= K_obs[None, :]).sum(axis=0)) / (len(K_null) + 1)}


def global_row(name, observed, null):
    return {'statistic': name, 'observed': float(observed), 'null_mean': float(null.mean()),
            'null_sd': float(null.std()), 'excess': float(observed - null.mean()),
            'null_low': float(np.quantile(null, .025)), 'null_high': float(np.quantile(null, .975)),
            'empirical_p': float(empirical_p(null, observed)), 'reps': len(null), 'p_resolution': 1/(len(null)+1)}


def pathway_tests(selected, members, gsz, symbols, K_obs, K_null, reach_obs, reach_null, fdr, chunk=1000):
    """Self-contained raw-sum mass, competitive mean difference, and independent-CNV count."""
    position = {int(g): i for i, g in enumerate(gsz)}
    M = np.zeros((len(selected), len(gsz)), dtype=np.float64)
    for t, (family, name) in enumerate(zip(selected.database, selected.term_name)):
        M[t, [position[g] for g in members[(family, name)]]] = 1
    n_in = M.sum(axis=1)
    n_out = len(gsz) - n_in
    T_obs = M @ K_obs
    C_obs = T_obs/n_in - (K_obs.sum() - T_obs)/n_out
    U_obs = ((reach_obs.astype(float) @ M.T) > 0).sum(axis=0)
    ge = {k: np.zeros(len(selected)) for k in 'TCU'}
    total = {k: np.zeros(len(selected)) for k in 'TCU'}
    B = len(K_null)
    for start in range(0, B, chunk):
        Kc = K_null[start:start+chunk].astype(np.float64)
        T = Kc @ M.T
        C = T/n_in - (Kc.sum(axis=1, keepdims=True) - T)/n_out
        U = ((reach_null[start:start+chunk].astype(np.float64) @ M.T) > 0).sum(axis=1)
        for key, null, obs in [('T', T, T_obs), ('C', C, C_obs), ('U', U, U_obs)]:
            ge[key] += (null >= obs - 1e-9*np.maximum(1, np.abs(obs))).sum(axis=0)
            total[key] += null.sum(axis=0)
    out = selected[['database', 'term_id', 'term_name', 'ontology', 'n_genes_in_gsz']].copy()
    out['observed_convergence_mass'] = T_obs
    out['null_mean_convergence_mass'] = total['T']/B
    out['self_contained_empirical_p'] = (1 + ge['T'])/(B + 1)
    out['observed_competitive_statistic'] = C_obs
    out['null_mean_competitive_statistic'] = total['C']/B
    out['competitive_empirical_p'] = (1 + ge['C'])/(B + 1)
    out['observed_independent_cnv_count'] = U_obs
    out['null_mean_independent_cnv_count'] = total['U']/B
    out['independent_cnv_empirical_p'] = (1 + ge['U'])/(B + 1)
    for col in ('self_contained', 'competitive', 'independent_cnv'):
        out[f'{col}_bh_q'] = np.nan
        for _, idx in out.groupby('database').groups.items():
            out.loc[idx, f'{col}_bh_q'] = bh(out.loc[idx, f'{col}_empirical_p'].to_numpy())
    s, c = out.self_contained_bh_q < fdr, out.competitive_bh_q < fdr
    out['pattern'] = np.select([s & c, s & ~c, ~s & c], ['A_both', 'B_self_contained_only', 'C_relative_only'], 'D_neither')
    out['genes'] = [','.join(f'{symbols.get(g, g)}({int(K_obs[position[g]])})'
                             for g in sorted(members[(f, n)], key=lambda g: -K_obs[position[g]]))
                    for f, n in zip(out.database, out.term_name)]
    return out.sort_values(['database', 'self_contained_empirical_p', 'competitive_empirical_p'])


# --------------------------------------------------------------------------- analysis

def analyze_null(label, null, *, gsz, zs, scz_seeds, cnv_seeds, symbols, pathway_sel, members, fdr, out, both_pairwise):
    names, reach_obs, own_obs = null['cnvs'], null['reach_obs'], null['own_obs']
    reach, own = null['reach'], null['own']
    K_obs = reach_obs.sum(axis=0)
    K_null = reach.sum(axis=1).astype(np.int16)
    Kx_obs = (reach_obs & ~own_obs).sum(axis=0)
    Kx_null = (reach & ~own).sum(axis=1).astype(np.int16)
    idx = [int(np.searchsorted(zs['nodes'], g)) for g in gsz]
    s, w = zs[SCHEMA][idx], zs[GWAS][idx]
    genes = pd.DataFrame({'entrez': gsz, 'symbol': [symbols.get(int(g), str(g)) for g in gsz],
                          'schema_nps': s, 'gwas_nps': w, 'schema_gwas_product': s*w, 'in_g_sz': True,
                          'k_cnv_recurrence': K_obs})
    for j, n in enumerate(names):
        genes[f'reached_{n}'] = reach_obs[j].astype(int)
    for key, value in gene_statistics(K_obs, K_null).items():
        genes[key] = value
    genes['k_seed_excluded'] = Kx_obs
    for key, value in gene_statistics(Kx_obs, Kx_null, 'seed_excluded_').items():
        genes[key] = value
    genes['is_schema_seed'] = genes.entrez.isin(scz_seeds[SCHEMA])
    genes['is_gwas_seed'] = genes.entrez.isin(scz_seeds[GWAS])
    genes['cnv_seed_of'] = [','.join(n for n in names if int(g) in set(cnv_seeds[n])) for g in gsz]
    genes['is_any_cnv_seed'] = genes.cnv_seed_of != ''
    genes = genes.merge(both_pairwise, on='entrez', how='left')
    loo = pd.DataFrame({'entrez': gsz, 'symbol': genes.symbol, 'k_full': K_obs})
    for j, n in enumerate(names):
        loo[f'k_without_{n}'] = K_obs - reach_obs[j]
    loo['min_k_leave_one_out'] = loo[[f'k_without_{n}' for n in names]].min(axis=1)
    genes['min_k_leave_one_out'] = loo.min_k_leave_one_out.to_numpy()
    genes = genes.sort_values(['empirical_p', 'k_cnv_recurrence'], ascending=[True, False])
    write_table(genes, resolve_path(out/'gene_recurrence_ranking.tsv'))
    write_table(genes.sort_values(['seed_excluded_empirical_p', 'k_seed_excluded'], ascending=[True, False])
                [['entrez', 'symbol', 'k_cnv_recurrence', 'k_seed_excluded', 'cnv_seed_of'] +
                 [c for c in genes if c.startswith('seed_excluded_')]], resolve_path(out/'gene_recurrence_seed_excluded.tsv'))
    write_table(genes[['entrez', 'symbol'] + [f'reached_{n}' for n in names] +
                      ['k_cnv_recurrence', 'schema_nps', 'gwas_nps', 'empirical_p', 'bh_q', 'p_maxT_gsz']],
                resolve_path(out/'cnv_reach_matrix.tsv'))
    write_table(loo.sort_values('k_full', ascending=False), resolve_path(out/'leave_one_cnv_out.tsv'))
    glob = pd.DataFrame([global_row('mean_k_cnv_recurrence', K_obs.mean(), K_null.mean(axis=1)),
                         global_row('mean_k_seed_excluded', Kx_obs.mean(), Kx_null.mean(axis=1))])
    write_table(glob, resolve_path(out/'global_gsz_convergence.tsv'))
    rows = []
    for j, n in enumerate(names):
        keep = [i for i in range(len(names)) if i != j]
        rows.append({'omitted_cnv': n, **global_row('mean_k_cnv_recurrence', reach_obs[keep].sum(axis=0).mean(),
                                                   reach[:, keep].sum(axis=1).mean(axis=1))})
    write_table(pd.DataFrame(rows), resolve_path(out/'leave_one_cnv_out_global.tsv'))
    pathways = None
    if pathway_sel is not None:
        pathways = pathway_tests(pathway_sel, members, gsz, symbols, K_obs, K_null, reach_obs, reach, fdr)
        for family, frame in pathways.groupby('database'):
            write_table(frame, resolve_path(out/f'{family.lower()}_convergence.tsv'))
    figures(label, genes, names, glob, K_null, K_obs, pathways, fdr, out)
    return genes, glob, pathways


def figures(label, genes, names, glob, K_null, K_obs, pathways, fdr, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    null_mean = K_null.mean(axis=1)
    ax.hist(null_mean, bins=60, color='#9aa7b8')
    ax.axvline(K_obs.mean(), color='#c0392b', lw=2, label=f'observed {K_obs.mean():.2f}')
    ax.set(xlabel='Mean CNV recurrence over G_SZ', ylabel='Null draws',
           title=f'Global G_SZ convergence ({label}); p = {glob.empirical_p.iloc[0]:.2g}')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(resolve_path(out/'global_gsz_convergence.png'), dpi=200)
    plt.close(fig)
    top = genes.head(40)
    fig, ax = plt.subplots(figsize=(0.45*len(names) + 3, 0.22*len(top) + 1.5))
    ax.imshow(top[[f'reached_{n}' for n in names]].to_numpy(), aspect='auto', cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha='right')
    ax.set_yticks(range(len(top)), [f'{s} (K={k}, p={p:.1g})' for s, k, p in zip(top.symbol, top.k_cnv_recurrence, top.empirical_p)],
                  fontsize=7)
    ax.set_title('CNV reach (NPS > 1): top 40 G_SZ genes by empirical p', fontsize=9)
    fig.tight_layout()
    fig.savefig(resolve_path(out/'cnv_reach_heatmap.png'), dpi=200)
    plt.close(fig)
    if pathways is None:
        return
    for family, frame in pathways.groupby('database'):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        colours = frame.pattern.map({'A_both': '#c0392b', 'B_self_contained_only': '#e39c37',
                                     'C_relative_only': '#2e86c1', 'D_neither': '#b8bec6'})
        ax.scatter(-np.log10(frame.self_contained_empirical_p), -np.log10(frame.competitive_empirical_p),
                   s=8 + 1.5*frame.n_genes_in_gsz, c=colours, alpha=.75, lw=0)
        for _, r in frame[frame.pattern != 'D_neither'].nsmallest(12, 'self_contained_empirical_p').iterrows():
            ax.annotate(r.term_name[:45], (-np.log10(r.self_contained_empirical_p), -np.log10(r.competitive_empirical_p)),
                        fontsize=5)
        ax.set(xlabel='-log10 p, self-contained mass', ylabel='-log10 p, competitive',
               title=f'{family}: {len(frame)} terms (colour = BH q < {fdr} pattern)')
        fig.tight_layout()
        fig.savefig(resolve_path(out/f'{family.lower()}_convergence.png'), dpi=200)
        plt.close(fig)


def pilot_reproduction(results_root, prop, cnv_seeds, gsz_idx, K_null):
    """Compare against the earlier run_convergence degree null, if it exists (same draws by design)."""
    for manifest in sorted((results_dir(results_root, 'multi_cnv_convergence')/'degree').glob('**/manifest.json')):
        m = json.loads(manifest.read_text())
        i = m.get('identity', {})
        if m.get('status') != 'complete' or i.get('null_model') != 'degree' or i.get('seeds') != cnv_seeds:
            continue
        pilot = np.load(manifest.parent/'null_counts.npy', mmap_mode='r')
        n = min(len(pilot), len(K_null))
        same = bool(np.array_equal(np.asarray(pilot[:n])[:, gsz_idx], K_null[:n]))
        return {'pilot': str(manifest.parent), 'rows_compared': n, 'identical_gsz_counts': same,
                'pilot_graph_matches': i['propagation']['graph'] == prop.graph_id,
                'pilot_source_matches': i['propagation']['software']['source'] == prop.metadata['software']['source']}
    return {'pilot': None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--network', default='PCNET2.2_brain')
    parser.add_argument('--config', default=str(ROOT/'configs/scz_core_analysis.json'))
    args = parser.parse_args(argv)
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    ctx = pairwise_runs(args.network, cfg, log)
    prop, config, runs = ctx['prop'], ctx['config'], ctx['runs']
    A, nodes, symbols, _, coords, regions = ctx['inputs']
    entities = ctx['entities']
    fdr = cfg.get('fdr', config.fdr)
    base = results_dir(ctx['results_root'], 'scz_core_gene_test')

    gsz_idx, zs, scz_seeds = build_gsz(prop, entities, config, runs['schema_gwas'])
    gsz = prop.nodes[gsz_idx]
    zs['nodes'] = prop.nodes
    log(f'G_SZ: {len(gsz)} genes (SCHEMA x GWAS NetColoc candidates, overlap policy {config.overlap})')
    cnv_seeds = {n: sorted(set(map(int, e['seeds'])) & prop.pos.keys())
                 for n, e in entities.items() if e['kind'] == 'cnv'}
    scz_union = set(scz_seeds[SCHEMA]) | set(scz_seeds[GWAS])

    null_dirs = {}
    for model, reps in cfg['nulls'].items():
        if not reps:
            continue
        seeds, windows, excluded = cnv_seeds, None, ()
        if model == 'degree_no_scz':
            seeds = {n: [g for g in s if g not in scz_union] for n, s in cnv_seeds.items()}
            excluded = tuple(sorted(scz_union))
        if model == 'region':
            r = regions.set_index('entity')
            windows = {n: consecutive_windows(coords, prop.nodes, r.loc[n, 'chrom'], len(seeds[n])) for n in seeds}
        log(f'Core null {model}: {reps} draws')
        null_dirs[model] = core_null(prop, seeds, gsz_idx, config=config, reps=reps, null_model=model,
                                     output_root=base/'nulls', windows=windows, excluded=excluded, progress=log)

    annotation_dir = ROOT/'input/annotations'
    families = cfg['pathways']['families']
    identity = {'network': args.network, 'algorithm': 'scz_core_analysis_v1', 'code': file_hash(__file__), 'config': cfg,
                'nulls': {m: file_hash(d/'manifest.json') for m, d in null_dirs.items()},
                'pairwise': {k: file_hash(d/'manifest.json') for k, d in runs.items()},
                'annotations': {f: file_hash(annotation_dir/n) for f, n in families.items()}}
    out = run_dir(base, identity, 'analysis')
    from cnv_overlap.results import reuse_completed
    if reuse_completed(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    finalize_manifest(out/'manifest.json', {'status': 'running', 'identity': identity})

    primary = pairwise_summaries(runs, out)
    write_table(pd.DataFrame({'entrez': gsz, 'symbol': [symbols.get(int(g), str(g)) for g in gsz],
                              'schema_nps': zs[SCHEMA][gsz_idx], 'gwas_nps': zs[GWAS][gsz_idx],
                              'is_schema_seed': np.isin(gsz, scz_seeds[SCHEMA]),
                              'is_gwas_seed': np.isin(gsz, scz_seeds[GWAS])}), resolve_path(out/'g_sz_genes.tsv'))
    write_table(pairwise_annotations(runs, primary, gsz, symbols, fdr), resolve_path(out/'recurrence_pairwise_annotations.tsv'))
    profiles = {n: np.asarray(prop.nps(s)) for n, s in cnv_seeds.items()} | {n: zs[n] for n in SCZ_TARGETS}
    ents = {n: {**e, 'seeds': cnv_seeds[n] if e['kind'] == 'cnv' else scz_seeds[n]} for n, e in entities.items()}
    table, _, _ = scz_overlap_membership(prop.nodes, ents, profiles, config)
    both_pairwise = table[['entrez', 'n_cnvs_both', 'cnvs_both']].rename(
        columns={'n_cnvs_both': 'n_cnvs_both_pairwise', 'cnvs_both': 'cnvs_both_pairwise'})

    terms, members = term_overlaps(families, annotation_dir, symbols, prop.nodes, gsz)
    write_table(terms, resolve_path(out/'term_sizes.tsv'))
    write_table(size_distribution(terms, len(gsz)), resolve_path(out/'term_size_distribution.tsv'))
    selected = select_terms(terms, cfg['pathways'], len(gsz))
    if selected is None:
        log('Pathway size cutoffs are not frozen in the config: wrote term sizes only, no pathway P values.')
    else:
        log(f'Pathway terms tested: {selected.groupby("database").size().to_dict()}')

    results = {}
    for model, d in null_dirs.items():
        sub = out/model
        sub.mkdir(exist_ok=True)
        log(f'Analysing {model} null')
        null = load_null(d)
        seeds = cnv_seeds if model != 'degree_no_scz' else {n: [g for g in s if g not in scz_union] for n, s in cnv_seeds.items()}
        results[model] = analyze_null(model, null, gsz=gsz, zs=zs, scz_seeds=scz_seeds, cnv_seeds=seeds,
                                      symbols=symbols, pathway_sel=selected, members=members, fdr=fdr, out=sub,
                                      both_pairwise=both_pairwise)
        if model == 'degree':
            results['pilot'] = pilot_reproduction(ctx['results_root'], prop, cnv_seeds, gsz_idx,
                                                  null['reach'].sum(axis=1).astype(np.int16))

    # Overlap between CNV seeds and SCHEMA/GWAS seeds: named genes and an observed-only check.
    overlap_rows = [{'entrez': g, 'symbol': symbols.get(g, str(g)), 'cnv': n, 'is_schema_seed': g in scz_seeds[SCHEMA],
                     'is_gwas_seed': g in scz_seeds[GWAS], 'in_g_sz': g in set(map(int, gsz))}
                    for n, s in cnv_seeds.items() for g in s if g in scz_union]
    overlap = pd.DataFrame(overlap_rows, columns=['entrez', 'symbol', 'cnv', 'is_schema_seed', 'is_gwas_seed', 'in_g_sz'])
    if 'degree' in results:
        g = results['degree'][0].set_index('entrez')
        overlap['k_cnv_recurrence'] = overlap.entrez.map(g.k_cnv_recurrence)
        overlap['k_seed_excluded'] = overlap.entrez.map(g.k_seed_excluded)
    write_table(overlap, resolve_path(out/'cnv_scz_seed_overlap.tsv'))
    threshold = config.thresholds.individual
    K_full = sum(np.asarray(prop.nps(s))[gsz_idx] > threshold for s in cnv_seeds.values())
    reduced = {n: [g for g in s if g not in scz_union] for n, s in cnv_seeds.items()}
    K_red = sum(np.asarray(prop.nps(s))[gsz_idx] > threshold for s in reduced.values())
    changed = np.flatnonzero(K_full != K_red)
    check = pd.DataFrame({'entrez': gsz[changed], 'symbol': [symbols.get(int(x), str(x)) for x in gsz[changed]],
                          'k_with_overlap_seeds': K_full[changed], 'k_without_overlap_seeds': K_red[changed]})
    write_table(check, resolve_path(out/'overlap_removed_check.tsv'))
    overlap_summary = {'affected_cnvs': sorted(n for n in cnv_seeds if len(reduced[n]) != len(cnv_seeds[n])),
                       'mean_k_with': float(K_full.mean()), 'mean_k_without': float(K_red.mean()),
                       'genes_changed': int(len(changed))}

    lines = [f'# SCZ multi-CNV convergence: {args.network}', '',
             f'Analysis directory: `{out}`', '',
             '## Pairwise NetColoc (gene-label null)', '']
    size = primary[primary.family == 'target_pair'].iloc[0]
    lines += [f'- SCHEMA x GWAS: candidate set {int(size.observed)} genes vs null mean {size.null_mean:.1f} '
              f'(ratio {size.size_ratio:.2f}); NetColoc p = {size.p:.3g}.',
              f'- CNV x SCHEMA pairs with q < {fdr}: {int(((primary.b == SCHEMA) & (primary.q < fdr)).sum())}; '
              f'CNV x GWAS: {int(((primary.b == GWAS) & (primary.q < fdr)).sum())}; '
              f'CNV x CNV: {int(((primary.family == "region_pair") & (primary.q < fdr)).sum())}.', '',
              f'## G_SZ: {len(gsz)} genes', '']
    for model in null_dirs:
        genes, glob, pathways = results[model]
        g0 = glob.iloc[0]
        lines += [f'## Null: {model} ({int(g0.reps)} draws)' + (' - primary' if model == cfg['primary_null'] else ' - sensitivity'), '',
                  f'- Global mean K over G_SZ: observed {g0.observed:.3f} vs null {g0.null_mean:.3f} '
                  f'(95% null {g0.null_low:.3f}-{g0.null_high:.3f}); empirical p = {g0.empirical_p:.3g}.',
                  f'- Seed-excluded global mean: observed {glob.iloc[1].observed:.3f} vs null {glob.iloc[1].null_mean:.3f}; '
                  f'p = {glob.iloc[1].empirical_p:.3g}.',
                  f'- Genes with BH q < {fdr}: {int((genes.bh_q < fdr).sum())} (smallest q {genes.bh_q.min():.3g}); '
                  f'smallest p_maxT_gsz {genes.p_maxT_gsz.min():.3g}. '
                  + ('No gene survives multiplicity correction: the gene list is a ranking, not a discovery set.'
                     if not (genes.bh_q < fdr).any() else ''),
                  '- Top genes: ' + ', '.join(f'{s} (K={k}, p={p:.2g})' for s, k, p in
                                               zip(genes.symbol[:10], genes.k_cnv_recurrence[:10], genes.empirical_p[:10]))]
        if pathways is not None:
            for family, frame in pathways.groupby('database'):
                counts = frame.pattern.value_counts().to_dict()
                lines.append(f'- {family}: {len(frame)} terms; pattern A (both) {counts.get("A_both", 0)}, '
                             f'B (self-contained only) {counts.get("B_self_contained_only", 0)}, '
                             f'C (relative only) {counts.get("C_relative_only", 0)}.')
        lines.append('')
    lines += ['## Overlap between CNV seeds and SCHEMA/GWAS seeds', '',
              f'- Overlapping genes: ' + (', '.join(f'{r.symbol} ({r.cnv}; in G_SZ: {r.in_g_sz})' for r in overlap.itertuples()) or 'none'),
              f'- Observed-only removal check: mean K {overlap_summary["mean_k_with"]:.3f} -> {overlap_summary["mean_k_without"]:.3f}; '
              f'{overlap_summary["genes_changed"]} G_SZ genes change K.', '']
    if 'pilot' in results and results['pilot'].get('pilot'):
        p = results['pilot']
        lines += [f'Pilot reproduction: first {p["rows_compared"]} degree draws identical to the 5,000-draw pilot over G_SZ: '
                  f'{p["identical_gsz_counts"]}.', '']
    lines += ['Status: exploratory analysis with a frozen specification; see the implementation plan, section 35.']
    (out/'summary.md').write_text('\n'.join(lines) + '\n')
    artifacts = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity, 'null_dirs': {m: str(d) for m, d in null_dirs.items()},
        'pairwise_runs': {k: str(d) for k, d in runs.items()}, 'g_sz_size': int(len(gsz)),
        'pathway_terms_tested': None if selected is None else selected.groupby('database').size().to_dict(),
        'overlap_removed_check': overlap_summary, 'pilot_reproduction': results.get('pilot'),
        'schema': 'k_cnv_recurrence replaces run_convergence n_cnvs; n_cnvs_both_pairwise is the pairwise-derived count',
        'artifacts': artifacts})
    log((out/'summary.md').read_text())
    log(f'Results: {out}')
    return out


if __name__ == '__main__':
    main()
