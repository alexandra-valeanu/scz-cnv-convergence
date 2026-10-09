"""SCHEMA–GWAS landscape, Leiden modules, CNV coverage and conditional null tests."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import networkx as nx
import numpy as np
import pandas as pd

from cnv_overlap.pipeline import AnalysisConfig
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import file_hash, run_dir, software, write_json
from cnv_overlap.reporting import write_graphml
from cnv_overlap.statistics import bh, hit_mask, rng_for, tail
from cnv_cosine_similarity import export_cosines
from cnv_pair_network import build_pair_network, significant_pairs
from notebook_analysis import scz_overlap_membership
from scz_core_analysis import run_target_pair
from scz_restricted_core_null import landscape_heat, sample_joint, sampling_plan
from scz_shared_network import load_inputs
from shared_network_cluster_context import (_context_paths, context_ora, context_summary,
                                            plot_context_heatmap)
from shared_network_leiden import leiden_membership
from publication_enrichment import analysis_tools, cluster_colors, module_names, summarize_enrichment

ROOT = Path(__file__).resolve().parents[1]


def save(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False)


def landscape_membership(nodes, symbols, entities, profiles, config, min_cnvs=3):
    """Three pair masks must all hold; two CNV-target masks do not imply the third."""
    table, pairs, both = scz_overlap_membership(nodes, entities, profiles, config, min_cnvs)
    landscape = hit_mask(profiles['schema'], profiles['gwaspri'], config.thresholds)
    if config.overlap == 'remove':
        landscape &= ~np.isin(nodes, list(set(entities['schema']['seeds']) & set(entities['gwaspri']['seeds'])))
    table.insert(1, 'symbol', [symbols.get(int(g), str(g)) for g in nodes])
    table['in_scz_landscape'] = landscape
    table = table.rename(columns={'shared_core': 'shared_core_unrestricted'})
    table['shared_core_in_landscape'] = table.shared_core_unrestricted & landscape
    for name in sorted(entities):
        table[f'nps_{name}'] = profiles[name]
        table[f'seed_{name}'] = np.isin(nodes, entities[name]['seeds'])
    table['schema_gwas_product'] = profiles['schema']*profiles['gwaspri']
    table['is_any_seed'] = table[[c for c in table if c.startswith('seed_')]].any(axis=1)
    return table, pairs, both


def conditional_null(prop, profiles, seeds, names, landscape, both, config, thresholds, reps, out,
                     symbols=None, progress=print):
    """Matched random seed sets on the fixed landscape: are the SAME genes picked by chance?

    Each joint draw replaces every CNV seed by a gene from the same degree bin.
    Random genes never include any CNV, SCHEMA or GWAS seed. One injective mapping
    of the CNV seed union keeps each CNV's seed count, degree profile and every
    between-CNV seed overlap. SCHEMA/GWAS profiles and the landscape stay fixed.
    No genomic-window null is used. Propagation paths and NPS references remain
    those of the full interactome; only output rows are restricted, for efficiency.

    Primary endpoints concern gene identity: per gene, how often random seed sets
    reach it with both targets in at least as many CNVs (K); per core, how many of
    the observed core genes a random core contains. Core size is a secondary test.
    """
    if config.overlap != 'bin':
        raise ValueError('The matched-seed landscape null requires overlap=bin')
    out.mkdir(parents=True, exist_ok=True)
    symbols = symbols or {}
    idx = np.flatnonzero(landscape)
    nodes = prop.nodes[idx]
    used = {n: seeds[n] for n in (*names, 'schema', 'gwaspri')}
    excluded = set().union(*map(set, used.values()))
    groups, balance = sampling_plan(prop, used, names)
    save(balance, out/'degree_pool_qc.tsv')
    save(pd.DataFrame([{'entrez': g, 'symbol': symbols.get(int(g), str(g)),
                        'seed_of': ','.join(n for n in used if g in used[n])} for g in sorted(excluded)]),
         out/'excluded_random_seeds.tsv')
    observed_by_cnv = np.array([both[n][idx].sum() for n in names])
    K_obs = np.stack([both[n][idx] for n in names]).sum(axis=0)
    cores_obs = np.array([K_obs >= k for k in thresholds])
    refs = {n: tuple(v[idx] for v in prop.reference(prop.seed_indices(seeds[n]))[:2]) for n in names}
    kernel = np.asarray(prop.H[idx])
    K_null = np.empty((reps, len(idx)), dtype=np.uint8)
    coverage = np.empty((reps, len(names)), dtype=int)
    for rep in range(reps):
        draws = sample_joint(prop, used, names, groups,
                             rng_for(config.propagation.seed, 'landscape_matched_seeds', rep))
        if excluded & {g for n in names for g in draws[n]}:
            raise AssertionError('A random seed set contains a CNV, SCHEMA or GWAS seed')
        recurrent = np.zeros(len(idx), dtype=np.uint8)
        for j, n in enumerate(names):
            mu, sd = refs[n]
            z = (np.log(landscape_heat(kernel, prop.seed_indices(draws[n]))) - mu)/sd
            reached = (hit_mask(z, profiles['schema'][idx], config.thresholds)
                       & hit_mask(z, profiles['gwaspri'][idx], config.thresholds))
            recurrent += reached
            coverage[rep, j] = reached.sum()
        K_null[rep] = recurrent
        if (rep+1) % 250 == 0 or rep+1 == reps:
            progress(f'Landscape matched-seed null: {rep+1}/{reps}', flush=True)
    cores_null = K_null[:, None, :] >= np.asarray(thresholds)[None, :, None]  # draw x threshold x gene
    sizes = cores_null.sum(axis=2)
    shared = (cores_null & cores_obs[None]).sum(axis=2)
    union = (cores_null | cores_obs[None]).sum(axis=2)
    jaccard = np.where(union > 0, shared/np.maximum(union, 1), np.nan)
    np.savez_compressed(out/'draws.npz', k_null=K_null, cnv_coverage=coverage, core_sizes=sizes,
                        overlap_with_observed=shared, thresholds=thresholds, cnvs=names, gene_order=nodes)

    # Per gene: do random seed sets reach THIS gene with both targets in >= K_obs CNVs as often?
    genes = pd.DataFrame({'entrez': nodes, 'symbol': [symbols.get(int(g), str(g)) for g in nodes],
                          'k_observed': K_obs, 'k_null_mean': K_null.mean(axis=0),
                          'n_draws_k_ge_observed': (K_null >= K_obs).sum(axis=0)})
    genes['p_gene'] = (1 + genes.n_draws_k_ge_observed)/(reps + 1)
    genes['q_gene'] = bh(genes.p_gene)  # all landscape genes; K_obs = 0 gives P = 1
    for j, k in enumerate(thresholds):
        genes[f'in_observed_core_min{k}'] = cores_obs[j]
        genes[f'random_core_frequency_min{k}'] = cores_null[:, j].mean(axis=0)
    genes['cnvs_both'] = [','.join(n for n in names if both[n][i]) for i in idx]
    genes = genes.sort_values(['k_observed', 'p_gene', 'symbol'], ascending=[False, True, True])
    save(genes, out/'gene_identity_tests.tsv')

    # Per core: how many of the observed core genes does a random core contain?
    rows = []
    for j, k in enumerate(thresholds):
        n_obs = int(cores_obs[j].sum())
        core = genes[genes[f'in_observed_core_min{k}']]
        specific = core.q_gene < config.fdr
        rows.append({'min_cnvs': k, 'observed_core_genes': n_obs,
                     'null_mean_core_genes': sizes[:, j].mean(),
                     'null_mean_shared_with_observed': shared[:, j].mean(),
                     'null_shared_low': np.quantile(shared[:, j], .025),
                     'null_shared_high': np.quantile(shared[:, j], .975),
                     'null_mean_fraction_recovered': shared[:, j].mean()/n_obs if n_obs else np.nan,
                     'null_mean_jaccard': np.nanmean(jaccard[:, j]) if np.isfinite(jaccard[:, j]).any() else np.nan,
                     'draws_containing_whole_observed_core': int((shared[:, j] == n_obs).sum()) if n_obs else 0,
                     'observed_genes_q_gene_lt_fdr': int(specific.sum()),
                     'genes_q_gene_lt_fdr': ','.join(core.symbol[specific].astype(str)),
                     'genes_q_gene_ge_fdr': ','.join(core.symbol[~specific].astype(str))})
    identity = pd.DataFrame(rows)
    save(identity, out/'core_gene_identity.tsv')
    save(pd.DataFrame([{'replicate': r, 'min_cnvs': k, 'random_core_genes': int(sizes[r, j]),
                        'shared_with_observed': int(shared[r, j]), 'jaccard': jaccard[r, j]}
                       for r in range(reps) for j, k in enumerate(thresholds)]), out/'null_core_overlap.tsv')

    # Secondary: core size and per-CNV coverage under the same matched draws.
    tests = pd.DataFrame([{'min_cnvs': k, **tail(cores_obs[j].sum(), sizes[:, j])}
                          for j, k in enumerate(thresholds)])
    tests['q_core_thresholds'] = bh(tests.p)
    save(tests, out/'core_size_tests.tsv')
    per_cnv = pd.DataFrame([{'cnv': n, **tail(observed_by_cnv[j], coverage[:, j])}
                            for j, n in enumerate(names)])
    per_cnv['q_cnv_coverage'] = bh(per_cnv.p)
    save(per_cnv, out/'cnv_coverage_tests.tsv')
    return tests, identity, genes


def export_graph(A, nodes, genes, out):
    out.mkdir(parents=True, exist_ok=True)
    pos = {int(g): i for i, g in enumerate(nodes)}
    ids = genes.entrez.astype(int).tolist()
    selected = [pos[g] for g in ids]
    sub = A[selected][:, selected].tocoo()
    graph = nx.Graph()
    for record in genes.to_dict('records'):
        gene = str(record.pop('entrez'))
        graph.add_node(gene, **{k: ('' if pd.isna(v) else v) for k, v in record.items()})
    graph.add_edges_from((str(ids[i]), str(ids[j])) for i, j in zip(sub.row, sub.col) if i < j)
    genes = genes.copy()
    genes['degree_in_subnetwork'] = [graph.degree(str(g)) for g in ids]
    save(genes, out/'genes.tsv')
    save(pd.DataFrame(graph.edges(), columns=['source', 'target']), out/'edges.tsv')
    save(genes[genes.degree_in_subnetwork == 0], out/'isolates.tsv')
    write_graphml(graph, out/'network.graphml')
    return graph


def enrich(queries, background, paths, out, fdr, title):
    out.mkdir(parents=True, exist_ok=True)
    results = context_ora(queries, background, paths, min_term_size=5)
    save(results, out/'all_terms.tsv')
    save(results[results.q_group < fdr], out/'significant_terms.tsv')
    summary = context_summary(results, queries, fdr)
    save(summary, out/'context_summary.tsv')
    save(pd.DataFrame({'symbol': sorted(background)}), out/'background.tsv')
    plot_context_heatmap(results, {q: f'{q} (n={len(g)})' for q, g in queries.items()}, fdr,
                         out/'tissue_celltype', title)
    # Compact functional figure; full results and gene hits remain in the tables.
    import matplotlib.pyplot as plt
    selected = results[(results.cluster == next(iter(queries))) & (results.group == 'function')
                       & (results.q_group < fdr)].head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    if len(selected):
        import textwrap
        ax.barh(range(len(selected)), -np.log10(selected.q_group.clip(lower=1e-300)), color='#4477AA')
        ax.set_yticks(range(len(selected)), [textwrap.fill(t, 55) for t in selected.term], fontsize=9)
        ax.set_xlabel('−log₁₀(BH q within annotation group, across all queries)')
    else:
        ax.text(.5, .5, 'No functional terms pass FDR', ha='center', transform=ax.transAxes)
    ax.set_title(title+'\nDescriptive enrichment after network selection')
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'function_enrichment.{ext}', dpi=180)
    plt.close(fig)
    summarize_enrichment(results, paths, out, fdr)
    return summary


def plot_landscape(graph, genes, assignments, out, seed, names=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    # Lay out the giant component at full scale. Pack the remaining components
    # into a labelled strip so isolates cannot compress the biological network.
    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(map(int, c))))
    if not components:
        save(pd.DataFrame(columns=['entrez', 'x', 'y']), out/'layout.tsv')
        for name in ('modules', 'modules_all_labels', 'cnv_recurrence'):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(.5, .5, 'No genes meet the SCHEMA × GWAS landscape definition', ha='center')
            ax.axis('off')
            for ext in ('png', 'pdf'):
                fig.savefig(out/f'{name}.{ext}', dpi=180)
            plt.close(fig)
        return
    pos = nx.spring_layout(graph.subgraph(components[0]), seed=seed, iterations=150)
    extras = components[1:]
    for j, component in enumerate(extras):
        local = nx.spring_layout(graph.subgraph(component), seed=seed, iterations=100)
        centre = -1 + 2*(j+.5)/max(1, len(extras))
        for node, xy in local.items():
            pos[node] = np.array([centre, -1.3]) + xy*min(.10, .7/max(1, len(extras)))
    save(pd.DataFrame([{'entrez': g, 'x': xy[0], 'y': xy[1]} for g, xy in pos.items()]), out/'layout.tsv')
    values = genes.set_index(genes.entrez.astype(str))
    palette = cluster_colors(assignments.values())
    overview_labels = set(graph) if len(graph) <= 80 else set()
    if not overview_labels:
        for cluster in sorted(set(assignments.values())):
            candidates = sorted((g for g in graph if assignments.get(g) == cluster),
                                key=lambda g: (-graph.degree(g), int(g)))
            selected = []
            for gene in candidates:
                if all(np.linalg.norm(pos[gene]-pos[other]) >= .10 for other in overview_labels):
                    selected.append(gene)
                    overview_labels.add(gene)
                if len(selected) == 3:
                    break
    save(pd.DataFrame([{'entrez': g, 'symbol': values.at[g, 'symbol'],
                       'labelled_in_overview': g in overview_labels} for g in graph]), out/'figure_labels.tsv')
    for name, title in [('modules', 'SCHEMA × GWAS schizophrenia disorder landscape'),
                         ('modules_all_labels', 'Schizophrenia landscape: all gene labels'),
                         ('cnv_recurrence', 'CNV convergence within the schizophrenia landscape')]:
        fig, ax = plt.subplots(figsize=(13, 11), constrained_layout=True)
        ax.add_collection(LineCollection([(pos[a], pos[b]) for a, b in graph.edges()],
                                         colors='#aaaaaa', linewidths=.25, alpha=.25))
        if name.startswith('modules'):
            colors = [palette[assignments[g]] if g in assignments else '#bbbbbb'
                      for g in graph]
            nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=22)
            from matplotlib.patches import Patch
            import textwrap
            nx.draw_networkx_labels(graph, pos, ax=ax,
                labels={g: str(values.at[g, 'symbol']) for g in graph
                        if name == 'modules_all_labels' or g in overview_labels},
                font_size=5 if name == 'modules_all_labels' else 7)
            handles = [Patch(color=palette[c],
                             label=textwrap.fill(f'C{c}: {(names or {}).get(c, "No enriched biological process")}', 45))
                       for c in sorted(set(assignments.values()))]
            if handles:
                ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=7)
        else:
            artist = nx.draw_networkx_nodes(graph, pos, ax=ax,
                node_color=[values.at[g, 'n_cnvs_both'] for g in graph], cmap=plt.get_cmap('viridis'),
                vmin=0, vmax=max(1, int(genes.n_cnvs_both.max())), node_size=22)
            fig.colorbar(artist, ax=ax, shrink=.55, ticks=range(int(genes.n_cnvs_both.max())+1),
                         label='Distinct CNVs qualifying with BOTH targets')
        if extras:
            ax.text(0, -1.48, f'Other components and isolates ({sum(map(len, extras))} genes retained)',
                    ha='center', fontsize=9, color='#555555')
        ax.set_title(f'{title}\n{len(graph)} genes; {graph.number_of_edges()} interactome edges; identical saved layout')
        if name == 'modules' and len(graph) > 80:
            ax.set_title(ax.get_title() + '\nSelected gene labels; all labels in modules_all_labels.pdf', fontsize=11)
        ax.axis('off')
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{ext}', dpi=180)
        plt.close(fig)


def plot_membership(genes, both, nodes, names, threshold, out):
    import matplotlib.pyplot as plt
    pos = {int(g): i for i, g in enumerate(nodes)}
    core = genes[genes.n_cnvs_both >= threshold].sort_values(['n_cnvs_both', 'symbol'], ascending=[False, True])
    idx = [pos[int(g)] for g in core.entrez]
    matrix = pd.DataFrame({n: both[n][idx].astype(int) for n in names}, index=core.symbol)
    matrix.to_csv(out/'core_gene_cnv_matrix.tsv', sep='\t')
    fig, ax = plt.subplots(figsize=(10, max(4, .27*len(core)+2)), constrained_layout=True)
    if len(core):
        ax.imshow(matrix.to_numpy(), aspect='auto', cmap='Blues', vmin=0, vmax=1)
        ax.set_yticks(range(len(core)), core.symbol, fontsize=8)
        ax.set_xticks(range(len(names)), names, rotation=45, ha='right')
    else:
        ax.text(.5, .5, 'No genes meet the core definition', ha='center', transform=ax.transAxes)
    ax.set_title(f'Shared CNV core within the schizophrenia landscape (≥{threshold} CNVs)\n'
                 'Filled cells: the same CNV qualifies with both SCHEMA and GWAS')
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'core_gene_cnv_matrix.{ext}', dpi=180)
    plt.close(fig)


def plot_null(tests, primary, out):
    import matplotlib.pyplot as plt
    draws = np.load(out/'draws.npz')
    sizes = draws['core_sizes']
    low, high = np.quantile(sizes, [.025, .975], axis=0)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.fill_between(tests.min_cnvs, low, high, color='#cde2fb', label='Null 2.5–97.5 percentiles')
    ax.plot(tests.min_cnvs, tests.null_mean, 'o--', color='#4477AA', label='Matched random seed sets: mean')
    ax.plot(tests.min_cnvs, tests.observed, 'o-', color='#AA3377', label='Observed landscape core')
    ax.set_xticks(tests.min_cnvs)
    ax.set(xlabel='Minimum distinct CNVs qualifying with both targets', ylabel='Genes within fixed landscape',
           title=f'Landscape core size (secondary)\nPrimary threshold: ≥{primary} CNVs')
    ax.legend()
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'core_size_sensitivity.{ext}', dpi=180)
    plt.close(fig)


def plot_gene_identity(identity, genes, primary, fdr, out):
    """Left: observed core genes found again in each random core. Right: the same, per gene."""
    import matplotlib.pyplot as plt
    draws = np.load(out/'draws.npz')
    j = list(draws['thresholds']).index(primary)
    shared = draws['overlap_with_observed'][:, j]
    row = identity[identity.min_cnvs == primary].iloc[0]
    n_obs = int(row.observed_core_genes)
    core = genes[genes[f'in_observed_core_min{primary}']].sort_values(f'random_core_frequency_min{primary}')
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, max(5, .3*len(core)+2)), constrained_layout=True,
                                      gridspec_kw={'width_ratios': [1, 1.2]})
    left.hist(shared, bins=np.arange(n_obs+2)-.5, color='#4477AA')
    left.axvline(n_obs, color='#AA3377', lw=2, label=f'All {n_obs} observed core genes')
    left.set(xlabel='Observed core genes also in the random core', ylabel='Random seed sets',
             title=f'How many of the real core genes do random seed sets find?\n'
                   f'mean {row.null_mean_shared_with_observed:.2f} of {n_obs}')
    left.legend()
    if len(core):
        colors = ['#AA3377' if q < fdr else '#bbbbbb' for q in core.q_gene]
        right.barh(range(len(core)), core[f'random_core_frequency_min{primary}'], color=colors)
        right.set_yticks(range(len(core)), [f'{s} (K={k})' for s, k in zip(core.symbol, core.k_observed)], fontsize=8)
        right.set_xlim(0, 1)
    else:
        right.text(.5, .5, 'No observed core genes', ha='center', transform=right.transAxes)
    right.set(xlabel='Fraction of random seed sets that put this gene in the core',
              title=f'Per gene (pink: q < {fdr}, CNVs reach it more often than random seed sets)')
    fig.suptitle(f'Are the same genes found by chance? Landscape core ≥{primary} CNVs; '
                 f'{len(shared)} matched random seed sets without CNV, SCHEMA or GWAS seeds')
    for ext in ('pdf', 'png'):
        fig.savefig(out/f'gene_identity.{ext}', dpi=180)
    plt.close(fig)


def plot_cnv_pairs(pairs, names, out):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    graph = nx.Graph()
    graph.add_nodes_from(names)
    graph.add_edges_from((r.a, r.b) for r in pairs.itertuples())
    pos = nx.circular_layout(names)
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    scale = max(1., float(pairs.excess.max())) if len(pairs) else 1.
    ax.add_collection(LineCollection([(pos[r.a], pos[r.b]) for r in pairs.itertuples()],
        linewidths=[1+5*max(0, r.excess)/scale for r in pairs.itertuples()], colors='#6da7ec'))
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=280, node_color='#184f95')
    for n, xy in pos.items():
        ax.text(*(xy*1.16), n, ha='center', va='center', fontsize=9)
    ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), title=f'{len(pairs)} significant CNV × CNV pairs\n'
           'Edge width: observed-minus-expected shared genes; all tested CNVs shown')
    ax.axis('off')
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'cnv_pair_circle.{ext}', dpi=180)
    plt.close(fig)


def run_pipeline(settings, progress=print):
    network = settings['network']
    spec, root, inputs = load_inputs(network, settings.get('min_seeds', 3),
        settings.get('convergence_config'), settings.get('disorder_config'))
    config = AnalysisConfig.from_dict(spec['analysis'])
    if config.overlap != 'bin':
        raise ValueError('Organised enrichment currently requires overlap=bin; do not silently change backgrounds')
    thresholds = tuple(sorted(set(settings.get('core_thresholds', [2, 3, 4]))))
    primary = settings.get('primary_min_cnvs', 3)
    reps = settings.get('null_reps', 5000)
    if primary not in thresholds or any(type(k) is not int or k < 1 for k in thresholds) or type(reps) is not int or reps < 2:
        raise ValueError('Require positive integer thresholds including the primary threshold, and null_reps >= 2')
    A, nodes, symbols, entities, *_ = inputs
    prop = Propagator(A, nodes, root/spec['cache_dir'], config.propagation)
    A, nodes = prop.A, prop.nodes
    names = sorted(n for n, e in entities.items() if e['kind'] == 'cnv')
    pair_run = ROOT/settings['cnv_pair_run']
    pair_manifest = json.loads((pair_run/'manifest.json').read_text())
    if pair_manifest.get('status') != 'complete' or pair_manifest['identity']['graph'] != prop.graph_id:
        raise ValueError('CNV pair source must be complete and use this exact graph')
    original = pair_manifest['identity']
    if original['config'] != json.loads(json.dumps(asdict(config))):
        raise ValueError('CNV pair source analysis settings differ')
    if {n: e for n, e in entities.items() if e['kind'] == 'cnv'} != original['entities']:
        raise ValueError('CNV pair source seed definitions differ')
    for f, checksum in pair_manifest.get('artifacts', {}).items():
        if file_hash(pair_run/f) != checksum:
            raise ValueError(f'CNV pair artifact changed: {f}')
    paths = _context_paths(ROOT/'input/annotations')
    dependencies = ['scz_landscape_pipeline.py', 'cnv_cosine_similarity.py', 'notebook_analysis.py',
                    'scz_core_analysis.py', 'shared_network_cluster_context.py', 'shared_network_leiden.py',
                    'cnv_pair_network.py', 'scz_shared_network.py', 'network_inputs.py',
                    'publication_enrichment.py', 'brain_pair_enrichment.py', 'shared_network_orsum.py',
                    'scz_restricted_core_null.py']
    identity = {'algorithm': 'organised_scz_landscape_v2_matched_seed_gene_identity', 'settings': settings, 'config': asdict(config),
                'propagation': prop.metadata, 'entities': entities, 'software': software(),
                'symbols': {str(g): symbols.get(int(g), str(g)) for g in nodes},
                'pair_source': file_hash(pair_run/'manifest.json'),
                'code': {f: file_hash(ROOT/'scripts'/f) for f in dependencies},
                'package_code': {p.name: file_hash(p) for p in sorted((ROOT/'src/cnv_overlap').glob('*.py'))},
                'annotations': {n: file_hash(p) for n, p in paths.items()},
                'analysis_tools': analysis_tools()}
    paper = ROOT/'s44319-026-00733-4.pdf'
    if paper.exists():
        identity['reference_pdf_sha256'] = file_hash(paper)
    output_root = ROOT/settings.get('output_root', f'results/{network}/schizophrenia_landscape')
    out = run_dir(output_root, identity, 'landscape')
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        if old['status'] == 'complete':
            for f, checksum in old['artifacts'].items():
                if file_hash(out/f) != checksum:
                    raise ValueError(f'Completed landscape artifact changed: {f}')
            return out
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', {'status': 'running', 'identity': identity})
    landscape_dir, cnv_dir, pair_dir, cosine_dir = [out/n for n in
        ('01_schema_gwas_landscape', '02_cnv_in_landscape', '03_cnv_cnv', '04_cosine_similarity')]
    for directory in (landscape_dir, cnv_dir, pair_dir, cosine_dir):
        directory.mkdir(exist_ok=True)
    progress('Loading cached NPS profiles', flush=True)
    seeds = {n: sorted(set(e['seeds']) & prop.pos.keys()) for n, e in entities.items()}
    profiles = {n: np.asarray(prop.nps(s)) for n, s in seeds.items()}
    table, pair_masks, both = landscape_membership(nodes, symbols, entities, profiles, config, primary)
    landscape = table.in_scz_landscape.to_numpy()
    save(table, out/'all_gene_membership.tsv')
    genes = table[landscape].copy()
    save(pd.DataFrame([{'entity': n, 'input_seeds': len(set(entities[n]['seeds'])),
                       'retained_seeds': len(s), 'excluded_seeds': ','.join(map(str, sorted(set(entities[n]['seeds'])-set(s))))}
                      for n, s in seeds.items()]), out/'seed_qc.tsv')
    progress(f'SCHEMA × GWAS landscape: {len(genes)} genes; testing target pair', flush=True)
    target_run = run_target_pair(prop, symbols, entities, config, output_root=landscape_dir/'pair_test',
                                provenance={'source': 'organised_scz_landscape'}, fdr=config.fdr)
    target_genes = pd.read_csv(target_run/'candidate_genes.tsv', sep='\t')
    if set(target_genes.entrez) != set(genes.entrez):
        raise ValueError('Landscape disagrees with independently exported target-pair candidates')
    shutil.copy2(target_run/'pair_tests.tsv', landscape_dir/'pair_tests.tsv')
    graph = export_graph(A, nodes, genes, landscape_dir)
    assignments = leiden_membership(graph, seed=config.propagation.seed,
                                    resolution=settings.get('leiden_resolution', 1.0))
    genes['leiden_cluster'] = genes.entrez.astype(str).map(assignments).astype('Int64')
    save(genes, landscape_dir/'gene_modules.tsv')
    min_cluster = settings.get('min_cluster_genes', 5)
    queries = {'landscape': set(genes.symbol.astype(str))}
    for cluster, group in genes.groupby('leiden_cluster'):
        if len(group) >= min_cluster:
            queries[f'C{cluster}'] = set(group.symbol.astype(str))
    save(pd.DataFrame([{'module': f'C{c}', 'genes': len(g), 'tested_for_enrichment': len(g) >= min_cluster}
                       for c, g in genes.groupby('leiden_cluster')]), landscape_dir/'modules.tsv')
    progress('Annotating landscape and modules against the tested interactome', flush=True)
    enrich(queries, set(table.symbol.astype(str)), paths, landscape_dir/'enrichment', config.fdr,
           'Schizophrenia disorder landscape and modules')
    annotation = module_names(genes, pd.read_csv(landscape_dir/'enrichment/all_terms.tsv', sep='\t'), config.fdr)
    palette = cluster_colors(assignments.values())
    annotation['color'] = annotation.leiden_cluster.map(palette)
    save(annotation, landscape_dir/'module_names.tsv')
    labels = dict(zip(annotation.leiden_cluster, annotation.biological_process))
    genes['cluster_biological_process'] = genes.leiden_cluster.map(labels)
    genes['cluster_color'] = genes.leiden_cluster.map(palette).fillna('#bbbbbb')
    save(genes, landscape_dir/'gene_modules.tsv')
    nx.set_node_attributes(graph, {n: int(c) for n, c in assignments.items()}, 'leiden_cluster')
    nx.set_node_attributes(graph, {n: labels[c] for n, c in assignments.items()}, 'cluster_biological_process')
    nx.set_node_attributes(graph, {n: palette.get(assignments.get(n), '#bbbbbb') for n in graph}, 'cluster_color')
    write_graphml(graph, landscape_dir/'network.graphml')
    plot_landscape(graph, genes, assignments, landscape_dir, config.propagation.seed, labels)
    save(genes, cnv_dir/'landscape_gene_membership.tsv')
    reached = pd.DataFrame({'entrez': genes.entrez, 'symbol': genes.symbol})
    for n in names:
        reached[n] = profiles[n][landscape] > config.thresholds.individual
    save(reached, cnv_dir/'propagation_reached_gene_cnv.tsv')
    save(pd.DataFrame([{'cnv': n, 'reached_genes': int(reached[n].sum()),
                       'nonseed_reached_genes': int((reached[n] & ~genes[f'seed_{n}']).sum()),
                       'fraction_of_landscape': float(reached[n].mean())} for n in names]),
         cnv_dir/'propagation_coverage.tsv')
    coverage = pd.DataFrame([{'cnv': n, 'landscape_genes': int((both[n] & landscape).sum()),
        'fraction_of_landscape': float((both[n] & landscape).sum()/landscape.sum()) if landscape.any() else np.nan,
        'both_targets_all_genes': int(both[n].sum()),
        'both_targets_outside_landscape': int((both[n] & ~landscape).sum())} for n in names])
    save(coverage, cnv_dir/'cnv_coverage.tsv')
    save(genes.groupby('n_cnvs_both').size().rename('genes').reset_index(), cnv_dir/'recurrence_distribution.tsv')
    module_rows = []
    for cluster, group in genes.groupby('leiden_cluster'):
        for n in names:
            count = int(both[n][group.index.to_numpy()].sum())
            module_rows.append({'module': f'C{cluster}', 'cnv': n, 'module_genes': len(group),
                                'reached_genes': count, 'fraction_of_module': count/len(group)})
    save(pd.DataFrame(module_rows), cnv_dir/'cnv_module_coverage.tsv')
    core_queries = {}
    for k in thresholds:
        core = genes[genes.n_cnvs_both >= k]
        core_queries[f'core_min{k}'] = set(core.symbol.astype(str))
        export_graph(A, nodes, core, cnv_dir/f'core_min{k}')
    excluded = table[table.shared_core_unrestricted & ~landscape]
    save(excluded, cnv_dir/'unrestricted_primary_core_outside_landscape.tsv')
    # Conditional ORA asks whether a core favours processes beyond those in the landscape itself.
    progress('Annotating CNV cores against the fixed landscape background', flush=True)
    enrich(core_queries, set(genes.symbol.astype(str)), paths, cnv_dir/'enrichment', config.fdr,
           'CNV cores within the schizophrenia landscape')
    plot_membership(genes, both, nodes, names, primary, cnv_dir)
    null_dir = cnv_dir/'conditional_null'
    tests, gene_identity, gene_tests = conditional_null(prop, profiles, seeds, names, landscape, both, config,
                                                        thresholds, reps, null_dir, symbols, progress)
    plot_null(tests, primary, null_dir)
    plot_gene_identity(gene_identity, gene_tests, primary, config.fdr, null_dir)
    progress('Exporting CNV cosine heatmaps and CNV-pair evidence', flush=True)
    export_cosines(profiles, names, pair_masks, landscape, nodes, config.thresholds.individual, cosine_dir)
    pairs = significant_pairs(pair_run, fdr=config.fdr)
    save(pairs, pair_dir/'significant_pairs.tsv')
    shutil.copy2(pair_run/'pair_tests.tsv', pair_dir/'all_pair_tests.tsv')
    plot_cnv_pairs(pairs, names, pair_dir)
    from cnv_overlap.visualization import pair_heatmap
    import matplotlib.pyplot as plt
    figure = pair_heatmap(pair_run, family='region_pair', metric='z', fdr=config.fdr)
    for ext in ('pdf', 'png'):
        figure.savefig(pair_dir/f'cnv_pair_z_scores.{ext}', dpi=180, bbox_inches='tight')
    plt.close(figure)
    pair_network = None
    if len(pairs):
        pair_network = build_pair_network(pair_run, inputs, annotation_dir=ROOT/'input/annotations',
            output_root=pair_dir/'gene_network', fdr=config.fdr, seed=config.propagation.seed,
            resolution=settings.get('leiden_resolution', 1.0), min_cluster_genes=min_cluster)
    primary_test = tests[tests.min_cnvs == primary].iloc[0]
    primary_identity = gene_identity[gene_identity.min_cnvs == primary].iloc[0]
    target_test = pd.read_csv(landscape_dir/'pair_tests.tsv', sep='\t').query("statistic == 'size'").iloc[0]
    summary = pd.DataFrame([{'analysis': 'schema_gwas_landscape', 'genes': len(genes),
                            'null_mean': target_test.null_mean, 'p': target_test.p, 'q': target_test.q,
                            'inference': 'NetColoc normal-tail label-null test; see empirical diagnostic'},
        *[{'analysis': f'landscape_core_min{r.min_cnvs}', 'genes': int(r.observed),
           'null_mean': r.null_mean, 'p': r.p, 'q': r.q_core_thresholds,
           'null_mean_shared_with_observed': i.null_mean_shared_with_observed,
           'null_mean_jaccard': i.null_mean_jaccard,
           'genes_q_gene_lt_fdr': i.observed_genes_q_gene_lt_fdr,
           'inference': 'matched random seed sets (no CNV/SCHEMA/GWAS seeds); size P/Q secondary, '
                        'gene identity in core_gene_identity.tsv and gene_identity_tests.tsv'}
          for r, i in zip(tests.itertuples(), gene_identity.itertuples())]])
    save(summary, out/'analysis_summary.tsv')
    index = [
        ('main', 'SCHEMA–GWAS landscape modules', '01_schema_gwas_landscape/modules.pdf'),
        ('main', 'CNV recurrence on identical landscape layout', '01_schema_gwas_landscape/cnv_recurrence.pdf'),
        ('main', 'Landscape-restricted primary core matrix', '02_cnv_in_landscape/core_gene_cnv_matrix.pdf'),
        ('main', 'Significant CNV-pair circular graph', '03_cnv_cnv/cnv_pair_circle.pdf'),
        ('complementary', 'CNV propagation cosine', '04_cosine_similarity/cnv_nps_pair_union.pdf'),
        ('complementary', 'Paper-style CNV × SCHEMA joint cosine', '04_cosine_similarity/cnv_schema_joint_pair_union.pdf'),
        ('complementary', 'Paper-style CNV × GWAS joint cosine', '04_cosine_similarity/cnv_gwaspri_joint_pair_union.pdf'),
        ('sensitivity', 'Whole-interactome cosine', '04_cosine_similarity/cnv_nps_full_network.pdf'),
        ('sensitivity', 'Fixed-landscape cosine', '04_cosine_similarity/cnv_nps_scz_landscape.pdf'),
        ('main', 'Are the same core genes found by random seed sets?', '02_cnv_in_landscape/conditional_null/gene_identity.pdf'),
        ('statistics', 'Per-gene matched-seed tests (landscape genes)', '02_cnv_in_landscape/conditional_null/gene_identity_tests.tsv'),
        ('statistics', 'Observed-core genes recovered by random cores', '02_cnv_in_landscape/conditional_null/core_gene_identity.tsv'),
        ('statistics', 'Landscape core-size null tests (secondary)', '02_cnv_in_landscape/conditional_null/core_size_tests.tsv'),
        ('qc', 'Genes excluded from random seed sets', '02_cnv_in_landscape/conditional_null/excluded_random_seeds.tsv'),
        ('sensitivity', 'Landscape core-size null figure', '02_cnv_in_landscape/conditional_null/core_size_sensitivity.pdf'),
        ('statistics', 'Per-CNV landscape coverage null tests', '02_cnv_in_landscape/conditional_null/cnv_coverage_tests.tsv'),
        ('source', 'Completed CNV-pair source', str(pair_run.resolve())),
    ]
    if pair_network:
        index.append(('supporting', 'CNV-pair gene network and modules', str(pair_network.relative_to(out))))
    save(pd.DataFrame(index, columns=['role', 'result', 'path']), out/'results_index.tsv')
    (out/'README.md').write_text(
        '# Schizophrenia disorder landscape\n\n'
        f'Network: {network}. SCHEMA × GWAS defines the landscape independently of CNVs: **{len(genes)} genes**. '
        f'Individual NPS > {config.thresholds.individual}; NPS product ≥ {config.thresholds.product}. '
        'All nodes, including isolates and genes with no CNV support, are retained.\n\n'
        f'Primary core: landscape genes where the same ≥{primary} CNVs each qualify with BOTH targets: '
        f'**{int(primary_test.observed)} genes**. The unrestricted core has '
        f'{int(table.shared_core_unrestricted.sum())} genes; {len(excluded)} fall outside the landscape and are separately listed.\n\n'
        f'Matched random seed sets ({reps} joint draws): every CNV seed is replaced by a gene from the same '
        'degree bin, never a CNV, SCHEMA or GWAS seed; seed counts and between-CNV seed overlaps are kept; '
        'SCHEMA/GWAS and the landscape stay fixed. No genomic-window null is used.\n\n'
        f'**Same genes by chance?** A random ≥{primary}-CNV core contains on average '
        f'{primary_identity.null_mean_shared_with_observed:.2f} of the {int(primary_identity.observed_core_genes)} '
        f'observed core genes (mean Jaccard {primary_identity.null_mean_jaccard:.3f}). '
        f'{int(primary_identity.observed_genes_q_gene_lt_fdr)} observed core genes are reached by more CNVs than '
        f'random seed sets reach them (per-gene BH q < {config.fdr} over all landscape genes): '
        f'{primary_identity.genes_q_gene_lt_fdr or "none"}. '
        'Per-gene P = (1 + draws with random K ≥ observed K)/(B + 1).\n\n'
        f'Core size (secondary): null mean {primary_test.null_mean:.3f}, '
        f'empirical P={primary_test.p:.6g}, BH Q={primary_test.q_core_thresholds:.6g} across '
        f'{len(thresholds)} core thresholds. Per-CNV coverage is a separate BH family. '
        'All thresholds and the primary definition are exploratory continuations of the existing analysis. '
        'The null does not preserve genomic contiguity. '
        'Full-interactome propagation and reference normalisation remain unchanged.\n\n'
        '## Results organisation\n\n'
        '- `01_schema_gwas_landscape`: standalone pair test, all genes/edges, Leiden modules, '
        'functional and tissue/cell-type enrichment against the tested interactome, and two figures sharing one layout.\n'
        '- `02_cnv_in_landscape`: coverage, module coverage, recurrence, cores at each threshold, '
        'gene × CNV matrix, conditional nulls, and core enrichment against the landscape background.\n'
        '- `03_cnv_cnv`: all pair statistics, significant pairs, circular graph including unconnected CNVs, '
        'and the induced union gene network with its modules and annotations.\n'
        '- `04_cosine_similarity`: complementary cosine and joint-product heatmaps, '
        'universe sensitivities, exact score/selection inputs and pair-specific union sizes.\n\n'
        'Use `results_index.tsv` for main, complementary and sensitivity outputs, and `analysis_summary.tsv` '
        'for numeric findings. Gene tables distinguish input seeds from propagated genes. '
        'Leiden singleton communities have no assigned module; all genes remain in the landscape. '
        'Small modules are listed even when not tested for enrichment. '
        'Enrichment is descriptive after selection; BH is within annotation group across all queries in '
        'each enrichment directory. It does not calibrate the network selection process.\n\n'
        'The paper’s method is documented in `04_cosine_similarity/README.md`. Cosine does not determine '
        'the edges in the significant CNV-pair graph.\n')
    write_json(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'artifacts': {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*'))
                      if p.is_file() and p != out/'manifest.json'}})
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root/'current.json', {'run': str(out.relative_to(ROOT)),
                                           'manifest_sha256': file_hash(out/'manifest.json')})
    (output_root/'README.md').write_text(f'# Current organised analysis\n\n[Read results]({out.name}/README.md)\n\n'
                                       f'[Results index]({out.name}/results_index.tsv)\n\n'
                                       'Run `.venv/bin/python scripts/scz_landscape_pipeline.py --config configs/scz_landscape.json`.\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/scz_landscape.json'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    print(run_pipeline(json.loads(Path(args.config).read_text())), flush=True)


if __name__ == '__main__':
    main()
