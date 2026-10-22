"""Induced gene network of significant CNV pairs with Leiden and context annotations."""
from cnv_overlap.results import finalize_manifest, resolve_path
import argparse
import json
import textwrap
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from cnv_overlap.provenance import file_hash, graph_hash, run_dir, write_json
from cnv_overlap.reporting import write_graphml
from shared_network_cluster_context import (CONTEXT_LIBRARIES, SEQUENTIAL, _context_paths, _reuse_complete,
                                            context_ora, context_summary, plot_context_heatmap)
from shared_network_leiden import leiden_membership

NETWORK = 'network'
# Categorical slots for the largest clusters; smaller clusters share one neutral colour.
CLUSTER_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
OTHER_COLOR = '#b4b2a9'


def significant_pairs(run, *, fdr=.05, null='netcoloc'):
    """Return size-test pairs below the FDR cutoff, sorted by q-value."""
    tests = pd.read_csv(resolve_path(Path(run)/'pair_tests.tsv'), sep='\t')
    pairs = tests[(tests['null'] == null) & (tests.statistic == 'size') & (tests.q < fdr)]
    pairs = pairs.sort_values('q')[['a', 'b', 'observed', 'null_mean', 'excess', 'z', 'p', 'q']].copy()
    pairs['pair'] = pairs.a + ' × ' + pairs.b
    return pairs.reset_index(drop=True)


def pair_gene_membership(run, pairs, symbols):
    """One row per union gene: supporting significant pairs, CNVs and seed status."""
    candidates = pd.read_csv(resolve_path(Path(run)/'candidate_genes.tsv'), sep='\t').merge(pairs[['a', 'b', 'pair']])
    rows = []
    for entrez, group in candidates.groupby('entrez'):
        cnvs = sorted(set(group.a) | set(group.b))
        seed_of = sorted(set(group.a[group.seed_a]) | set(group.b[group.seed_b]))
        rows.append({'entrez': int(entrez), 'symbol': symbols.get(int(entrez), str(entrez)),
                     'n_pairs': len(group), 'pairs': '; '.join(p for p in pairs.pair if p in set(group.pair)),
                     'n_cnvs': len(cnvs), 'cnvs': '; '.join(cnvs), 'seed_of': '; '.join(seed_of),
                     'mean_joint_nps': float(np.sqrt(group.nps_a.clip(lower=0)*group.nps_b.clip(lower=0)).mean())})
    return pd.DataFrame(rows).sort_values(['n_pairs', 'mean_joint_nps'], ascending=False).reset_index(drop=True)


def build_pair_network(run, inputs, *, annotation_dir, output_root, fdr=.05, null='netcoloc',
                       seed=42, resolution=1.0, min_cluster_genes=5, min_term_size=5):
    """Export the union of significant-pair genes, Leiden modules and annotations.

    Return the checksummed output directory. Require matching network inputs,
    overlap=bin and at least one significant pair."""
    run, annotation_dir = Path(run), Path(annotation_dir)
    manifest = json.loads((run/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('CNV x CNV run is incomplete')
    if manifest['identity']['config']['overlap'] != 'bin':
        raise ValueError('Pair network supports overlap=bin only (the background is every network gene)')
    A, nodes, symbols = inputs[:3]
    if graph_hash(A, nodes) != manifest['identity']['graph']:
        raise ValueError('Inputs use a different network than the CNV x CNV run')
    paths = _context_paths(annotation_dir)
    identity = {'source': str(run.resolve()), 'run_manifest': file_hash(run/'manifest.json'), 'graph': manifest['identity']['graph'],
                'code': file_hash(__file__), 'fdr': fdr, 'null': null, 'seed': seed, 'resolution': resolution,
                'min_cluster_genes': min_cluster_genes, 'min_term_size': min_term_size,
                'correction': 'BH within annotation group across whole network and clusters',
                'libraries': {name: file_hash(p) for name, p in paths.items()}}
    out = run_dir(output_root, identity, 'significant-pairs', f'fdr{fdr}')
    if _reuse_complete(out):
        return out
    pairs = significant_pairs(run, fdr=fdr, null=null)
    if pairs.empty:
        raise ValueError(f'No CNV x CNV pair reaches q < {fdr}')
    membership = pair_gene_membership(run, pairs, symbols)

    pos = {int(g): i for i, g in enumerate(nodes)}
    ids = [g for g in membership.entrez if g in pos]
    sub = A[[pos[g] for g in ids]][:, [pos[g] for g in ids]].tocoo()
    graph = nx.Graph()
    graph.add_nodes_from(str(g) for g in ids)
    graph.add_edges_from((str(ids[i]), str(ids[j])) for i, j in zip(sub.row, sub.col) if i < j)
    membership['degree_in_pair_network'] = membership.entrez.astype(str).map(dict(graph.degree())).astype(int)
    membership['degree_in_background'] = [float(A[pos[g]].sum()) for g in membership.entrez]

    assignments = leiden_membership(graph, seed=seed, resolution=resolution)
    membership['leiden_cluster'] = membership.entrez.astype(str).map(assignments).astype('Int64')
    clustered = membership[membership.leiden_cluster.notna()].copy()
    singletons = membership[membership.leiden_cluster.isna()].drop(columns='leiden_cluster').assign(
        reason=lambda d: ['isolated' if graph.degree(str(g)) == 0 else 'size-1 community' for g in d.entrez])

    background = {str(symbols.get(int(g), g)) for g in nodes}
    queries = {NETWORK: set(membership.symbol.astype(str))}
    sizes = clustered.leiden_cluster.value_counts()
    for cluster in sorted(sizes.index):
        if sizes[cluster] >= min_cluster_genes:
            queries[str(cluster)] = set(clustered.symbol[clustered.leiden_cluster == cluster].astype(str))
    enrichment = context_ora(queries, background, paths, min_term_size)
    small = {str(c): set(g.symbol.astype(str)) for c, g in clustered.groupby('leiden_cluster')
             if str(c) not in queries}
    context = context_summary(enrichment, queries, fdr, untested=small)

    composition = cluster_pair_composition(clustered, pairs)
    clusters = cluster_table(clustered, composition, enrichment, fdr, min_cluster_genes)
    labels = dict(zip(clusters.leiden_cluster, clusters.annotation))
    clustered['cluster_annotation'] = clustered.leiden_cluster.map(labels)
    for gene, cluster in assignments.items():
        graph.nodes[gene]['leiden_cluster'] = int(cluster)
        graph.nodes[gene]['cluster_annotation'] = str(labels[cluster])
    for record in membership.to_dict('records'):
        attributes = graph.nodes[str(record['entrez'])]
        attributes.update({k: record[k] for k in ('symbol', 'n_pairs', 'pairs', 'n_cnvs', 'cnvs', 'seed_of')})

    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(resolve_path(out/'significant_pairs.tsv'), sep='\t', index=False)
    membership.to_csv(resolve_path(out/'gene_membership.tsv'), sep='\t', index=False)
    clustered.to_csv(resolve_path(out/'genes.tsv'), sep='\t', index=False)
    singletons.to_csv(resolve_path(out/'excluded_singletons.tsv'), sep='\t', index=False)
    pd.DataFrame(sorted(graph.edges()), columns=['a', 'b']).to_csv(resolve_path(out/'edges.tsv'), sep='\t', index=False)
    clusters.to_csv(resolve_path(out/'clusters.tsv'), sep='\t', index=False)
    composition.to_csv(resolve_path(out/'cluster_pair_composition.tsv'), sep='\t')
    enrichment.to_csv(resolve_path(out/'context_enrichment.tsv'), sep='\t', index=False)
    enrichment[enrichment.q_group < fdr].to_csv(resolve_path(out/'context_enrichment_significant.tsv'), sep='\t', index=False)
    context.to_csv(resolve_path(out/'cluster_context.tsv'), sep='\t', index=False)
    pd.DataFrame({'symbol': sorted(background)}).to_csv(resolve_path(out/'background_genes.tsv'), sep='\t', index=False)
    write_graphml(graph, resolve_path(out/'network.graphml'))
    plot_cnv_graph(pairs, out/'cnv_pair_graph')
    plot_clusters(graph, assignments, clusters, membership, seed, out/'leiden_clusters')
    plot_composition(composition, clusters, out/'cluster_pair_composition')
    heatmap_labels = {q: f'Whole network (n={len(m)})' if q == NETWORK else f'C{q} (n={len(m)})'
                      for q, m in queries.items()}
    plot_context_heatmap(enrichment, heatmap_labels, fdr, out/'tissue_celltype_enrichment',
                         'Brain cell-type enrichment: significant CNV × CNV network and its clusters')
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'interpretation': 'Union of candidate genes of significant CNV x CNV pairs; descriptive Leiden clusters '
                          'and ORA; BH within annotation group; not adjusted for pair or network selection',
        'artifacts': {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}})
    return out


def cluster_pair_composition(clustered, pairs):
    """Clusters x significant pairs: fraction of each cluster's genes that are candidates of that pair."""
    rows = {}
    for cluster, group in clustered.groupby('leiden_cluster'):
        supported = group.pairs.str.split('; ')
        rows[int(cluster)] = {p: supported.map(lambda s: p in s).mean() for p in pairs.pair}
    return pd.DataFrame.from_dict(rows, orient='index')[list(pairs.pair)].rename_axis('leiden_cluster')


def cluster_table(clustered, composition, enrichment, fdr, min_cluster_genes):
    rows = []
    for cluster, group in clustered.groupby('leiden_cluster'):
        cluster = int(cluster)
        hits = enrichment[(enrichment.cluster == str(cluster)) & (enrichment.library == 'GO Biological Process')
                          & (enrichment.q_group < fdr)].sort_values(['q_group', 'p'])
        if len(group) < min_cluster_genes:
            label = f'not tested (< {min_cluster_genes} genes)'
        else:
            label = hits.iloc[0].term if not hits.empty else 'No enriched biological process at FDR threshold'
        share = composition.loc[cluster].sort_values(ascending=False)
        rows.append({'leiden_cluster': cluster, 'n_genes': len(group), 'annotation': label,
                     'significant_function_synapse_terms': len(hits),
                     'n_multi_pair_genes': int((group.n_pairs >= 2).sum()),
                     'cnvs': '; '.join(sorted({c for s in group.cnvs for c in s.split('; ')})),
                     'main_pairs': '; '.join(f'{p} ({f:.0%})' for p, f in share[share >= .2].items()),
                     'hub_genes': ','.join(group.sort_values('degree_in_pair_network', ascending=False)
                                           .symbol.astype(str).head(8)),
                     'genes': ','.join(sorted(group.symbol.astype(str)))})
    return pd.DataFrame(rows).sort_values('leiden_cluster').reset_index(drop=True)


def plot_cnv_graph(pairs, stem):
    """CNVs joined by their significant pairs; edge width = z score of the overlap."""
    import matplotlib.pyplot as plt
    graph = nx.Graph()
    for r in pairs.itertuples():
        graph.add_edge(r.a, r.b, z=r.z, q=r.q)
    # Walk each component depth-first so connected CNVs sit next to each other on the circle.
    order = []
    for component in sorted(nx.connected_components(graph), key=len, reverse=True):
        start = min(component, key=lambda n: (graph.degree(n), n))
        order.extend(nx.dfs_preorder_nodes(graph.subgraph(component), start))
    layout = nx.circular_layout(order)
    fig, ax = plt.subplots(figsize=(7, 7))
    z = np.array([graph.edges[e]['z'] for e in graph.edges()])
    nx.draw_networkx_edges(graph, layout, ax=ax, width=1 + 6*z/z.max(), edge_color='#6da7ec')
    nx.draw_networkx_nodes(graph, layout, ax=ax, node_color='#184f95', node_size=260,
                           edgecolors='white', linewidths=2)
    for node, (x, y) in layout.items():
        ax.text(1.16*x, 1.16*y, node, ha='center', va='center', fontsize=9, fontweight='bold', color='#222222')
    ax.set_title(f'{len(pairs)} significant CNV × CNV pairs', fontsize=11)
    handles = [plt.Line2D([], [], color='#6da7ec', lw=1 + 6*v/z.max()) for v in (z.min(), z.max())]
    ax.legend(handles, [f'min z = {z.min():.1f}', f'max z = {z.max():.1f}'], title='Edge width: z score of the overlap',
              loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False, fontsize=8,
              title_fontsize=9, handlelength=3)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')
    for suffix in ('png', 'pdf'):
        fig.savefig(f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_clusters(graph, assignments, clusters, membership, seed, stem):
    """Clustered genes coloured by Leiden cluster; genes supported by >= 3 pairs are labelled."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch
    graph = graph.subgraph(assignments)
    colored = list(clusters.leiden_cluster[:len(CLUSTER_COLORS)])
    color = {c: CLUSTER_COLORS[i] for i, c in enumerate(colored)}
    fig, ax = plt.subplots(figsize=(15, 11))
    layout = grouped_layout(graph, assignments, clusters, seed)
    within = [(layout[a], layout[b]) for a, b in graph.edges() if assignments[a] == assignments[b]]
    between = [(layout[a], layout[b]) for a, b in graph.edges() if assignments[a] != assignments[b]]
    ax.add_collection(LineCollection(between, colors='#d3d1c7', linewidths=.1, alpha=.06, zorder=0))
    ax.add_collection(LineCollection(within, colors='#b4b2a9', linewidths=.15, alpha=.5, zorder=0))
    n_pairs = membership.set_index(membership.entrez.astype(str)).n_pairs
    order = sorted(graph, key=lambda g: n_pairs[g])
    nx.draw_networkx_nodes(graph, layout, ax=ax, nodelist=order,
                           node_color=[color.get(assignments[g], OTHER_COLOR) for g in order],
                           node_size=[12 + 30*(n_pairs[g] - 1) for g in order], edgecolors='white', linewidths=.4)
    labelled = {g: graph.nodes[g]['symbol'] for g in graph if n_pairs[g] >= 3}
    nx.draw_networkx_labels(graph, layout, ax=ax, labels=labelled, font_size=7, font_color='#222222')
    handles = [Patch(color=color[r.leiden_cluster], label=textwrap.fill(
        f'C{r.leiden_cluster} ({r.n_genes} genes): {r.annotation}', 50))
        for r in clusters.itertuples() if r.leiden_cluster in color]
    rest = clusters[~clusters.leiden_cluster.isin(colored)]
    if len(rest):
        handles.append(Patch(color=OTHER_COLOR, label=f'{len(rest)} smaller clusters ({rest.n_genes.sum()} genes)'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    ax.set_title('Union network of significant CNV × CNV pairs: Leiden communities\n'
                 'node size = number of significant pairs supporting the gene; labels = genes in ≥ 3 pairs; '
                 'darker edges within clusters, faint edges between',
                 fontsize=11)
    ax.axis('off')
    for suffix in ('png', 'pdf'):
        fig.savefig(f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def grouped_layout(graph, assignments, clusters, seed):
    """Each cluster laid out on its own and placed on a ring, scaled by the square root of its size."""
    sizes = clusters.set_index('leiden_cluster').n_genes
    radius = {c: np.sqrt(n) for c, n in sizes.items()}
    angles = np.cumsum([0] + [radius[c] for c in sizes.index])
    angles = 2*np.pi*angles/angles[-1]
    ring = 1.3*sum(radius.values())/np.pi
    layout = {}
    for i, cluster in enumerate(sizes.index):
        members = [g for g in graph if assignments[g] == cluster]
        local = nx.spring_layout(graph.subgraph(members), seed=seed, iterations=100)
        centre = ring*np.array([np.cos((angles[i] + angles[i + 1])/2), np.sin((angles[i] + angles[i + 1])/2)])
        layout.update({g: centre + .9*radius[cluster]*p for g, p in local.items()})
    return layout


def plot_composition(composition, clusters, stem, max_clusters=25):
    """Which significant pairs each cluster's genes come from."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    shown = composition.loc[clusters.leiden_cluster[:max_clusters]]
    sizes = clusters.set_index('leiden_cluster').n_genes
    fig, ax = plt.subplots(figsize=(1.1 + .55*shown.shape[1], 1.5 + .32*len(shown)), layout='constrained')
    image = ax.imshow(shown.to_numpy(), cmap=ListedColormap(SEQUENTIAL), vmin=0, vmax=1, aspect='auto')
    for (i, j), f in np.ndenumerate(shown.to_numpy()):
        if f >= .05:
            ax.text(j, i, f'{f:.0%}', ha='center', va='center', fontsize=6.5,
                    color='white' if f >= .6 else '#333333')
    ax.set_yticks(range(len(shown)), [f'C{c} (n={sizes[c]})' for c in shown.index], fontsize=8)
    ax.set_xticks(range(shown.shape[1]), shown.columns, rotation=50, ha='right', rotation_mode='anchor', fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=.9, aspect=12, format=lambda v, _: f'{v:.0%}')
    ax.set_title("Leiden clusters by contributing CNV × CNV pair\n"
                 "cell = share of the cluster's genes that are candidates of the pair (a gene can count for several)",
                 loc='left', fontsize=9)
    for suffix in ('png', 'pdf'):
        fig.savefig(f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--network', default='PCNET2.2_brain')
    parser.add_argument('--run', required=True, help='Completed CNV x CNV run directory')
    parser.add_argument('--config', default='configs/cnv_convergence.json')
    parser.add_argument('--resolution', type=float, default=1.0)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    from cnv_overlap.pipeline import AnalysisConfig
    from network_inputs import load_network_inputs
    from results_layout import results_dir
    spec, root, inputs, _ = load_network_inputs(Path(args.config), args.network, min_seeds=3)
    config = AnalysisConfig.from_dict(spec['analysis'])
    out = build_pair_network(Path(args.run), inputs, annotation_dir=root/'input/annotations',
                             output_root=results_dir(root/spec['results_root'], 'cnv_pair_network'),
                             fdr=config.fdr, resolution=args.resolution)
    print(out)
    return out


if __name__ == '__main__':
    main()
