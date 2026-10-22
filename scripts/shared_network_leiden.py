"""Reproducible Leiden communities and descriptive GO/Reactome annotations.

Singletons (isolated genes and any size-1 community) are excluded from the clustering
and listed in excluded_singletons.tsv.
"""
from cnv_overlap.results import finalize_manifest, resolve_path
import json
import textwrap
from importlib.metadata import version
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx
import pandas as pd

from cnv_overlap.enrichment import load_gmt, ora
from cnv_overlap.provenance import file_hash, run_dir, write_json
from cnv_overlap.statistics import bh
from cnv_overlap.reporting import write_graphml
from shared_network_enrichment import LIBRARIES


def leiden_membership(graph, *, seed=42, resolution=1.0):
    """Leiden per connected component; genes left in singletons get no cluster."""
    if resolution <= 0:
        raise ValueError('Leiden resolution must be positive')
    groups = []
    for component in sorted(nx.connected_components(graph), key=lambda c: min(map(int, c))):
        nodes = sorted(component, key=int)
        subgraph = graph.subgraph(nodes)
        if not subgraph.number_of_edges():
            continue
        pos = {g: i for i, g in enumerate(nodes)}
        local = ig.Graph(n=len(nodes), edges=sorted((min(pos[a], pos[b]), max(pos[a], pos[b]))
                                                   for a, b in subgraph.edges()), directed=False)
        partition = leidenalg.find_partition(local, leidenalg.RBConfigurationVertexPartition,
            seed=seed, resolution_parameter=resolution, n_iterations=-1)
        groups.extend({nodes[i] for i in group} for group in partition if len(group) > 1)
    groups.sort(key=lambda group: (-len(group), min(map(int, group))))
    return {g: i for i, group in enumerate(groups, 1) for g in group}


def annotate_leiden(network_dir, *, annotation_dir, output_root, seed=42, resolution=1.0, fdr=.05):
    network_dir, annotation_dir = Path(network_dir), Path(annotation_dir)
    manifest = json.loads((network_dir/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Network export is incomplete')
    for name, checksum in manifest['artifacts'].items():
        if file_hash(network_dir/name) != checksum:
            raise ValueError(f'Network artifact changed: {name}')
    index = pd.read_csv(resolve_path(network_dir/'network_index.tsv'), sep='\t')
    core = index[index.kind == 'shared_core']
    if core.empty:
        raise ValueError('Leiden shared core requires both SCHEMA and GWAS targets')
    paths = {name: annotation_dir/file for name, file in LIBRARIES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('Cluster annotation snapshots missing: ' + ', '.join(missing))
    identity = {'source': str(network_dir.resolve()), 'network_manifest': file_hash(network_dir/'manifest.json'), 'code': file_hash(__file__),
                'seed': seed, 'resolution': resolution, 'iterations': -1, 'fdr': fdr,
                'partition': 'RBConfigurationVertexPartition per connected component',
                'singletons': 'isolated genes and size-1 communities excluded',
                'leidenalg': version('leidenalg'), 'igraph': version('igraph'),
                'libraries': {name: file_hash(p) for name, p in paths.items()}}
    out = run_dir(output_root, identity, 'leiden')
    if (out/'manifest.json').exists():
        previous = json.loads((out/'manifest.json').read_text())
        if previous['status'] == 'complete':
            for name, checksum in previous['artifacts'].items():
                if file_hash(out/name) != checksum:
                    raise ValueError(f'Leiden artifact changed: {name}')
            return out
    directory = network_dir/core.iloc[0].directory
    genes = pd.read_csv(resolve_path(directory/'genes.tsv'), sep='\t')
    edges = pd.read_csv(resolve_path(directory/'edges.tsv'), sep='\t')
    graph = nx.Graph()
    for record in genes.to_dict('records'):
        gene = str(record.pop('entrez'))
        graph.add_node(gene, **record)
    graph.add_edges_from((str(a), str(b)) for a, b in edges.itertuples(index=False, name=None))
    assignments = leiden_membership(graph, seed=seed, resolution=resolution)
    clustered = genes.entrez.astype(str).isin(assignments)
    singletons = genes[~clustered].assign(
        reason=lambda d: ['isolated' if graph.degree(str(g)) == 0 else 'size-1 community' for g in d.entrez])
    genes = genes[clustered].copy()
    genes['leiden_cluster'] = genes.entrez.astype(str).map(assignments).astype(int)
    graph = graph.subgraph(assignments).copy()
    universe = pd.read_csv(resolve_path(network_dir/'gene_membership.tsv'), sep='\t')
    eligible = pd.Series(True, index=universe.index)
    original = manifest['identity']
    if original['overlap'] == 'remove':
        cnvs = [n for n, e in original['entities'].items() if e['kind'] == 'cnv']
        counts = sum(~(universe[f'seed_{n}'] & (universe.seed_schema | universe.seed_gwaspri)) for n in cnvs)
        eligible = counts >= original['min_cnvs']
    background = set(universe.loc[eligible, 'symbol'].dropna().astype(str))
    frames = []
    for library, path in paths.items():
        terms = load_gmt(path)
        for cluster, group in genes.groupby('leiden_cluster'):
            result = ora(set(group.symbol.astype(str)), background, terms)
            result['library'], result['leiden_cluster'] = library, cluster
            frames.append(result)
    enrichment = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['term', 'description', 'p', 'q', 'library', 'leiden_cluster', 'genes'])
    enrichment['q_all_cluster_terms'] = bh(enrichment.p)
    enrichment = enrichment.sort_values(['q_all_cluster_terms', 'p', 'library', 'term'])
    rows = []
    for cluster, group in genes.groupby('leiden_cluster'):
        hits = enrichment[(enrichment.leiden_cluster == cluster) & (enrichment.q_all_cluster_terms < fdr)
                          & (enrichment.library == 'GO Biological Process')]
        label = 'No enriched biological process at FDR threshold'
        if not hits.empty:
            best = hits.iloc[0]
            label = best.term
        rows.append({'leiden_cluster': cluster, 'n_genes': len(group), 'annotation': label,
                     'significant_terms': len(hits), 'genes': ','.join(group.symbol.astype(str))})
    summary = pd.DataFrame(rows, columns=['leiden_cluster', 'n_genes', 'annotation', 'significant_terms', 'genes'])
    labels = dict(zip(summary.leiden_cluster, summary.annotation))
    genes['cluster_annotation'] = genes.leiden_cluster.map(labels)
    for gene, cluster in assignments.items():
        graph.nodes[gene]['leiden_cluster'] = int(cluster)
        graph.nodes[gene]['cluster_annotation'] = str(labels[cluster])
    out.mkdir(parents=True, exist_ok=True)
    genes.to_csv(resolve_path(out/'genes.tsv'), sep='\t', index=False)
    summary.to_csv(resolve_path(out/'clusters.tsv'), sep='\t', index=False)
    singletons.to_csv(resolve_path(out/'excluded_singletons.tsv'), sep='\t', index=False)
    enrichment.to_csv(resolve_path(out/'cluster_enrichment.tsv'), sep='\t', index=False)
    pd.DataFrame({'symbol': sorted(background)}).to_csv(resolve_path(out/'background_genes.tsv'), sep='\t', index=False)
    write_graphml(graph, resolve_path(out/'network.graphml'))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 10))
    if graph:
        from matplotlib.collections import LineCollection
        pos = nx.spring_layout(graph, seed=seed, k=.7, iterations=200)
        colors = {c: plt.get_cmap('tab20')((c-1) % 20) for c in labels}
        ax.add_collection(LineCollection([(pos[a], pos[b]) for a, b in graph.edges()],
                                         colors='#aaaaaa', linewidths=.3))
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=[colors[assignments[g]] for g in graph], node_size=110)
        nx.draw_networkx_labels(graph, pos, ax=ax,
            labels={g: graph.nodes[g].get('symbol', g) for g in graph}, font_size=7)
        from matplotlib.patches import Patch
        handles = [Patch(color=colors[r.leiden_cluster],
            label=textwrap.fill(f'C{r.leiden_cluster} ({r.n_genes} genes): {r.annotation}', 45))
            for r in summary.itertuples()]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    else:
        ax.text(.5, .5, 'No shared-core genes outside singletons', ha='center')
    ax.set_title(f'CNV × SCHEMA × GWAS: Leiden communities (resolution={resolution}); '
                 f'{len(singletons)} singleton genes excluded')
    ax.axis('off')
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'leiden_clusters.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'interpretation': 'Descriptive clusters and ORA; singletons excluded; BH across all clusters and libraries; '
                          'not adjusted for network selection',
        'artifacts': {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}})
    return out
