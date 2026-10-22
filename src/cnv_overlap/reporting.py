"""Export completed-run figures and descriptive pair-network annotations."""
from cnv_overlap.results import resolve_path, finalize_manifest
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .enrichment import annotate_run
from .interpretation import candidate_network
from .provenance import digest, file_hash, write_json
from .visualization import display_name, pair_heatmap, enrichment_dotplot


def write_graphml(graph, path):
    """Use XML directly: NetworkX 2.8's writer references removed NumPy aliases."""
    namespace = 'http://graphml.graphdrawing.org/xmlns'
    ET.register_namespace('', namespace)
    def element(tag, **attrs):
        """Create an element in the GraphML XML namespace."""
        return ET.Element(f'{{{namespace}}}{tag}', attrs)
    root = element('graphml')
    types = {}
    for _, attrs in graph.nodes(data=True):
        for key, value in attrs.items():
            types[key] = 'boolean' if isinstance(value, bool) else 'double' if isinstance(value, float) else 'long' if isinstance(value, int) else 'string'
    for key, kind in sorted(types.items()):
        root.append(element('key', **{'id': key, 'for': 'node', 'attr.name': key, 'attr.type': kind}))
    container = element('graph', id='candidates', edgedefault='undirected')
    root.append(container)
    for gene, attrs in graph.nodes(data=True):
        node = element('node', id=str(gene))
        container.append(node)
        for key, value in attrs.items():
            data = element('data', key=key)
            data.text = str(value).lower() if isinstance(value, bool) else str(value)
            node.append(data)
    for i, (u, v) in enumerate(graph.edges()):
        container.append(element('edge', id=f'e{i}', source=str(u), target=str(v)))
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def export_report(run_dir, inputs, *, output_dir, gmt_paths=(), fdr=.05):
    """Verify a completed run and export figures, networks, and optional enrichment.

    Write to a separate report directory to preserve the source run manifest.
    """
    run_dir, out = Path(run_dir), Path(output_dir)
    if out.resolve() == run_dir.resolve() or run_dir.resolve() in out.resolve().parents:
        raise ValueError('Keep reports outside the immutable analysis run directory')
    manifest = json.loads((run_dir/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Reports require a completed analysis run')
    for name, checksum in manifest['artifacts'].items():
        if file_hash(run_dir/name) != checksum:
            raise ValueError(f'Run artifact changed: {name}')
    A, nodes, symbols, *_ = inputs
    from .provenance import graph_hash
    if graph_hash(A, nodes) != manifest['identity']['graph']:
        raise ValueError('Report network differs from the analysis network')
    symbols = {int(g): symbol for g, symbol in manifest['identity']['symbols'].items()}
    report_identity = {'source': str(run_dir.resolve()), 'run_manifest_sha256': file_hash(run_dir/'manifest.json'),
        'fdr': fdr, 'gmt': {str(p): file_hash(p) for p in gmt_paths}}
    if (out/'report_manifest.json').exists():
        previous = json.loads((out/'report_manifest.json').read_text())
        if any(previous.get(k) != v for k, v in report_identity.items()):
            raise ValueError('Report inputs/settings changed; choose a new output directory')
    from .results import reuse_completed
    if reuse_completed(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    tests = pd.read_csv(resolve_path(run_dir/'pair_tests.tsv'), sep='\t')
    null = manifest['primary_null']
    primary = tests[(tests['null'] == null) & (tests.statistic == 'size')]
    def save(fig, name):
        """Save a figure as PNG and PDF, then close it."""
        for suffix in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{suffix}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    for family in sorted(primary.family.unique()):
        # COLOC score, and the same colours labelled with colocalised network gene counts.
        save(pair_heatmap(run_dir, null, family, fdr), family)
        save(pair_heatmap(run_dir, null, family, fdr, metric='ratio_genes'), f'{family}_genes')
    primary.to_csv(resolve_path(out/'primary_pairs.tsv'), sep='\t', index=False)
    candidates = pd.read_csv(resolve_path(run_dir/'candidate_genes.tsv'), sep='\t')
    network_rows = []
    for row in primary.itertuples():
        if not row.q < fdr:
            continue
        query = candidates[(candidates.a == row.a) & (candidates.b == row.b)]
        graph, membership = candidate_network(A, nodes, query)
        key = digest([row.a, row.b])[:12]
        directory = out/'networks'/key
        directory.mkdir(parents=True, exist_ok=True)
        attributes = query.set_index('entrez').to_dict('index')
        for gene in graph:
            graph.nodes[gene].update({k: v for k, v in attributes[gene].items()
                                      if isinstance(v, (str, int, float, bool))})
        community_map = membership.set_index('entrez').community.to_dict()
        nx.set_node_attributes(graph, community_map, 'community')
        write_graphml(graph, resolve_path(directory/'network.graphml'))
        membership['symbol'] = membership.entrez.map(symbols)
        membership.to_csv(resolve_path(directory/'communities.tsv'), sep='\t', index=False)
        query.to_csv(resolve_path(directory/'genes.tsv'), sep='\t', index=False)
        fig, ax = plt.subplots(figsize=(9, 7))
        if not len(graph):
            ax.text(.5, .5, 'No candidate genes', ha='center')
        elif len(graph) <= 150:
            pos = nx.spring_layout(graph, seed=42)
            groups = membership.set_index('entrez').community.to_dict()
            # NetworkX 2.8 calls removed np.alltrue for a single-edge graph.
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection([(pos[a], pos[b]) for a, b in graph.edges()],
                                             colors='black', alpha=.12, linewidths=.5))
            nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=60,
                                   node_color=[groups[g] for g in graph], cmap='tab20')
            # Label only the ten strongest joint-score genes; export retains all genes.
            labelled = query.assign(product=query.nps_a*query.nps_b).nlargest(10, 'product')
            nx.draw_networkx_labels(graph, pos, {int(r.entrez): r.symbol for r in labelled.itertuples()},
                                    ax=ax, font_size=7)
        else:
            groups = membership.set_index('entrez').community.to_dict()
            summary = nx.Graph()
            for g, group in groups.items():
                summary.add_node(group, size=sum(v == group for v in groups.values()))
            for u, v in graph.edges():
                x, y = groups[u], groups[v]
                if x != y:
                    summary.add_edge(x, y, weight=summary.get_edge_data(x, y, {}).get('weight', 0)+1)
            pos = nx.spring_layout(summary, seed=42, weight='weight')
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection([(pos[a], pos[b]) for a, b in summary.edges()],
                colors='black', linewidths=[.3+np.log1p(d['weight'])/3 for _, _, d in summary.edges(data=True)]))
            nx.draw_networkx_nodes(summary, pos, ax=ax,
                node_size=[80+20*np.sqrt(summary.nodes[g]['size']) for g in summary])
            nx.draw_networkx_labels(summary, pos, ax=ax,
                labels={g: f"C{g}\n{summary.nodes[g]['size']} genes" for g in summary}, font_size=7)
        ax.set_title(f'{display_name(row.a)} × {display_name(row.b)}: shared network genes' +
                     ('\nnodes = gene communities' if len(graph) > 150 else ''))
        ax.axis('off')
        for suffix in ('png', 'pdf'):
            fig.savefig(directory/f'network.{suffix}', dpi=200, bbox_inches='tight')
        plt.close(fig)
        network_rows.append({'a': row.a, 'b': row.b, 'q': row.q, 'nodes': len(graph),
                             'edges': graph.number_of_edges(), 'directory': str(directory.relative_to(out))})
    pd.DataFrame(network_rows, columns=['a', 'b', 'q', 'nodes', 'edges', 'directory']).to_csv(
        resolve_path(out/'network_index.tsv'), sep='\t', index=False)
    if gmt_paths:
        annotation = annotate_run(run_dir, gmt_paths, [symbols[int(g)] for g in nodes],
                                  output_dir=out/'enrichment')
        # Correction includes all tested pair-terms; display only primary-discovery pairs.
        selected = primary.loc[primary.q < fdr, ['a', 'b']]
        displayed = annotation.merge(selected, on=['a', 'b'])
        save(enrichment_dotplot(displayed, fdr=fdr), 'enrichment_summary')
    write_json(out/'report_manifest.json', {**report_identity, 'null': null,
        'enrichment': 'descriptive after network selection' if gmt_paths else 'not run: no GMT supplied'})
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': report_identity})
    return out
