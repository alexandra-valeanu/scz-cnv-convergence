"""CNV-blind resolution and random-seed stability of frozen landscape communities."""
from cnv_overlap.results import analysis_from_source, resolve_artifact, finalize_manifest, resolve_path
import argparse
import itertools
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from cnv_overlap.provenance import file_hash, run_dir, software, write_json
from shared_network_leiden import leiden_membership


def adjusted_rand(a, b):
    """Adjusted Rand index using the contingency-table definition."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError('Aligned 1D labels required')
    if len(a) < 2:
        return 1.
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max()+1, bi.max()+1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    choose = lambda x: (x*(x-1)/2).sum()
    cells = choose(table)
    rows, cols = choose(table.sum(axis=1)), choose(table.sum(axis=0))
    expected = rows*cols/(len(a)*(len(a)-1)/2)
    denominator = (rows+cols)/2-expected
    return float((cells-expected)/denominator) if denominator else 1.


def labels_for(nodes, assignment):
    # Unassigned singleton nodes must not be collapsed into a fictitious community.
    return np.array([assignment.get(n, -(i+1)) for i, n in enumerate(nodes)])


def run(source, resolutions=(.5, .75, 1., 1.25, 1.5, 2., 3.), seeds=tuple(range(42, 52))):
    source = Path(source).resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    files = ['01_schema_gwas_landscape/gene_modules.tsv', '01_schema_gwas_landscape/edges.tsv']
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete source')
    for f in files:
        if file_hash(resolve_artifact(source, f)) != manifest['artifacts'][manifest.get('artifact_paths', {}).get(f, f)]:
            raise ValueError(f'Changed source artifact: {f}')
    identity = {'source': str(source), 'source_manifest': file_hash(source/'manifest.json'),
                'resolutions': resolutions, 'seeds': seeds, 'software': software(),
                'code': file_hash(__file__), 'leiden_code': file_hash(Path(__file__).with_name('shared_network_leiden.py'))}
    out = run_dir(analysis_from_source(source, 'landscape_resolution'), identity, 'resolution')
    from cnv_overlap.results import reuse_completed
    if reuse_completed(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    finalize_manifest(out/'manifest.json', {'status': 'running', 'identity': identity})
    genes = pd.read_csv(resolve_artifact(source, files[0]), sep='\t')
    edges = pd.read_csv(resolve_artifact(source, files[1]), sep='\t')
    graph = nx.Graph()
    nodes = genes.entrez.astype(str).tolist()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges.astype(str).itertuples(index=False, name=None))
    connected = [n for n in nodes if graph.degree(n) > 0]
    partitions, metrics, memberships, overlaps = {}, [], [], []
    for resolution in resolutions:
        for seed in seeds:
            assignments = leiden_membership(graph, seed=seed, resolution=resolution)
            sizes = pd.Series(list(assignments.values())).value_counts()
            n_singleton = len(nodes)-len(assignments)
            metrics.append({'resolution': resolution, 'seed': seed, 'communities_ge2': len(sizes),
                            'eligible_modules_ge5': int((sizes >= 5).sum()),
                            'largest_module': int(sizes.max()), 'median_module': float(sizes.median()),
                            'genes_in_modules_ge5': int(sizes[sizes >= 5].sum()),
                            'genes_in_small_or_unassigned': int(sizes[sizes < 5].sum()+n_singleton)})
            partitions[resolution, seed] = labels_for(connected, assignments)
            memberships.extend({'resolution': resolution, 'seed': seed, 'entrez': int(n),
                                'module': assignments.get(n), 'symbol': symbol}
                               for n, symbol in zip(nodes, genes.symbol))
            if seed == seeds[0]:
                for cluster in sorted(sizes.index):
                    group = {n for n, c in assignments.items() if c == cluster}
                    for baseline, frame in genes.dropna(subset=['leiden_cluster']).groupby('leiden_cluster'):
                        ref = set(frame.entrez.astype(str))
                        intersection = len(group & ref)
                        if intersection:
                            overlaps.append({'resolution': resolution, 'module': cluster,
                                'genes': len(group), 'baseline_module': int(baseline),
                                'intersection': intersection, 'jaccard': intersection/len(group | ref)})
        print(f'{source.parent.parent.name}: resolution {resolution} complete', flush=True)
    pairwise = pd.DataFrame([{'resolution': r, 'seed_a': a, 'seed_b': b,
                              'adjusted_rand': adjusted_rand(partitions[r, a], partitions[r, b])}
                             for r in resolutions for a, b in itertools.combinations(seeds, 2)])
    metrics = pd.DataFrame(metrics)
    summary = metrics.groupby('resolution').agg(modules_min=('eligible_modules_ge5', 'min'),
        modules_median=('eligible_modules_ge5', 'median'), modules_max=('eligible_modules_ge5', 'max'),
        largest_median=('largest_module', 'median'), module_size_median=('median_module', 'median'),
        small_unassigned_median=('genes_in_small_or_unassigned', 'median'))
    summary = summary.join(pairwise.groupby('resolution').adjusted_rand.agg(ari_mean='mean', ari_min='min'))
    summary.to_csv(resolve_path(out/'resolution_summary.tsv'), sep='\t')
    metrics.to_csv(resolve_path(out/'partition_metrics.tsv'), sep='\t', index=False)
    pairwise.to_csv(resolve_path(out/'seed_stability.tsv'), sep='\t', index=False)
    pd.DataFrame(memberships).to_csv(resolve_path(out/'memberships.tsv'), sep='\t', index=False)
    pd.DataFrame(overlaps).to_csv(resolve_path(out/'baseline_module_overlap.tsv'), sep='\t', index=False)
    pd.DataFrame([{'resolution_a': a, 'resolution_b': b,
                   'adjusted_rand': adjusted_rand(partitions[a, seeds[0]], partitions[b, seeds[0]])}
                  for a, b in itertools.combinations(resolutions, 2)]).to_csv(resolve_path(out/'resolution_similarity.tsv'), sep='\t', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(summary.index, summary.modules_median, 'o-')
    axes[0].fill_between(summary.index, summary.modules_min, summary.modules_max, alpha=.2)
    axes[0].set_ylabel('Eligible modules (>=5 genes)')
    axes[1].plot(summary.index, summary.largest_median, 'o-', label='Largest')
    axes[1].plot(summary.index, summary.module_size_median, 'o-', label='Median')
    axes[1].set_ylabel('Module genes'); axes[1].legend()
    axes[2].plot(summary.index, summary.ari_mean, 'o-', label='Mean')
    axes[2].plot(summary.index, summary.ari_min, 'o-', label='Minimum')
    axes[2].set_ylabel('Between-seed adjusted Rand index'); axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Leiden resolution')
        ax.axvline(1, color='grey', linestyle=':', linewidth=1)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'resolution_stability.{ext}', dpi=180)
    plt.close(fig)
    (out/'README.md').write_text('# Leiden resolution sensitivity\n\n'
        'Seven resolutions, ten seeds per resolution; no CNV results used. ARI compares partitions on '
        'non-isolated landscape genes, assigning any unassigned singleton its own label. '
        'Singletons are included in coverage/fragmentation metrics. Stability across optimizer seeds '
        'is not stability to perturbation of edges, gene sets or networks. Resolution comparisons use seed 42. '
        'Do not choose a resolution based on CNV significance. Larger resolution generally gives smaller '
        'communities; module count alone does not establish an optimum.\n')
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'artifacts': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.name != 'manifest.json'}})
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    args = parser.parse_args()
    print(run(args.source_run))
