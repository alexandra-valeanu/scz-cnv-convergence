"""Descriptive ORA of networks and modules, corrected within annotation groups."""
from cnv_overlap.results import finalize_manifest, resolve_path
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from cnv_overlap.enrichment import load_gmt, ora
from cnv_overlap.provenance import file_hash, run_dir, write_json
from cnv_overlap.statistics import bh

# library name -> (annotation group, GMT file)
CONTEXT_LIBRARIES = {
    'GO Biological Process': ('function', 'GO_Biological_Process_2026.gmt'),
    'GO Cellular Component': ('function', 'GO_Cellular_Component_2026.gmt'),
    'GO Molecular Function': ('function', 'GO_Molecular_Function_2026.gmt'),
    'Reactome pathways': ('function', 'ReactomePathways.gmt'),
    'SynGO 2024': ('synapse', 'SynGO_2024.gmt'),
    'MitoCarta 3.0': ('mitochondria', 'MitoCarta3.0.gmt'),
    'HPA broad brain cell type': ('broad brain cell type', 'HPA_brain_broad_cell_types.gmt'),
    'HPA brain region': ('brain region', 'HPA_brain_region_elevated.gmt'),
    'HPA brain cell type (snRNA)': ('cell type', 'HPA_brain_cell_type_elevated.gmt'),
}
HPA_GENES = 'HPA_gene_specificity.tsv'
HPA_GENE_COLUMNS = {
    'RNA tissue specificity': 'hpa_tissue_category', 'RNA tissue specific nTPM': 'hpa_tissues',
    'RNA brain regional specificity': 'hpa_brain_region_category',
    'RNA brain regional specific nTPM': 'hpa_brain_regions',
    'RNA single nuclei brain specificity': 'hpa_brain_cell_type_category',
    'RNA single nuclei brain specific nCPM': 'hpa_brain_cell_types',
    'RNA single cell type specificity': 'hpa_cell_type_category',
    'RNA single cell type specific nCPM': 'hpa_cell_types',
}
# Heatmap panels: library, and whether to show every context or only those nominally hit somewhere.
HEATMAP_PANELS = [('HPA broad brain cell type', False), ('HPA brain cell type (snRNA)', False)]
SEQUENTIAL = ['#ffffff', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
CORE = 'core'
TARGET_NAMES = {'schema': 'SCHEMA', 'gwaspri': 'GWAS'}


def annotate_cluster_context(cluster_dir, *, annotation_dir, output_root, fdr=.05, min_cluster_genes=3,
                             min_term_size=5):
    cluster_dir, annotation_dir = Path(cluster_dir), Path(annotation_dir)
    manifest = json.loads((cluster_dir/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Cluster run is incomplete')
    for name, checksum in manifest['artifacts'].items():
        if file_hash(cluster_dir/name) != checksum:
            raise ValueError(f'Cluster artifact changed: {name}')
    paths = _context_paths(annotation_dir)
    genes = pd.read_csv(resolve_path(cluster_dir/'genes.tsv'), sep='\t')
    column, method = 'leiden_cluster', 'leiden'
    if column not in genes:
        raise ValueError('Cluster context requires a Leiden run (genes.tsv with leiden_cluster)')
    identity = {'source': str(cluster_dir.resolve()), 'clusters_manifest': file_hash(cluster_dir/'manifest.json'), 'code': file_hash(__file__),
                'fdr': fdr, 'min_cluster_genes': min_cluster_genes, 'min_term_size': min_term_size,
                'correction': 'BH within annotation group across core and clusters',
                'libraries': {name: file_hash(p) for name, p in paths.items()}}
    out = run_dir(output_root, identity, f'{method}-context')
    if _reuse_complete(out):
        return out
    background = set(pd.read_csv(resolve_path(cluster_dir/'background_genes.tsv'), sep='\t').symbol.astype(str))
    genes['symbol'] = genes.symbol.astype(str)
    singletons = pd.read_csv(resolve_path(cluster_dir/'excluded_singletons.tsv'), sep='\t')
    singletons['symbol'] = singletons.symbol.astype(str)
    queries = {CORE: set(genes.symbol) | set(singletons.symbol)}
    for cluster, group in genes.groupby(column):
        if len(group) >= min_cluster_genes:
            queries[str(cluster)] = set(group.symbol)
    enrichment = context_ora(queries, background, paths, min_term_size)
    small = genes.groupby(column).filter(lambda g: str(g.name) not in queries)
    summary = context_summary(enrichment, queries, fdr,
                              untested={str(c): set(g.symbol) for c, g in small.groupby(column)})
    gene_context = gene_annotations(genes, singletons, column, paths, enrichment, fdr)
    out.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(resolve_path(out/'context_enrichment.tsv'), sep='\t', index=False)
    enrichment[enrichment.q_group < fdr].to_csv(resolve_path(out/'context_enrichment_significant.tsv'), sep='\t', index=False)
    summary.to_csv(resolve_path(out/'cluster_context.tsv'), sep='\t', index=False)
    gene_context.to_csv(resolve_path(out/'gene_context.tsv'), sep='\t', index=False)
    labels = {q: f'Shared core (n={len(m)})' if q == CORE else f'{method.upper()} C{q} (n={len(m)})'
              for q, m in queries.items()}
    plot_context_heatmap(enrichment, labels, fdr, out/'tissue_celltype_enrichment',
                         f'Brain cell-type enrichment: shared core and {method.upper()} clusters')
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'interpretation': 'Descriptive ORA of the shared core and its clusters; BH within annotation group; '
                          'HPA "elevated" = enriched, group enriched or enhanced; not adjusted for network selection',
        'artifacts': {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}})
    return out


def _context_paths(annotation_dir):
    paths = {name: annotation_dir/file for name, (_, file) in CONTEXT_LIBRARIES.items()}
    paths[HPA_GENES] = annotation_dir/HPA_GENES
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('Context annotation snapshots missing (run scripts/fetch_cluster_annotations.py): '
                                + ', '.join(missing))
    return paths


def _reuse_complete(out):
    """True when out holds a complete, unmodified run."""
    if not (out/'manifest.json').exists():
        return False
    previous = json.loads((out/'manifest.json').read_text())
    if previous['status'] != 'complete':
        return False
    for name, checksum in previous['artifacts'].items():
        if file_hash(out/name) != checksum:
            raise ValueError(f'Context artifact changed: {name}')
    return True


def context_ora(queries, background, paths, min_term_size):
    """ORA of every query against every library; BH within annotation group across all queries."""
    frames = []
    for library, (group_name, _) in CONTEXT_LIBRARIES.items():
        terms = load_gmt(paths[library])
        for query, members in queries.items():
            result = ora(members, background, terms, min_size=min_term_size)
            result.insert(0, 'library', library)
            result.insert(0, 'group', group_name)
            result.insert(0, 'cluster', query)
            frames.append(result.drop(columns='q'))
    enrichment = pd.concat(frames, ignore_index=True)
    enrichment['q_group'] = np.nan
    for _, index in enrichment.groupby('group').groups.items():
        enrichment.loc[index, 'q_group'] = bh(enrichment.loc[index, 'p'].to_numpy())
    enrichment['order'] = enrichment.cluster.map({q: i for i, q in enumerate(queries)})
    return enrichment.sort_values(['order', 'group', 'q_group', 'p', 'library', 'term']).drop(columns='order')


def annotate_network_context(network_dir, *, annotation_dir, output_root, fdr=.05, min_term_size=5):
    """Context ORA for each colocalised SCZ network instead of its Leiden clusters."""
    network_dir, annotation_dir = Path(network_dir), Path(annotation_dir)
    manifest = json.loads((network_dir/'manifest.json').read_text())
    for name, checksum in manifest['artifacts'].items():
        if file_hash(network_dir/name) != checksum:
            raise ValueError(f'Network artifact changed: {name}')
    if manifest['identity']['overlap'] != 'bin':
        raise ValueError('Network context supports overlap=bin only (the background is every network gene)')
    paths = _context_paths(annotation_dir)
    identity = {'source': str(network_dir.resolve()), 'networks_manifest': file_hash(network_dir/'manifest.json'), 'code': file_hash(__file__),
                'fdr': fdr, 'min_term_size': min_term_size,
                'correction': 'BH within annotation group across all colocalised networks',
                'libraries': {name: file_hash(p) for name, p in paths.items()}}
    out = run_dir(output_root, identity, 'colocalised-networks-context')
    if _reuse_complete(out):
        return out
    universe = pd.read_csv(resolve_path(network_dir/'gene_membership.tsv'), sep='\t')
    background = set(universe.symbol.dropna().astype(str))
    index = pd.read_csv(resolve_path(network_dir/'network_index.tsv'), sep='\t')
    # Shared core first, then each CNV with both targets, then every CNV x target pair.
    index['order'] = index.kind.map({'shared_core': 0, 'cnv_both': 1, 'pair': 2})
    index = index.sort_values(['order', 'cnv', 'target'], kind='stable')
    queries, labels, members = {}, {}, []
    for row in index.itertuples():
        genes = pd.read_csv(resolve_path(network_dir/row.directory/'genes.tsv'), sep='\t', usecols=['entrez', 'symbol'])
        genes['symbol'] = genes.symbol.astype(str)
        name = CORE if row.kind == 'shared_core' else f'{row.cnv} × {row.target}'
        queries[name] = set(genes.symbol)
        target = ' + '.join(TARGET_NAMES.get(t, t) for t in row.target.split('+'))
        labels[name] = (f'Shared core, ≥{manifest["identity"]["min_cnvs"]} CNVs' if row.kind == 'shared_core'
                        else f'{row.cnv} × {target}') + f' (n={len(genes)})'
        members.append(genes.assign(network=name))
    enrichment = context_ora(queries, background, paths, min_term_size)
    summary = context_summary(enrichment, queries, fdr)
    gene_context = network_gene_annotations(pd.concat(members, ignore_index=True), paths)
    out.mkdir(parents=True, exist_ok=True)
    enrichment.to_csv(resolve_path(out/'context_enrichment.tsv'), sep='\t', index=False)
    enrichment[enrichment.q_group < fdr].to_csv(resolve_path(out/'context_enrichment_significant.tsv'), sep='\t', index=False)
    summary.rename(columns={'cluster': 'network'}).to_csv(resolve_path(out/'network_context.tsv'), sep='\t', index=False)
    gene_context.to_csv(resolve_path(out/'gene_context.tsv'), sep='\t', index=False)
    plot_context_heatmap(enrichment, labels, fdr, out/'tissue_celltype_enrichment',
                         'Brain cell-type enrichment of SCZ colocalised networks')
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'interpretation': 'Descriptive ORA of each colocalised network against all network genes; BH within '
                          'annotation group; HPA "elevated" = enriched, group enriched or enhanced; '
                          'not adjusted for network selection',
        'artifacts': {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}})
    return out


def network_gene_annotations(members, paths):
    """One row per network gene: HPA specificity and the colocalised networks containing it."""
    table = members.groupby(['entrez', 'symbol'], sort=False).network.agg(
        n_networks='size', networks='; '.join).reset_index()
    table['shared_core'] = table.networks.str.split('; ').map(lambda n: CORE in n)
    hpa = pd.read_csv(paths[HPA_GENES], sep='\t', dtype=str)
    hpa = hpa.drop_duplicates('Gene').rename(columns={'Gene': 'symbol', **HPA_GENE_COLUMNS})
    table = table.merge(hpa.drop(columns='Ensembl'), on='symbol', how='left')
    table = table.fillna({c: 'not in HPA' for c in HPA_GENE_COLUMNS.values() if c.endswith('_category')})
    return table.sort_values(['n_networks', 'symbol'], ascending=[False, True])


def context_summary(enrichment, queries, fdr, untested=None):
    """One row per tested gene set: best significant term per annotation group."""
    groups = list(dict.fromkeys(g for g, _ in CONTEXT_LIBRARIES.values()))
    rows = []
    for query, members in queries.items():
        row = {'cluster': query, 'n_genes': len(members)}
        for group_name in groups:
            hits = enrichment[(enrichment.cluster == query) & (enrichment.group == group_name)
                              & (enrichment.q_group < fdr)]
            key = group_name.replace(' ', '_')
            row[f'{key}_significant'] = len(hits)
            row[f'top_{key}'] = ('; '.join(f'{h.term} [{h.library}; {h.overlap}/{h.query_size}; '
                                           f'FE {h.fold_enrichment:.1f}; q {h.q_group:.2g}]'
                                           for h in hits.head(3).itertuples()) or 'none at FDR')
        row['genes'] = ','.join(sorted(members))
        rows.append(row)
    for cluster, members in (untested or {}).items():
        rows.append({'cluster': cluster, 'n_genes': len(members), 'genes': ','.join(sorted(members)),
                     **{f'top_{g.replace(" ", "_")}': f'not tested (< cluster size threshold); see gene_context.tsv'
                        for g in groups}})
    return pd.DataFrame(rows)


def gene_annotations(genes, singletons, column, paths, enrichment, fdr):
    """Per-gene HPA specificity plus membership of significant function/synapse terms."""
    genes = pd.concat([genes.assign(clustering='clustered'),
                       singletons.assign(clustering='excluded singleton (' + singletons.reason + ')')],
                      ignore_index=True)
    hpa = pd.read_csv(paths[HPA_GENES], sep='\t', dtype=str)
    hpa = hpa.drop_duplicates('Gene').rename(columns={'Gene': 'symbol', **HPA_GENE_COLUMNS})
    table = genes[['symbol', column, 'clustering']].merge(hpa.drop(columns='Ensembl'), on='symbol', how='left')
    table = table.fillna({c: 'not in HPA' for c in HPA_GENE_COLUMNS.values() if c.endswith('_category')})
    significant = enrichment[(enrichment.q_group < fdr) & (enrichment.cluster == CORE)
                             & enrichment.group.isin(['function', 'synapse'])]
    for group_name in ('synapse', 'function'):
        terms = {}
        for hit in significant[significant.group == group_name].itertuples():
            for gene in hit.genes.split(','):
                terms.setdefault(gene, []).append(f'{hit.term} [{hit.library}]')
        table[f'core_{group_name}_terms'] = table.symbol.map(lambda g: '; '.join(terms.get(g, [])[:5]))
    return table.sort_values([column, 'symbol'], na_position='last')


def plot_context_heatmap(enrichment, labels, fdr, stem, title, max_columns=30):
    """labels maps each tested gene set, in row order, to its axis label."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    rows, labels = list(labels), list(labels.values())
    panels, empty = [], []
    for library, filtered in HEATMAP_PANELS:
        data = enrichment[enrichment.library == library]
        if filtered:
            data = data[data.term.isin(data.loc[(data.p < .05) & (data.overlap > 0), 'term'].unique())]
        if not (data.overlap > 0).any():
            empty.append(library)
            continue
        q = data.pivot(index='cluster', columns='term', values='q_group').reindex(rows)
        order = list(q.min().sort_values().index[:max_columns]) if filtered else sorted(q.columns)
        overlap = data.pivot(index='cluster', columns='term', values='overlap').reindex(rows)[order]
        panels.append((library, filtered, q[order].astype(float), overlap))
    if not panels:
        return
    bounds = [0, .5, 1, 1.3, 2, 3, 4, 6, 50]
    cmap, norm = ListedColormap(SEQUENTIAL), BoundaryNorm(bounds, len(SEQUENTIAL))
    cell = .3
    # Fixed positions in inches, so every panel starts at the same left edge (cells stay square).
    left, top, gap = 3.5, .7, 2.9
    panel_w = cell*max(len(p[2].columns) for p in panels)
    fig_w = left + panel_w + 1.6
    fig_h = top + sum(cell*len(rows) + gap for _ in panels)
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes, y = [], top
    for p in panels:
        h = cell*len(rows)
        axes.append(fig.add_axes([left/fig_w, 1 - (y + h)/fig_h, cell*len(p[2].columns)/fig_w, h/fig_h]))
        y += h + gap
    for ax, (library, filtered, q, overlap) in zip(axes, panels):
        score = -np.log10(q)
        image = ax.imshow(score.fillna(0).to_numpy(), cmap=cmap, norm=norm, aspect='equal')
        ax.set_yticks(range(len(rows)), labels, fontsize=8)
        ax.set_xticks(range(len(q.columns)), [textwrap.shorten(c, 38) for c in q.columns],
                      rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
        for (i, j), k in np.ndenumerate(overlap.to_numpy()):
            if k > 0:
                significant = q.iat[i, j] < fdr
                ax.text(j, i, f'{int(k)}{"*" if significant else ""}', ha='center', va='center', fontsize=6.5,
                        color='white' if score.iat[i, j] >= 2 else '#333333',
                        fontweight='bold' if significant else 'normal')
        ax.set_xticks(np.arange(-.5, len(q.columns)), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
        ax.grid(which='minor', color='#e6e6e6', linewidth=.6)
        ax.tick_params(which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(library + (f' (top {max_columns} contexts with nominal p < 0.05 in any row)' if filtered else ''),
                     loc='left', fontsize=10)
    bar_h = min(3.0, fig_h - top - 1)
    cax = fig.add_axes([(left + panel_w + .4)/fig_w, 1 - (top + bar_h)/fig_h, .18/fig_w, bar_h/fig_h])
    bar = fig.colorbar(image, cax=cax, ticks=bounds[:-1])
    bar.set_label('−log10 q (BH within annotation group)')
    note = f'cell text = overlapping genes; * and bold = q < {fdr}'
    if empty:
        note += '\nno tested gene is HPA-elevated in any context of: ' + ', '.join(empty)
    fig.text(0, 1 - .25/fig_h, title, ha='left', va='top', fontsize=11)
    fig.text(0, 0, note, ha='left', va='top', fontsize=8, color='#555555')
    for suffix in ('png', 'pdf'):
        fig.savefig(f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
