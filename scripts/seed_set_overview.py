"""Export seed counts, propagated-set sizes, and UpSet plots for a selected run."""
from cnv_overlap.results import network_root as configured_network_root, resolve_path, finalize_manifest, select_analysis
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from cnv_overlap.statistics import Thresholds
from network_inputs import NETWORKS
from results_layout import results_dir

ROOT = Path(__file__).resolve().parents[1]


def save(fig, stem):
    """Write a figure as PNG and PDF, then close it."""
    for ext in ('png', 'pdf'):
        fig.savefig(stem.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def draw_sizes(ax, table, threshold, row_gap=False):
    """Plot seed and propagated-set sizes on a logarithmic gene-count axis."""
    y = pd.Series(range(len(table)), dtype=float)
    h = 0.38
    seed_y, prop_y = y - h/2, y + h/2
    ax.barh(seed_y, table.raw_seeds, height=h, color='#c9c8c3', label='Seed genes')
    ax.barh(seed_y, table.retained_seeds, height=h, color='#2a78d6', label='Seed genes in network')
    ax.barh(prop_y, table.propagated_genes, height=h, color='#eb6834',
            label=f'After propagation (NPS > {threshold})')
    for yi, raw, kept in zip(seed_y, table.raw_seeds, table.retained_seeds):
        ax.text(raw*1.08, yi, f'{kept}/{raw}' if kept != raw else f'{raw}', va='center', fontsize=8)
    for yi, n in zip(prop_y, table.propagated_genes):
        ax.text(n*1.08, yi, f'{n:,}', va='center', fontsize=8)
    ax.set_xscale('log')
    ax.set_xlim(1, table.propagated_genes.max()*3)
    ax.set_ylim(len(table) - 0.5, -0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set(yticks=list(y), yticklabels=table.entity, xlabel='Genes (log scale)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2 if row_gap else 3,
              fontsize=9)


def intersections(members):
    """Exclusive UpSet intersections: genes per exact combination of sets, largest first."""
    counts = members[members.any(axis=1)].value_counts()
    return counts.sort_values(ascending=False, kind='stable')


def draw_upset(fig, cell, members, sizes, colour, title, max_cols, raw_sizes=None):
    """UpSet plot in one gridspec cell: set sizes (left), intersection sizes (top), membership dots.

    Each column counts genes that are in exactly the marked sets and in no other set."""
    names = list(members.columns)
    counts = intersections(members)
    shown = counts.head(max_cols)
    grid = cell.subgridspec(2, 2, width_ratios=[1, 5], height_ratios=[1.3, 1], wspace=0.32, hspace=0.04)
    top = fig.add_subplot(grid[0, 1])
    dots = fig.add_subplot(grid[1, 1], sharex=top)
    left = fig.add_subplot(grid[1, 0], sharey=dots)
    x = np.arange(len(shown))
    top.bar(x, shown.values, width=0.65, color=colour)
    for xi, n in zip(x, shown.values):
        top.text(xi, n, f'{n:,}', ha='center', va='bottom', fontsize=7, rotation=90 if n >= 1000 else 0)
    top.set_ylabel('Genes in exactly\nthese sets')
    top.set_ylim(0, shown.max()*1.25)
    top.spines[['top', 'right']].set_visible(False)
    top.tick_params(axis='x', bottom=False, labelbottom=False)
    hidden = len(counts) - len(shown)
    top.set_title(title + (f'  ({len(shown)} largest of {len(counts)} combinations)' if hidden else ''),
                  loc='left', fontsize=11)
    rows = np.arange(len(names))
    for r in rows[::2]:
        dots.axhspan(r - 0.5, r + 0.5, color='#f2f2f0', zorder=0)
    for xi, combo in zip(x, shown.index):
        on = [r for r, flag in zip(rows, combo) if flag]
        dots.scatter([xi]*len(rows), rows, s=28, color='#dddddd', zorder=1)
        dots.plot([xi, xi], [min(on), max(on)], color='#333333', lw=1.6, zorder=2)
        dots.scatter([xi]*len(on), on, s=34, color='#333333', zorder=3)
    dots.set_xlim(-0.6, len(shown) - 0.4)
    dots.set_ylim(len(names) - 0.5, -0.5)
    dots.set_yticks(rows, names, fontsize=9)
    dots.tick_params(length=0)
    dots.spines[:].set_visible(False)
    dots.set_xticks([])
    if raw_sizes is not None:
        left.barh(rows, raw_sizes, height=0.6, color='#c9c8c3', label='Seed genes')
    left.barh(rows, sizes, height=0.6, color=colour, label='In network' if raw_sizes is not None else None)
    for r, n in zip(rows, sizes if raw_sizes is None else raw_sizes):
        left.text(n, r, f'{n:,} ', ha='right', va='center', fontsize=7)
    left.set_xlim(max(sizes if raw_sizes is None else raw_sizes)*1.45, 0)
    left.set_xlabel('Genes in set')
    left.spines[['top', 'left']].set_visible(False)
    left.tick_params(axis='y', left=False, labelleft=False)
    if raw_sizes is not None:
        left.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0, 1.0))
    return top


def upset_figure(members, sizes, colour, title, stem, max_cols=30, raw_sizes=None):
    """Save one stand-alone UpSet plot; width follows the number of combinations shown."""
    n_cols = min(len(intersections(members)), max_cols)
    fig = plt.figure(figsize=(3.2 + 0.36*n_cols + 1.6, 0.32*members.shape[1] + 4.2))
    draw_upset(fig, fig.add_gridspec(1, 1)[0], members, sizes, colour, title, max_cols, raw_sizes)
    save(fig, stem)


def overview(network, source=None):
    """Export seed counts, propagation counts, and figures for a selected disorder run."""
    network_root = configured_network_root(network, ROOT/'results')
    run = select_analysis(network_root, 'cnv_disorder_overlap', source=source)
    from cnv_overlap.results import verify_run
    source_manifest = verify_run(run)
    threshold = source_manifest['identity'].get('config', {}).get('thresholds', {}).get('individual', Thresholds().individual)
    qc = pd.read_csv(resolve_path(run/'seed_qc.tsv'), sep='\t')
    nps = pd.read_csv(resolve_path(run/'nps.tsv'), sep='\t', index_col='entrez')
    qc['propagated_genes'] = [int((nps[e] > threshold).sum()) for e in qc.entity]
    table = qc[['entity', 'kind', 'raw_seeds', 'retained_seeds', 'propagated_genes', 'median_degree']]

    from cnv_overlap.provenance import file_hash, run_dir
    identity = {'network': network, 'source': str(run.resolve()), 'code': file_hash(__file__), 'source_manifest': file_hash(run/'manifest.json'), 'threshold': threshold}
    out = run_dir(results_dir(network_root, 'seed_sets'), identity, 'seed-sets')
    if (out/'manifest.json').exists():
        from cnv_overlap.results import verify_run
        verify_run(out)
        return out
    (out/'figures').mkdir(parents=True, exist_ok=True)
    table.to_csv(resolve_path(out/'seed_sets.tsv'), sep='\t', index=False)
    (out/'source.json').write_text(json.dumps(
        {'run': str(run.resolve()), 'nps_threshold': threshold,
         'propagated_genes': f'genes in the network with NPS > {threshold}'}, indent=2))

    names = list(table.entity)
    entities = json.loads((run/'manifest.json').read_text())['identity']['entities']
    seed_members = pd.DataFrame({n: nps.index.isin(entities[n]['seeds']) for n in names}, index=nps.index)
    prop_members = nps[names] > threshold
    for label, members in (('seeds', seed_members), ('propagated', prop_members)):
        combos = intersections(members)
        pd.DataFrame({'sets': [' & '.join(n for n, f in zip(names, c) if f) for c in combos.index],
                      'n_sets': [sum(c) for c in combos.index], 'genes': combos.values}
                     ).to_csv(resolve_path(out/f'upset_{label}.tsv'), sep='\t', index=False)

    fig, ax = plt.subplots(figsize=(8, 0.6*len(table)+1.4))
    draw_sizes(ax, table, threshold)
    ax.set_title(network)
    save(fig, out/'figures'/'seed_set_sizes')
    upset_figure(seed_members, table.retained_seeds.values, '#2a78d6',
                 f'{network}: seed genes in the network', out/'figures'/'upset_seeds',
                 raw_sizes=table.raw_seeds.values)
    upset_figure(prop_members, table.propagated_genes.values, '#eb6834',
                 f'{network}: genes after propagation (NPS > {threshold})', out/'figures'/'upset_propagated')
    (resolve_path(out/'figures'/'seed_set_overview.png')).unlink(missing_ok=True)
    (resolve_path(out/'figures'/'seed_set_overview.pdf')).unlink(missing_ok=True)
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity})
    print(f'wrote {out}')
    return out


def main():
    """Run the overview for one network, or for every network that has a complete disorder run."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--network', default='PCNET2.2_brain', choices=NETWORKS + ('all',))
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.network != 'all':
        overview(args.network, args.source)
        return
    for network in NETWORKS:
        try:
            overview(network)
        except FileNotFoundError as ex:
            print(f'skip {network}: {ex}')


if __name__ == '__main__':
    main()
