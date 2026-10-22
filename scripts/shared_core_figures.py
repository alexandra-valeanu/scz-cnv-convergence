"""Plot shared-core functions, gene-by-CNV membership, and the core-size null.

Function labels use the first matching GO/SynGO pattern in a fixed priority
order. The supporting term is exported with each label. Core-size plots use
the saved sensitivity draws."""
from cnv_overlap.results import finalize_manifest, resolve_path
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cnv_overlap.enrichment import load_gmt
from cnv_overlap.provenance import file_hash, run_dir, write_json

FUNCTION_LIBRARIES = ('GO_Biological_Process_2026.gmt', 'GO_Cellular_Component_2026.gmt',
                      'GO_Molecular_Function_2026.gmt', 'SynGO_2024.gmt')
# (label, term-name pattern), highest priority first.
FUNCTION_CATEGORIES = (
    ('Glutamate receptors', r'glutamate receptor|glutamate-gated|\bAMPA\b|\bNMDA\b|kainate'),
    ('GABA-A / glycine receptors', r'GABA-A|GABA-gated|gamma-aminobutyric acid receptor|'
                                   r'ligand-gated monoatomic anion channel|inhibitory postsynaptic potential'),
    ('Cholinergic receptors', r'acetylcholine receptor|acetylcholine-gated|nicotinic'),
    ('Presynaptic release', r'synaptic vesicle|neurotransmitter secretion|presynaptic active zone|exocytosis'),
    ('Voltage-gated & other ion channels', r'voltage-gated|ion channel|sodium channel|potassium channel|calcium channel'),
    ('Synapse organisation & adhesion', r'synapse organization|synapse assembly|synaptic membrane adhesion|'
                                        r'cell adhesion|postsynaptic density|postsynaptic specialization'),
    ('Neuron projection & cytoskeleton', r'axon|dendrit|neuron projection|microtubule|cytoskeleton'),
    ('Gene regulation & RNA', r'transcription|chromatin|RNA binding|splicing|mRNA|histone'),
)
OTHER = 'Other / unclassified'
# Reference categorical order (dataviz palette, light mode); grey for Other.
FUNCTION_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
OTHER_COLOR = '#b4b2a9'
TILE_COLORS = {'both': '#1c5cab', 'schema': '#86b6ef', 'gwaspri': '#b4b2a9', 'none': '#f3f2ee'}
TILE_LABELS = {'both': 'SCHEMA + GWAS', 'schema': 'SCHEMA only (S)', 'gwaspri': 'GWAS only (G)', 'none': 'neither'}


def function_categories(symbols, annotation_dir):
    """Symbol → first matching category and the term that placed it there."""
    terms = {}
    for name in FUNCTION_LIBRARIES:
        terms.update({t: genes for t, (_, genes) in load_gmt(Path(annotation_dir)/name).items()})
    rows = []
    for symbol in symbols:
        label, evidence = OTHER, ''
        for category, pattern in FUNCTION_CATEGORIES:
            hits = sorted(t for t, genes in terms.items() if symbol in genes and re.search(pattern, t, re.I))
            if hits:
                label, evidence = category, hits[0]
                break
        rows.append({'symbol': symbol, 'function': label, 'evidence_term': evidence})
    return pd.DataFrame(rows)


def _colors():
    return {c: col for (c, _), col in zip(FUNCTION_CATEGORIES, FUNCTION_COLORS)} | {OTHER: OTHER_COLOR}


def _node_size(n):
    return 45*np.asarray(n, dtype=float)**2


def plot_function_network(genes, edges, out, *, min_cnvs):
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    graph = nx.Graph()
    graph.add_nodes_from(genes.entrez)
    graph.add_edges_from(edges[['source', 'target']].itertuples(index=False))
    info = genes.set_index('entrez')
    colors = _colors()
    isolates = sorted(nx.isolates(graph), key=lambda g: info.at[g, 'symbol'])
    connected = graph.subgraph([g for g in graph if g not in set(isolates)])
    pos = nx.spring_layout(connected, seed=42, k=.6, iterations=100) if connected else {}
    pos.update({g: (x, -1.4) for g, x in zip(isolates, np.linspace(-1, 1, len(isolates)))})
    big = len(graph) > 60
    fig, ax = plt.subplots(figsize=(18, 14) if big else (13, 10))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#c3c2b7', width=.6)
    order = sorted(graph, key=lambda g: info.at[g, 'n_cnvs_both'])  # large nodes on top
    ax.scatter([pos[g][0] for g in order], [pos[g][1] for g in order],
               s=_node_size([info.at[g, 'n_cnvs_both'] for g in order]),
               c=[colors[info.at[g, 'function']] for g in order], edgecolors='white', linewidths=1.2, zorder=3)
    for g in graph:
        n = info.at[g, 'n_cnvs_both']
        ax.annotate(info.at[g, 'symbol'], pos[g], xytext=(0, 4 + 2.2*n), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6.5 if big else 8, color='#1a1a19', zorder=4,
                    fontweight='bold' if n > min_cnvs else 'normal')
    present = genes.function.value_counts()
    handles = [Line2D([], [], marker='o', ls='', ms=9, mfc=colors[c], mec='white', label=f'{c} ({present[c]})')
               for c in list(colors) if c in present]
    legend1 = ax.legend(handles=handles, title='Function (GO/SynGO term match)', loc='upper left',
                        bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9, title_fontsize=10)
    ax.add_artist(legend1)
    sizes = sorted(genes.n_cnvs_both.unique())
    size_handles = [plt.scatter([], [], s=_node_size(n), c='#b4b2a9', edgecolors='white', label=f'{n} CNVs')
                    for n in sizes]
    ax.legend(handles=size_handles, title='CNVs with SCHEMA + GWAS', loc='upper left',
              bbox_to_anchor=(1.0, .45), frameon=False, labelspacing=1.6, borderpad=1, fontsize=9, title_fontsize=10)
    ax.set_title(f'Shared core, ≥{min_cnvs} CNVs × SCHEMA + GWAS: {len(graph)} genes, '
                 f'{graph.number_of_edges()} edges\nColour: function category; size: number of CNVs', fontsize=12)
    ax.axis('off')
    fig.subplots_adjust(left=.02, right=.74, top=.93, bottom=.02)
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'function_network.{suffix}', dpi=200, bbox_inches='tight',
                    bbox_extra_artists=[legend1, ax.get_legend()])
    plt.close(fig)


def gene_cnv_states(genes, cnv_names):
    """Long table: one tile state per gene × CNV, plus whether the gene is that CNV's seed."""
    def members(value):
        return set(str(value).split(',')) if isinstance(value, str) and value else set()
    rows = []
    for record in genes.to_dict('records'):
        schema, gwas = members(record.get('cnvs_schema')), members(record.get('cnvs_gwaspri'))
        for cnv in cnv_names:
            state = ('both' if cnv in schema and cnv in gwas else 'schema' if cnv in schema
                     else 'gwaspri' if cnv in gwas else 'none')
            rows.append({'symbol': record['symbol'], 'cnv': cnv, 'state': state,
                         'cnv_seed': bool(record.get(f'seed_{cnv}', False))})
    return pd.DataFrame(rows)


def plot_gene_cnv_matrix(genes, cnv_names, out, *, min_cnvs):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    colors = _colors()
    category_rank = {c: i for i, c in enumerate(colors)}
    genes = genes.assign(_rank=genes.function.map(category_rank)).sort_values(
        ['_rank', 'n_cnvs_both', 'symbol'], ascending=[True, False, True]).reset_index(drop=True)
    states = gene_cnv_states(genes, cnv_names)
    counts = states[states.state == 'both'].cnv.value_counts()
    cnvs = sorted(cnv_names, key=lambda c: (-counts.get(c, 0), c))
    codes = list(TILE_COLORS)
    grid = (states.pivot(index='symbol', columns='cnv', values='state')
            .loc[genes.symbol, cnvs].replace({s: i for i, s in enumerate(codes)}).to_numpy(dtype=float))
    seeds = states.pivot(index='symbol', columns='cnv', values='cnv_seed').loc[genes.symbol, cnvs].to_numpy(dtype=bool)
    n = len(genes)
    height = max(4.0, .17*n + 2.2)
    fig, (strip, ax, bar) = plt.subplots(1, 3, figsize=(9.5, height), sharey=True,
                                         gridspec_kw={'width_ratios': [.35, len(cnvs), 2.2], 'wspace': .04})
    strip.imshow(np.array([[category_rank[c]] for c in genes.function]), aspect='auto',
                 cmap=ListedColormap(list(colors.values())), vmin=-.5, vmax=len(colors)-.5, interpolation='nearest')
    strip.set_xticks([])
    strip.set_yticks(range(n), genes.symbol, fontsize=7 if n > 60 else 8.5)
    for side in ('top', 'right', 'bottom', 'left'):
        strip.spines[side].set_visible(False)
    ax.imshow(grid, aspect='auto', cmap=ListedColormap(list(TILE_COLORS.values())), vmin=-.5, vmax=len(codes)-.5,
              interpolation='nearest')
    ax.set_xticks(np.arange(len(cnvs)+1)-.5, minor=True)
    ax.set_yticks(np.arange(n+1)-.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(len(cnvs)), [f'{c}\n({counts.get(c, 0)})' for c in cnvs], rotation=90, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis='y', length=0)
    for (i, j), value in np.ndenumerate(grid):
        if codes[int(value)] in ('schema', 'gwaspri'):
            ax.text(j, i, 'S' if codes[int(value)] == 'schema' else 'G', ha='center', va='center',
                    fontsize=6.5 if n > 60 else 8, color='#1a1a19')
    r, c = np.nonzero(seeds)
    ax.scatter(c, r, marker='o', s=18, facecolors='white', edgecolors='#1a1a19', linewidths=.8, zorder=3)
    boundaries = np.flatnonzero(genes.function.to_numpy()[1:] != genes.function.to_numpy()[:-1]) + .5
    for y in boundaries:
        for axis in (strip, ax, bar):
            axis.axhline(y, color='#1a1a19', linewidth=.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar.barh(range(n), genes.n_cnvs_both, color='#1c5cab', height=.7)
    bar.set_xlim(0, len(cnvs))
    bar.xaxis.tick_top()
    bar.set_xlabel('CNVs with\nSCHEMA + GWAS', fontsize=8.5)
    bar.xaxis.set_label_position('top')
    bar.tick_params(axis='y', length=0)
    bar.axvline(min_cnvs, color='#1a1a19', linestyle=':', linewidth=.8)
    for side in ('right', 'bottom'):
        bar.spines[side].set_visible(False)
    handles = ([Patch(color=TILE_COLORS[s], label=TILE_LABELS[s]) for s in codes]
               + [Line2D([], [], marker='o', ls='', mfc='white', mec='#1a1a19', label='gene inside the CNV (seed)')]
               + [Patch(color=colors[c], label=c) for c in colors if c in set(genes.function)])
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, .06 - .35/height))
    fig.suptitle(f'Gene × CNV colocalisation, shared core ≥{min_cnvs} CNVs ({n} genes)\n'
                 'Rows grouped by function (left strip); column label: core genes with SCHEMA + GWAS',
                 fontsize=10.5, y=1 + .9/height)
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'gene_cnv_matrix.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return states


def core_size_null(sensitivity_dir, core_symbols, min_cnvs):
    """Observed size and null draws for the 'both' variant at this threshold."""
    sensitivity_dir = Path(sensitivity_dir)
    tests = pd.read_csv(resolve_path(sensitivity_dir/'core_size_tests.tsv'), sep='\t')
    null = np.load(sensitivity_dir/'null_sizes.npy')
    if null.shape[1] != len(tests):
        raise ValueError('Null draws do not match the sensitivity variants')
    membership = pd.read_csv(resolve_path(sensitivity_dir/'gene_membership.tsv'), sep='\t')
    results = {}
    for k in sorted(tests.loc[tests['mode'] == 'both', 'min_cnvs']):
        j = int(np.flatnonzero((tests['mode'] == 'both') & (tests.min_cnvs == k))[0])
        row = tests.iloc[j]
        p = (1 + int((null[:, j] >= row.observed_genes).sum()))/(len(null) + 1)
        if not np.isclose(p, row.p_empirical):
            raise ValueError('Null draws do not reproduce the stored empirical P-value')
        results[int(k)] = (row, null[:, j])
    if min_cnvs not in results:
        raise ValueError(f'No sensitivity null for the both-target core at ≥{min_cnvs} CNVs')
    expected = set(membership.loc[membership[f'both_min_{min_cnvs}'], 'symbol'].astype(str))
    if expected != set(map(str, core_symbols)):
        raise ValueError('Sensitivity run and network export define different cores; rerun with the same inputs')
    return results


def plot_core_size_null(results, out, *, min_cnvs):
    import matplotlib.pyplot as plt

    ks = sorted(results)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2*len(ks), 3.6))
    for ax, k in zip(np.atleast_1d(axes), ks):
        row, draws = results[k]
        bins = np.arange(draws.min(), max(draws.max(), row.observed_genes) + 2) - .5
        if len(bins) > 60:
            bins = 50
        highlight = k == min_cnvs
        ax.hist(draws, bins=bins, color='#86b6ef' if highlight else '#d3d1c7', edgecolor='white', linewidth=.4)
        ax.axvline(row.observed_genes, color='#e34948', linewidth=2)
        ax.axvspan(row.null_low, row.null_high, color='#1a1a19', alpha=.06, zorder=0)
        ax.text(.97, .95, f'observed {row.observed_genes:.0f}\nnull mean {row.null_mean:.1f}\n'
                          f'95% null {row.null_low:.0f}–{row.null_high:.0f}\n'
                          f'P = {row.p_empirical:.3g}, q = {row.q_sensitivity:.3g}',
                ha='right', va='top', transform=ax.transAxes, fontsize=8.5, color='#1a1a19')
        ax.set_title(f'≥{k} CNVs' + (' (this core)' if highlight else ''), fontsize=10,
                     fontweight='bold' if highlight else 'normal')
        ax.set_xlabel('Core genes (SCHEMA + GWAS)')
        ax.spines[['top', 'right']].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel('Null draws')
    fig.suptitle(f'Core size vs {len(draws):,} joint degree-matched CNV seed permutations '
                 '(SCZ profiles fixed; red line = observed)', fontsize=10.5)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'core_size_null.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def shared_core_figures(network_dir, *, sensitivity_dir, annotation_dir, output_root):
    """Write the three figures and their tables for the shared core of one network export."""
    network_dir, sensitivity_dir = Path(network_dir), Path(sensitivity_dir)
    manifest = json.loads((network_dir/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Shared-network export must be complete')
    min_cnvs = manifest['identity']['min_cnvs']
    cnv_names = sorted(n for n, e in manifest['identity']['entities'].items() if e['kind'] == 'cnv')
    index = pd.read_csv(resolve_path(network_dir/'network_index.tsv'), sep='\t')
    if 'shared_core' not in set(index.kind):
        raise ValueError('Shared core requires both SCHEMA and GWAS targets')
    core_dir = network_dir/index.loc[index.kind == 'shared_core', 'directory'].iloc[0]
    identity = {'network_manifest_sha256': file_hash(network_dir/'manifest.json'),
                'sensitivity_manifest_sha256': file_hash(sensitivity_dir/'manifest.json'),
                'code': file_hash(__file__),
                'gmt': {n: file_hash(Path(annotation_dir)/n) for n in FUNCTION_LIBRARIES}}
    out = run_dir(output_root, identity, 'core-figures', f'min{min_cnvs}cnvs')
    if (out/'manifest.json').exists():
        from cnv_overlap.results import verify_run
        verify_run(out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(resolve_path(core_dir/'genes.tsv'), sep='\t')
    genes['symbol'] = genes.symbol.astype(str)
    edges = pd.read_csv(resolve_path(core_dir/'edges.tsv'), sep='\t')
    genes = genes.merge(function_categories(genes.symbol, annotation_dir), on='symbol', how='left')
    null = core_size_null(sensitivity_dir, genes.symbol, min_cnvs)
    plot_function_network(genes, edges, out, min_cnvs=min_cnvs)
    states = plot_gene_cnv_matrix(genes, cnv_names, out, min_cnvs=min_cnvs)
    plot_core_size_null(null, out, min_cnvs=min_cnvs)
    genes[['entrez', 'symbol', 'function', 'evidence_term', 'n_cnvs_both', 'cnvs_both',
           'n_cnvs_schema', 'n_cnvs_gwaspri', 'gene_origin']].sort_values(
        ['function', 'n_cnvs_both', 'symbol'], ascending=[True, False, True]).to_csv(
        resolve_path(out/'gene_functions.tsv'), sep='\t', index=False)
    states.to_csv(resolve_path(out/'gene_cnv_matrix.tsv'), sep='\t', index=False)
    pd.DataFrame([row for row, _ in null.values()]).drop(columns='genes').to_csv(
        resolve_path(out/'core_size_test.tsv'), sep='\t', index=False)
    (out/'pies_network.txt').write_text(str((resolve_path(core_dir/'network.png')).resolve()) + '\n')
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'network_dir': str(network_dir.resolve()), 'sensitivity_dir': str(sensitivity_dir.resolve()),
        'function_categories': dict(FUNCTION_CATEGORIES), 'function_libraries': list(FUNCTION_LIBRARIES),
        'interpretation': 'Function categories are GO/SynGO term-name matches in fixed priority order '
                          '(descriptive labels). Core-size P-values reuse the sensitivity null draws.'})
    return out
