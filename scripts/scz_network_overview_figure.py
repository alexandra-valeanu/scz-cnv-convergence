"""Plot inputs, propagation, thresholds, label null, and the SCHEMA–GWAS network.

Reconstructs the candidate set and label null from the saved target-pair settings
and checks agreement with the source run."""
from cnv_overlap.results import network_root as configured_network_root, resolve_path
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT/'scripts'))

from results_layout import results_dir
from scz_core_analysis import (GWAS, SCHEMA, AnalysisConfig, Propagator, build_gsz, label_null,
                               load_network_inputs)

SCHEMA_COLOR, GWAS_COLOR, GSZ_COLOR = '#2a78d6', '#eb6834', '#4a3aa7'
BOTH_SEED_COLOR = '#1a1a19'
INK, MUTED, GRID, NULL_COLOR = '#1a1a19', '#6b6a64', '#e4e3dd', '#b4b2a9'


def load(network):
    cfg = json.loads((ROOT/'configs/scz_core_analysis.json').read_text())
    spec, root, inputs, _ = load_network_inputs(ROOT/cfg['disorder_config'], network, min_seeds=cfg['min_seeds'])
    config = AnalysisConfig.from_dict(spec['analysis'])
    A, nodes, symbols, entities = inputs[:4]
    prop = Propagator(A, nodes, root/spec['cache_dir'], config.propagation)
    from cnv_overlap.results import select_run
    pair_dir = select_run(results_dir(root/spec['results_root'], 'scz_target_pair'))
    if json.loads((pair_dir/'manifest.json').read_text())['identity']['graph'] != prop.graph_id:
        raise ValueError(f'{pair_dir} was not computed on {network}')
    gsz, zs, seeds = build_gsz(prop, entities, config, pair_dir)
    observed, null = label_null(zs[SCHEMA], zs[GWAS], prop.nodes, seeds[SCHEMA], seeds[GWAS],
                                reps=config.permutation_reps, seed=config.propagation.seed,
                                overlap=config.overlap, thresholds=config.thresholds)
    tests = pd.read_csv(resolve_path(pair_dir/'pair_tests.tsv'), sep='\t').set_index('statistic')
    if not np.isclose(np.mean(null['size']), tests.at['size', 'null_mean']):
        raise ValueError('Regenerated label null does not match the recorded target-pair null')
    raw = pd.read_csv(resolve_path(pair_dir/'seed_qc.tsv'), sep='\t').set_index('entity')
    return dict(source=pair_dir, prop=prop, config=config, symbols=symbols, gsz=gsz, zs=zs, seeds=seeds, raw=raw,
                null=np.asarray(null['size']), observed=observed['size'], test=tests.loc['size'])


def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7.5, length=3)


def panel_inputs(ax, d, network):
    from matplotlib.patches import FancyBboxPatch
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    n_nodes, n_edges = len(d['prop'].nodes), d['prop'].A.nnz//2
    boxes = [
        (0.70, SCHEMA_COLOR, 'Rare variants', 'SCHEMA exome meta-analysis\nQ < 0.05',
         f"{d['raw'].at[SCHEMA, 'raw_seeds']} genes  →  {d['raw'].at[SCHEMA, 'retained_seeds']} in network"),
        (0.39, GWAS_COLOR, 'Common variants', 'SCZ GWAS prioritized genes',
         f"{d['raw'].at[GWAS, 'raw_seeds']} genes  →  {d['raw'].at[GWAS, 'retained_seeds']} in network"),
        (0.04, MUTED, 'Interactome', f"{network.replace('_brain', '')} restricted to\nbrain-expressed genes (HPA TPM > 1)",
         f'{n_nodes:,} genes · {n_edges/1e6:.2f} M edges'),
    ]
    for y, color, head, body, count in boxes:
        ax.add_patch(FancyBboxPatch((0.02, y), 0.96, 0.27, boxstyle='round,pad=0,rounding_size=0.03',
                                    fc='white', ec=GRID, lw=1))
        ax.add_patch(FancyBboxPatch((0.02, y), 0.025, 0.27, boxstyle='square,pad=0', fc=color, ec='none'))
        ax.text(0.09, y + 0.215, head, fontsize=9, weight='bold', color=INK, va='center')
        ax.text(0.09, y + 0.125, body, fontsize=7.5, color=MUTED, va='center', linespacing=1.25)
        ax.text(0.09, y + 0.035, count, fontsize=7.5, color=INK, va='center')


def panel_propagation(ax, d):
    thr = d['config'].thresholds.individual
    bins = np.linspace(-4, 25, 88)
    for name, color, label in ((SCHEMA, SCHEMA_COLOR, 'SCHEMA'), (GWAS, GWAS_COLOR, 'GWAS')):
        z = np.clip(d['zs'][name], bins[0], bins[-1])
        ax.hist(z, bins=bins, histtype='step', color=color, lw=1.6, label=f'{label}: {(z > thr).sum():,} genes > {thr:g}')
    ax.axvline(thr, color=INK, lw=0.9, ls='--')
    ax.set_yscale('log')
    ax.set(xlabel='Network proximity score (NPS)', ylabel='Genes')
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right', handlelength=1.4)
    p = d['config'].propagation
    ax.text(0.98, 0.70, f'heat kernel, α = {p.alpha:g}\nz-score vs {p.reference_reps:,}\ndegree-matched random\nseed sets',
            transform=ax.transAxes, ha='right', va='top', fontsize=7, color=MUTED, linespacing=1.3)
    style(ax)


def panel_coloc(ax, d):
    t = d['config'].thresholds
    a, b = d['zs'][SCHEMA], d['zs'][GWAS]
    in_gsz = np.zeros(len(a), dtype=bool)
    in_gsz[d['gsz']] = True
    both = (a > t.individual) & (b > t.individual)
    ax.scatter(a[~in_gsz], b[~in_gsz], s=2, c=NULL_COLOR, alpha=0.35, lw=0, rasterized=True)
    ax.scatter(a[in_gsz], b[in_gsz], s=5, c=GSZ_COLOR, alpha=0.8, lw=0, rasterized=True)
    ax.set_xscale('symlog', linthresh=3, linscale=1.2)
    ax.set_yscale('symlog', linthresh=3, linscale=1.2)
    lim = (-4, 1.15*max(a.max(), b.max()))
    ax.set(xlim=lim, ylim=lim)
    ticks = [-3, 0, 1, 3, 10, 20]
    ax.set_xticks(ticks, [str(x) for x in ticks])
    ax.set_yticks(ticks, [str(x) for x in ticks])
    ax.minorticks_off()
    ax.axvline(t.individual, color=INK, lw=0.8, ls='--')
    ax.axhline(t.individual, color=INK, lw=0.8, ls='--')
    x = np.linspace(t.individual, t.product/t.individual, 100)
    ax.plot(x, t.product/x, color=INK, lw=1.2)
    ax.set(xlabel='NPS, SCHEMA', ylabel='NPS, GWAS')
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    ax.text(0.03, 0.97,
            f'{len(a):,} genes\n'
            f'both NPS > {t.individual:g}: {both.sum():,}\n'
            f'and product ≥ {t.product:g}: ', transform=ax.transAxes, va='top', fontsize=7.5, color=INK,
            linespacing=1.35)
    ax.text(0.03, 0.795, f'G$_{{SZ}}$ = {len(d["gsz"])} genes', transform=ax.transAxes, va='top', fontsize=8.5,
            color=GSZ_COLOR, weight='bold', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    ax.annotate(f'NPS$_S$ × NPS$_G$ = {t.product:g}', xy=(2.4, t.product/2.4), xytext=(4.5, 0.25),
                fontsize=7, color=MUTED, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    style(ax)


def panel_null(ax, d):
    null, obs, test = d['null'], d['observed'], d['test']
    lo = min(null.min(), obs)
    bins = np.arange(np.floor(lo/5)*5, obs + 10, 5)
    ax.hist(null, bins=bins, color=NULL_COLOR, ec='white', lw=0.4)
    ax.axvline(obs, color=GSZ_COLOR, lw=2)
    ax.text(obs, ax.get_ylim()[1]*0.97, f'observed \n{int(obs)} ', color=INK, fontsize=7.5, va='top', ha='right')
    ax.text(null.max() + 6, ax.get_ylim()[1]*0.97, f'null mean {null.mean():.0f}', color=MUTED, fontsize=7.5,
            va='top', ha='left')
    mant, exp = f'{test.p:.1e}'.split('e')
    ax.text(0.52, 0.5,
            f'{obs/null.mean():.1f}× expected\nz = {test.z:.1f}, p = {float(mant):.0f}×10$^{{{int(exp)}}}$\n'
            f'{int(test.n_ge)}/{len(null):,} permutations ≥ observed',
            transform=ax.transAxes, ha='center', va='center', fontsize=7.5, color=INK, linespacing=1.35)
    ax.set(xlabel='Joint-region size (genes)', ylabel='Gene-label permutations')
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    style(ax)


def panel_network(ax, d, n_labels=6):
    import networkx as nx
    from matplotlib.lines import Line2D
    prop, gsz, zs, symbols = d['prop'], d['gsz'], d['zs'], d['symbols']
    genes = prop.nodes[gsz]
    G = nx.from_scipy_sparse_array(prop.A[gsz][:, gsz])
    # Lay out the main component on its own; the few small components sit in a row
    # beneath it, otherwise the spring layout flings them out and collapses the core.
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos = nx.spring_layout(G.subgraph(comps[0]), iterations=300, seed=7)
    rest = [n for c in comps[1:] for n in sorted(c)]
    xmin = min(x for x, _ in pos.values()); xmax = max(x for x, _ in pos.values())
    ymin = min(y for _, y in pos.values())
    for k, n in enumerate(rest):
        pos[n] = np.array([xmin + (k + 0.5)*(xmax - xmin)/len(rest), ymin - 0.18])
    if rest:
        ax.text(xmin, ymin - 0.1, f'{len(rest)} genes outside the main component', fontsize=6.5, color=MUTED)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=NULL_COLOR, alpha=0.04, width=0.4)
    s_seed = np.isin(genes, d['seeds'][SCHEMA])
    g_seed = np.isin(genes, d['seeds'][GWAS])
    product = zs[SCHEMA][gsz]*zs[GWAS][gsz]
    size = 6 + 26*np.sqrt(product/product.max())
    xy = np.array([pos[i] for i in range(len(genes))])
    groups = [(~s_seed & ~g_seed, GSZ_COLOR, 'Network-implicated (not a seed)'),
              (s_seed & ~g_seed, SCHEMA_COLOR, 'SCHEMA seed'),
              (g_seed & ~s_seed, GWAS_COLOR, 'GWAS seed'),
              (s_seed & g_seed, BOTH_SEED_COLOR, 'SCHEMA and GWAS seed')]
    for mask, color, _ in groups:
        ax.scatter(xy[mask, 0], xy[mask, 1], s=size[mask], c=color, ec='white', lw=0.4,
                   zorder=3 if color == GSZ_COLOR else 4)
    # Labels fan out to the right margin so they stay readable over the dense core.
    top = np.argsort(-product)[:n_labels]
    right = xy[:, 0].max() + 0.06*np.ptp(xy[:, 0])
    ys = np.linspace(xy[:, 1].max(), xy[:, 1].max() - 0.45*np.ptp(xy[:, 1]), n_labels)
    for i, y in zip(top[np.argsort(-xy[top, 1])], ys):
        ax.annotate(symbols.get(int(genes[i]), str(genes[i])), xy[i], xytext=(right, y), textcoords='data',
                    fontsize=6.5, color=INK, va='center', zorder=5,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=1, shrinkB=2))
    span = np.ptp(xy[:, 0])
    ax.set_xlim(xy[:, 0].min() - 0.03*span, right + 0.14*span)
    ax.set_axis_off()
    handles = [Line2D([], [], marker='o', ls='', ms=6, mfc=c, mec='white', label=f'{label} ({mask.sum()})')
               for mask, c, label in groups]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=2, handletextpad=0.3, columnspacing=1.0)
    ax.text(0.5, 1.0, f'{G.number_of_nodes()} genes · {G.number_of_edges():,} edges · '
            f'largest component {len(max(nx.connected_components(G), key=len))}\n'
            f'node size ∝ √(NPS$_S$ × NPS$_G$); labels: top {n_labels} by product',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=7, color=MUTED, linespacing=1.3)


def figure(d, network, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(19, 4.6), facecolor='white')
    gs = fig.add_gridspec(1, 5, width_ratios=[1.05, 1, 1, 0.95, 1.45], wspace=0.42,
                          left=0.015, right=0.99, top=0.77, bottom=0.17)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    panel_inputs(axes[0], d, network)
    panel_propagation(axes[1], d)
    panel_coloc(axes[2], d)
    panel_null(axes[3], d)
    panel_network(axes[4], d)
    steps = [('a', 'Seed genes and network', 'Two independent SCZ gene sets'),
             ('b', 'Propagate each seed set', 'Proximity of every gene to the seeds'),
             ('c', 'Keep genes near both', 'NetColoc joint thresholds'),
             ('d', 'Test against chance', 'NPS$_S$ shuffled across genes, 5,000\u00d7'),
             ('e', 'Converged SCZ network (G$_{SZ}$)', 'Fixed target for the multi-CNV analysis')]
    for ax, (letter, title, sub) in zip(axes, steps):
        box = ax.get_position()
        fig.text(box.x0, 0.955, letter, fontsize=13, weight='bold', color=INK, va='top')
        fig.text(box.x0 + 0.014, 0.953, title, fontsize=10.5, weight='bold', color=INK, va='top')
        fig.text(box.x0 + 0.014, 0.905, sub, fontsize=8, color=MUTED, va='top')
    for left, right in zip(axes[:-1], axes[1:]):
        x0, x1 = left.get_position().x1, right.get_position().x0
        y = 0.49
        fig.patches.append(matplotlib.patches.FancyArrowPatch(
            (x0 + 0.004, y), (x1 - 0.028 if right is not axes[-1] else x1 - 0.004, y),
            transform=fig.transFigure, arrowstyle='-|>', mutation_scale=12, color=MUTED, lw=1.2))
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(out/f'scz_network_overview.{suffix}', dpi=300, facecolor='white')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--network', default='PCNET2.2_brain')
    args = parser.parse_args(argv)
    d = load(args.network)
    from cnv_overlap.provenance import run_dir, file_hash
    from cnv_overlap.results import finalize_manifest, verify_run
    identity = {'network': args.network, 'source': str(d['source']), 'source_manifest': file_hash(d['source']/'manifest.json'), 'code': file_hash(__file__)}
    out = run_dir(configured_network_root(args.network, ROOT/'results')/'3_shared_scz_networks/reports/overview', identity, 'overview')
    if (out/'manifest.json').exists():
        verify_run(out)
    else:
        figure(d, args.network, out)
        finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity})
    print(f'G_SZ {len(d["gsz"])} genes; wrote {out}')


if __name__ == '__main__':
    main()
