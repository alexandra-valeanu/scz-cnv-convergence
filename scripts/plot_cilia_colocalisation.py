"""Plot CNV, SCHEMA, and GWAS colocalization with cilia gene sets.

SCHEMA/GWAS appear when significant at q < 0.05: with either cilia list in
figures 1–2 and with both lists in the remaining figures."""
from cnv_overlap.results import network_root as configured_network_root, resolve_path
import argparse
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from results_layout import results_dir

ROOT = Path(__file__).resolve().parents[1]
SETS = {'scgsv2_first_order': 'SYSCILIA gold standard v2\n(primary cilium genes)',
        'wp4803_ciliopathies': 'WikiPathways WP4803\n(ciliopathy genes)'}
SHORT = {'scgsv2_first_order': 'SYSCILIA', 'wp4803_ciliopathies': 'WP4803'}

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SET_COLOR = {'scgsv2_first_order': BLUE, 'wp4803_ciliopathies': ORANGE}
SCZ_LABEL = {'schema': 'SCHEMA', 'gwaspri': 'GWAS'}


def display(q):
    return SCZ_LABEL.get(q, q)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})


def load(base):
    runs = {}
    for s in SETS:
        from cnv_overlap.results import select_run
        if (base/s/'selections.json').exists():
            bundle = select_run(base/s)
            record = json.loads((bundle/'manifest.json').read_text())
            run = Path(record['identity']['pair_run'])
            summary = bundle/'summary.tsv'
            mapping = bundle/'seed_mapping.tsv'
        else:
            run = ROOT/json.loads((base/f'{s}_current.json').read_text())['run']
            bundle = run
            summary = base/f'{s}_summary.tsv'
            mapping = base/f'{s}_seed_mapping.tsv'
        runs[s] = dict(run=run, bundle=bundle, mapping=mapping,
            summary=pd.read_csv(summary, sep='\t'),
            cand=pd.read_csv(resolve_path(run/'candidate_genes.tsv'), sep='\t'),
            tests=pd.read_csv(resolve_path(run/'pair_tests.tsv'), sep='\t'),
            nps=pd.read_csv(resolve_path(run/'nps.tsv'), sep='\t', index_col='entrez'))
    return runs


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'{name}.{ext}', dpi=300 if ext == 'png' else None)
    plt.close(fig)
    print('wrote', resolve_path(out/f'{name}.png'))


def fig_observed_vs_chance(runs, order, title, out):
    """Per CNV (and SCHEMA/GWAS): genes near both it and cilia genes, versus the chance range."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6 + 0.3*(len(order) - 9)), sharey=True)
    y = np.arange(len(order))[::-1]
    n_cnv = sum(q not in SCZ_LABEL for q in order)
    xmax = 620
    for r in runs.values():
        t = r['tests'].query("primary").set_index('a').loc[order]
        xmax = max(xmax, 1.2*max(t.observed.max(), (t.null_mean + 1.96*t.null_sd).max()))
    for ax, (s, r) in zip(axes, runs.items()):
        t = r['tests'].query("primary").set_index('a').loc[order]
        sm = r['summary'].set_index('query').loc[order]
        lo, hi = t.null_mean - 1.96*t.null_sd, t.null_mean + 1.96*t.null_sd
        ax.hlines(y, lo, hi, color='#cfcec9', lw=7, capstyle='round', zorder=1)
        ax.scatter(t.null_mean, y, marker='|', s=90, color=INK2, lw=1.5, zorder=2)
        sig = sm.significant.values
        ax.scatter(t.observed[sig], y[sig], s=70, color=SET_COLOR[s], zorder=3,
                   edgecolor='white', lw=1.5)
        ax.scatter(t.observed[~sig], y[~sig], s=60, facecolor='white',
                   edgecolor=MUTED, lw=1.5, zorder=3)
        for yi, (obs, ratio, sg) in zip(y, zip(t.observed, sm.size_ratio, sig)):
            if sg:
                ax.text(obs + 12, yi, f'{ratio:.1f}× chance', va='center',
                        fontsize=9, color=INK)
        ax.set_title(SETS[s], fontsize=11, color=INK, loc='left')
        if n_cnv < len(order):
            ax.axhline(y[n_cnv - 1] - 0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_xlim(0, xmax)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel('genes close to both the CNV (or gene set)\nand the cilia genes'
                      if n_cnv < len(order) else 'genes close to both the CNV and the cilia genes')
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, [display(q) for q in order])
    handles = [
        Line2D([], [], marker='o', ls='', color=INK2, markeredgecolor='white',
               markersize=9, label='observed – more than chance (q < 0.05)'),
        Line2D([], [], marker='o', ls='', markerfacecolor='white',
               markeredgecolor=MUTED, markersize=8, label='observed – not more than chance'),
        Line2D([], [], color='#cfcec9', lw=7, label='expected by chance (95% range,\n5,000 shuffles)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle(title,
                 x=0.06, ha='left', fontsize=13, color=INK, y=1.02)
    save(fig, out, '1_observed_vs_chance')


def fig_observed_vs_chance_heatmap(runs, order, title, out):
    """Figure 1 as a heatmap: observed / expected per CNV (rows) and cilia list (columns)."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle
    ratio = pd.concat([r['summary'].set_index('query').size_ratio.rename(s) for s, r in runs.items()],
                      axis=1).loc[order]
    sig = pd.concat([r['summary'].set_index('query').significant.rename(s) for s, r in runs.items()],
                    axis=1).loc[order]
    val = np.log2(ratio.values)
    lim = max(1.0, np.abs(val).max())
    cmap = LinearSegmentedColormap.from_list('below_above', ['#3b6fb6', '#efeeea', '#c2412d'])
    norm = TwoSlopeNorm(0, -lim, lim)

    fig, ax = plt.subplots(figsize=(5.4, 1.0 + 0.42*len(order)))
    im = ax.imshow(val, cmap=cmap, norm=norm, aspect='auto')
    for i in range(len(order)):
        for j in range(ratio.shape[1]):
            v, sg = ratio.values[i, j], sig.values[i, j]
            dark = val[i, j] > 0.55*lim
            ax.text(j, i, f'{v:.1f}×', ha='center', va='center', fontsize=9.5,
                    color='white' if dark else (INK if sg else INK2),
                    fontweight='bold' if sg else 'normal')
            if sg:
                ax.add_patch(Rectangle((j - 0.45, i - 0.42), 0.9, 0.84, fill=False, ec=INK, lw=1.6, zorder=3))
    ax.set_xticks(range(ratio.shape[1]), [SETS[s] for s in ratio.columns], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), [display(q) for q in order])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, ratio.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='minor', length=0)
    n_cnv = sum(q not in SCZ_LABEL for q in order)
    if n_cnv < len(order):
        ax.axhline(n_cnv - 0.5, color=INK2, lw=1.2, ls=(0, (3, 3)))

    ticks = [t for t in (0.5, 0.67, 1, 1.5, 2, 3) if abs(np.log2(t)) <= lim + 1e-9]
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04, ticks=np.log2(ticks))
    cb.ax.set_yticklabels([f'{t:g}×' for t in ticks], fontsize=8.5)
    cb.set_label('genes close to both, observed ÷ expected by chance', fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    fig.text(0.02, -0.01, 'outlined, bold = more than chance (q < 0.05, 5,000 shuffles)\n'
             'red = more genes than chance, blue = fewer', fontsize=8.5, color=INK2, va='top')
    fig.suptitle(title, x=0.02, ha='left', fontsize=12, color=INK, y=1.0)
    save(fig, out, '1_observed_vs_chance_heatmap')


def fig_set_agreement(runs, order, out):
    """Do the two independent cilia gene lists agree, CNV by CNV?"""
    a = runs['scgsv2_first_order']['summary'].set_index('query').loc[order]
    b = runs['wp4803_ciliopathies']['summary'].set_index('query').loc[a.index]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.axhline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    ratios = pd.concat([a.size_ratio, b.size_ratio])
    lim = (min(0.4, ratios.min() - 0.1), max(2.9, ratios.max() + 0.3))
    ax.plot(lim, lim, color=GRID, lw=1, zorder=0)
    both = a.significant & b.significant
    ax.scatter(a.size_ratio[both], b.size_ratio[both], s=80, color=BLUE,
               edgecolor='white', lw=1.5, zorder=3)
    ax.scatter(a.size_ratio[~both], b.size_ratio[~both], s=70, facecolor='white',
               edgecolor=MUTED, lw=1.5, zorder=3)
    nudge = {'16p13.11': (-0.06, 0.07)}
    for cnv in a.index[both]:
        dx, dy = nudge.get(cnv, (0.04, 0))
        ax.text(a.size_ratio[cnv] + dx, b.size_ratio[cnv] + dy, display(cnv), fontsize=9,
                va='center', color=INK if both[cnv] else INK2,
                ha='right' if dx < 0 else 'left')
    rest = a.index[~both]
    ax.annotate('\n'.join(map(display, rest)), xy=(a.size_ratio[rest].max() + 0.04, b.size_ratio[rest].mean()),
                xytext=(1.1, 0.62), fontsize=8.5, color=INK2, va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.text(lim[1] - 0.03, 1.03, 'chance level', ha='right', va='bottom',
            fontsize=8, color=MUTED)
    ax.set_xlabel('× chance, SYSCILIA gold standard v2')
    ax.set_ylabel('× chance, WikiPathways WP4803')
    n_scz = sum(q in SCZ_LABEL for q in a.index[both])
    what = f'{both.sum() - n_scz} CNVs' + (f' and {n_scz} SCZ gene set{"s" * (n_scz > 1)}' if n_scz else '')
    ax.set_title(f'Both cilia gene lists pick out the same\n{what}',
                 loc='left', fontsize=12, color=INK)
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='', color=BLUE, markeredgecolor='white',
               markersize=9, label='more than chance with both lists'),
        Line2D([], [], marker='o', ls='', markerfacecolor='white',
               markeredgecolor=MUTED, markersize=8, label='not more than chance')],
        loc='lower right', frameon=False, fontsize=9, bbox_to_anchor=(1, 0.08))
    save(fig, out, '2_cilia_lists_agree')


def gene_descriptions():
    with zipfile.ZipFile(ROOT/'input/annotations/proteinatlas.tsv.zip') as z:
        with z.open(z.namelist()[0]) as fh:
            hpa = pd.read_csv(fh, sep='\t', usecols=['Gene', 'Gene description'])
    return hpa.drop_duplicates('Gene').set_index('Gene')['Gene description']


def fig_shared_genes(runs, sig, out, min_cnvs=3):
    """Genes close to cilia genes AND to >= min_cnvs of the significant CNVs.

    Significant SCHEMA/GWAS sets are extra columns; they do not count towards min_cnvs.
    """
    sig_cnv = [q for q in sig if q not in SCZ_LABEL]
    rows = []
    for s, r in runs.items():
        c = r['cand'].query('a in @sig')
        rows.append(c.assign(set=s)[['symbol', 'a', 'set', 'seed_b']])
    c = pd.concat(rows)
    hits = c.groupby(['symbol', 'a']).set.nunique().unstack(fill_value=0)
    hits = hits.reindex(columns=sig, fill_value=0)
    n = (hits[sig_cnv] > 0).sum(axis=1)
    keep = hits[n >= min_cnvs].copy()
    # only genes found with both cilia lists for at least min_cnvs CNVs (robust core)
    keep = keep[(keep[sig_cnv] == 2).sum(axis=1) >= min_cnvs]
    is_seed = c.groupby('symbol').seed_b.any()
    desc = gene_descriptions()
    keep['n'] = (keep[sig_cnv] > 0).sum(axis=1)
    keep = keep.sort_values(['n', *sig], ascending=False)
    genes = keep.index.tolist()

    h = 0.24*len(genes) + 1.6
    fig, ax = plt.subplots(figsize=(8.2, h))
    for i, g in enumerate(genes):
        yi = len(genes) - 1 - i
        for j, cnv in enumerate(sig):
            v = keep.at[g, cnv]
            if v == 2:
                ax.scatter(j, yi, s=95, marker='s', color=BLUE, lw=0)
            elif v == 1:
                ax.scatter(j, yi, s=95, marker='s', color='#b7d3f6', lw=0)
        ax.text(-0.75, yi, g, ha='right', va='center', fontsize=9,
                color=INK, fontweight='bold' if not is_seed.get(g, False) else 'normal')
        d = str(desc.get(g, ''))
        ax.text(len(sig) - 0.4, yi, d[:62] + ('…' if len(d) > 62 else ''),
                va='center', fontsize=8, color=INK2)
    if len(sig_cnv) < len(sig):
        ax.axvline(len(sig_cnv) - 0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(range(len(sig)), [display(q) for q in sig], rotation=0, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.set_xlim(-0.6, len(sig) - 0.5)
    ax.set_ylim(-1.8, len(genes) - 0.3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.set_title(f'{len(genes)} genes close to cilia genes and to ≥{min_cnvs} of the {len(sig_cnv)} CNVs\n',
                 loc='left', fontsize=12, color=INK, x=-0.32)
    ax.text(-0.32, -1.2, f'Kept only if found with both cilia lists.  Bold name = not itself '
            f'on a cilia list (a new neighbour); plain = known cilia gene.'
            + ('\nSCHEMA / GWAS columns: the gene is also close to those schizophrenia genes.'
               if len(sig_cnv) < len(sig) else ''),
            transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK2, va='top')
    save(fig, out, '4_shared_genes')
    keep.assign(description=[desc.get(g, '') for g in genes],
                on_cilia_list=[bool(is_seed.get(g, False)) for g in genes]) \
        .to_csv(resolve_path(out/'4_shared_genes.tsv'), sep='\t')


def fig_overlap_counts(runs, sig, out):
    """How many candidate genes each CNV shares with the others (per cilia list)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, (s, r) in zip(axes, runs.items()):
        c = r['cand'].query('a in @sig')
        n = c.groupby('symbol').a.nunique()
        per = c.assign(n=c.symbol.map(n)).groupby(['a', 'n']).size().unstack(fill_value=0)
        ks = list(range(1, len(sig) + 1))
        per = per.reindex(index=sig[::-1], columns=ks, fill_value=0)
        left = np.zeros(len(per))
        cmap = plt.get_cmap('Blues' if s.startswith('scgs') else 'Oranges')
        shades = [cmap(v) for v in np.linspace(0.2, 0.95, len(ks))]
        unit = 'sets' if len(sig) > len([q for q in sig if q not in SCZ_LABEL]) else 'CNVs'
        for k, col in zip(per.columns, shades):
            ax.barh([display(q) for q in per.index], per[k], left=left, color=col, height=0.62,
                    edgecolor='white', lw=1.5, label=f'this {unit[:-1]} only' if k == 1 else f'{k} {unit}')
            left += per[k].values
        for yi, tot in enumerate(left):
            ax.text(tot + 6, yi, f'{int(tot)}', va='center', fontsize=9, color=INK2)
        ax.set_title(SETS[s].replace('\n', ' '), fontsize=10.5, loc='left', color=INK)
        ax.set_xlabel('candidate genes')
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        ax.legend(title='candidate gene shared by', frameon=False, fontsize=8,
                  title_fontsize=8, loc='upper left', bbox_to_anchor=(0, -0.28), ncol=min(len(ks), 4))
    fig.suptitle(f'Most candidate genes belong to one {unit[:-1]}; a smaller core is shared',
                 x=0.06, ha='left', fontsize=12, color=INK, y=1.04)
    save(fig, out, '5_candidates_shared_between_cnvs')


def venn2(ax, A, B, col_a, col_b, col_ab):
    """Area-proportional two-set Venn diagram: circle A on the left, B on the right."""
    from matplotlib.patches import Circle
    na, nb, nab = len(A), len(B), len(A & B)
    ra, rb = np.sqrt(na/np.pi), np.sqrt(nb/np.pi)

    def overlap(d):  # area of the lens between two circles at centre distance d
        if d >= ra + rb:
            return 0.0
        if d <= abs(ra - rb):
            return np.pi*min(ra, rb)**2
        a1 = ra**2*np.arccos((d*d + ra*ra - rb*rb)/(2*d*ra))
        a2 = rb**2*np.arccos((d*d + rb*rb - ra*ra)/(2*d*rb))
        return a1 + a2 - 0.5*np.sqrt((-d+ra+rb)*(d+ra-rb)*(d-ra+rb)*(d+ra+rb))

    lo, hi = abs(ra - rb), ra + rb
    for _ in range(60):  # overlap area shrinks as d grows
        mid = (lo + hi)/2
        lo, hi = (mid, hi) if overlap(mid) > nab else (lo, mid)
    d = (lo + hi)/2
    ca = Circle((0, 0), ra, color=col_a, ec='white', lw=2)
    ax.add_patch(ca)
    ax.add_patch(Circle((d, 0), rb, color=col_b, ec='white', lw=2))
    ax.add_patch(Circle((d, 0), rb, color=col_ab, ec='none', clip_path=ca))
    ax.add_patch(Circle((d, 0), rb, fill=False, ec='white', lw=2))

    scale = max(ra, d + rb)
    regions = [(na - nab, -ra, d - rb), (nab, max(d - rb, -ra), min(ra, d + rb)), (nb - nab, ra, d + rb)]
    for n, x0, x1 in regions:
        if x1 - x0 > 0.22*scale:
            ax.text((x0 + x1)/2, 0, f'{n:,}', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        else:  # region too thin for a label inside: put it just outside, to the right
            ax.annotate(f'{n:,}', ((x0 + x1)/2, 0), (x1 + 0.12*scale, 0.35*scale),
                        fontsize=9, color=INK, va='center',
                        arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.set_xlim(-ra - 0.05*scale, max(ra, d + rb) + 0.35*scale)
    ax.set_ylim(-max(ra, rb)*1.05, max(ra, rb)*1.05)
    ax.set_aspect('equal')
    ax.axis('off')


def fig_list_overlap(runs, base, sig, out, threshold=1.0):
    """How much the two cilia lists overlap: seeds, propagated neighbourhood, candidates."""
    s1, s2 = 'scgsv2_first_order', 'wp4803_ciliopathies'

    def seeds(s):
        m = pd.read_csv(runs[s]['mapping'], sep='\t')
        qc = pd.read_csv(resolve_path(runs[s]['run']/'seed_qc.tsv'), sep='\t').query("kind == 'target'")
        miss = {int(x) for x in str(qc.missing_ids.iloc[0]).split(',') if x != 'nan'}
        return set(m.entrez.dropna().astype(int)) - miss

    nps = pd.concat([runs[s1]['nps'][s1], runs[s2]['nps'][s2]], axis=1).dropna()
    near = {s: set(nps.index[nps[s] > threshold]) for s in (s1, s2)}
    rows = [('seed genes', seeds(s1), seeds(s2)),
            ('overlapping neighbourhood', near[s1], near[s2])]
    for cnv in sig:
        rows.append((display(cnv), *(set(runs[s]['cand'].loc[lambda d: d.a == cnv, 'entrez']) for s in (s1, s2))))

    BOTH = '#52514e'
    fig = plt.figure(figsize=(7.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, len(sig)], hspace=0.4)
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.15)
    ax_a = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    for ax, (label, A, B) in zip((ax_a, ax_a2), rows[:2]):
        venn2(ax, A, B, BLUE, ORANGE, BOTH)
        ax.set_title(label, fontsize=9.5, color=INK, pad=2)
        ax.text(0.5, -0.04, f'{len(A & B)/len(B):.0%} of WP4803 also in SYSCILIA',
                transform=ax.transAxes, ha='center', va='top', fontsize=8, color=INK2)
    # align the panel-a heading with panel b's (whose title sits at x=-0.3 of its axes)
    pb, pa = ax_c.get_position(), ax_a.get_position()
    fig.text(pb.x0 - 0.3*pb.width, pa.y1 + 0.07, 'a   Overlapping genes size',
             fontsize=10.5, color=INK)
    for ax, part, title in [(ax_c, rows[2:], 'b   Genes per CNV neighbourhood overlap')]:
        y = np.arange(len(part))[::-1]
        for yi, (label, A, B) in zip(y, part):
            segs = [(len(A - B), BLUE), (len(A & B), BOTH), (len(B - A), ORANGE)]
            tot, left = sum(n for n, _ in segs), 0
            for n, col in segs:
                ax.barh(yi, n/tot, left=left, color=col, height=0.6,
                        edgecolor='white', lw=2)
                if n/tot > 0.06:
                    ax.text(left + n/tot/2, yi, f'{n:,}', ha='center', va='center',
                            fontsize=9, color='white', fontweight='bold')
                else:
                    ax.text(left + n/tot/2, yi + 0.42, f'{n:,}', ha='center', va='bottom',
                            fontsize=9, color=INK)
                left += n/tot
            ax.text(1.02, yi, f'{len(A & B)/len(B):.0%} of WP4803\nalso in SYSCILIA',
                    va='center', fontsize=8, color=INK2)
        ax.set_yticks(y, [r[0] for r in part])
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.tick_params(axis='y', length=0)
        for sp in ('bottom', 'left'):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, loc='left', fontsize=10.5, color=INK, x=-0.3, pad=18)
        ax.text(-0.3 + 0.045, 1.02, 'only significant pairs', transform=ax.transAxes,
                fontsize=8.5, color=INK2, va='bottom')
    ax_a.legend(handles=[Patch(color=BLUE, label='SYSCILIA only'),
                         Patch(color=BOTH, label='both lists'),
                         Patch(color=ORANGE, label='WP4803 only')],
                ncol=3, frameon=False, fontsize=9, loc='lower left',
                bbox_to_anchor=(-0.07, 1.3))

    save(fig, out, '6_cilia_list_overlap')


def network_label(network):
    """'PCNET2.2_brain' -> 'brain-specific PCNet2.2 network', 'PCNET2.2' -> 'PCNet2.2 network'."""
    name = network.removesuffix('_brain')
    name = {'PCNET2.2': 'PCNet2.2', 'STRING_high_conf': 'STRING (high confidence)'}.get(name, name)
    return f'brain-specific {name} network' if network.endswith('_brain') else f'{name} network'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', default='PCNET2.2_brain')
    args = ap.parse_args()
    base = results_dir(configured_network_root(args.network, ROOT/'results'), 'cilia_colocalisation')
    runs = load(base)
    from cnv_overlap.provenance import run_dir, file_hash
    from cnv_overlap.results import finalize_manifest, verify_run
    identity = {'network': args.network, 'sources': {name: file_hash(r['bundle']/'manifest.json') for name, r in runs.items()}, 'code': file_hash(__file__)}
    out = run_dir(base/'comparison', identity, 'cilia-comparison')
    if (out/'manifest.json').exists():
        verify_run(out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    ref = runs['scgsv2_first_order']['summary'].sort_values('z', ascending=False)
    flags = pd.concat([r['summary'].set_index('query').significant.rename(s)
                       for s, r in runs.items()], axis=1)
    cnvs = ref.loc[ref.query_type == 'cnv', 'query'].tolist()
    # SCHEMA/GWAS are drawn only if they colocalise significantly with a cilia list.
    scz_any = [q for q in SCZ_LABEL if flags.loc[q].any()]
    scz_all = [q for q in SCZ_LABEL if flags.loc[q].all()]
    for q in SCZ_LABEL:
        print(f'{display(q)}: significant with', [s for s in runs if flags.at[q, s]] or 'neither cilia list')
    order = cnvs + scz_any
    sig = [c for c in cnvs if flags.loc[c].all()] + scz_all
    n_sig_cnv = sum(flags.loc[c].any() for c in cnvs)
    title = f'{n_sig_cnv} CNVs' + (f' and the {" and ".join(map(display, scz_any))} genes' if scz_any else '') \
        + f' sit close to cilia genes in the {network_label(args.network)}'
    fig_observed_vs_chance(runs, order, title, out)
    fig_observed_vs_chance_heatmap(runs, order, title, out)
    fig_set_agreement(runs, order, out)
    fig_shared_genes(runs, sig, out)
    fig_overlap_counts(runs, sig, out)
    fig_list_overlap(runs, base, sig, out)
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity})


if __name__ == '__main__':
    main()
