"""Draw a PCNet brain poster overview from saved results and a propagation schematic."""
import argparse
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.sparse.csgraph import shortest_path

from cnv_overlap.inputs import load_inputs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/PCNET2.2_brain'
SEEDS = RESULTS / '0_inputs/seed_sets/seed_sets.tsv'
PAIR_TESTS = RESULTS / '1_cnv_vs_cnv/convergence/cnv-cnv-alpha0.5-2f9f0ae3bc61/pair_tests.tsv'
SIG_PAIRS = RESULTS / '1_cnv_vs_cnv/pair_network/significant-pairs-fdr0.05-faa0015e5750/significant_pairs.tsv'
LANDSCAPE = RESULTS / '4_scz_landscape/landscape-8a4e9d44008d'
# Derived PCNET2.2_brain config holding the network and the CNV + SCHEMA/GWAS seed sets.
NETWORK_CONFIG = ROOT / 'data/network_configs/PCNET2.2_brain/cnv_disorder_overlap.json'
OUT = ROOT / 'docs/figures/poster_overview'

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
SURFACE, BAND = '#ffffff', '#ffffff'
SET_B, COLOC, NEGATIVE = '#008300', '#6250d6', '#e34948'
ALPHA, NPS_REPS, PERM_REPS = 0.5, 5000, 5000          # settings of the PCNET2.2_brain runs
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'
EDGE_BLUE = '#86b6ef'
MODULE_COLORS = {1: BLUE, 2: ORANGE, 3: AQUA, 4: YELLOW, 5: MAGENTA, 6: GREEN}
BLUES = LinearSegmentedColormap.from_list('blues', ['#f0efec', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
SET_LABELS = {'schema': 'SCHEMA', 'gwaspri': 'GWAS'}
DIST_COLORS = {0: '#0d366b', 1: '#2a78d6', 2: '#9ec5f4', 3: '#d6d5d0'}
DIST_LABELS = {0: 'seed gene itself', 1: '1 step (direct partner)', 2: '2 steps', 3: '3+ steps'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


def tsv(path):
    return pd.read_csv(path, sep='\t')


def short_name(process):
    return process.split(' (GO:')[0]


def clean(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def header(fig, x, y, badge, title, subtitle, square=False):
    """Step badge (circle for workflow, square for results), title and plain-language subtitle."""
    fig.text(x, y, badge, ha='center', va='center', fontsize=17, fontweight='bold', color='white',
             bbox=dict(boxstyle='square,pad=0.35' if square else 'circle,pad=0.32', fc=INK, ec='none'))
    fig.text(x + 0.016, y, title, ha='left', va='center', fontsize=19, fontweight='bold', color=INK)
    fig.text(x - 0.009, y - 0.034, subtitle, ha='left', va='center', fontsize=12.5, color=INK2,
             linespacing=1.35)


def band(fig, y0, y1, label):
    fig.patches.append(FancyBboxPatch((0.012, y0), 0.976, y1 - y0, boxstyle='round,pad=0,rounding_size=0.012',
                                      transform=fig.transFigure, fc=BAND, ec=GRID, lw=1.5, zorder=-10))
    if label:
        fig.text(0.022, (y0 + y1) / 2, label, rotation=90, ha='center', va='center', fontsize=15,
                 fontweight='bold', color=MUTED)


def arrow(fig, start, end, **kw):
    fig.patches.append(FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=28, lw=3, color=MUTED, **kw))


def chip(ax, x, y, text, color, w=0.2, h=0.085, fs=11):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.025',
                                fc=color, ec='none'))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color='white', fontweight='bold')


# ================================================================= workflow (schematics)
def step_seeds(ax, seeds):
    cnvs = seeds[seeds['kind'] == 'cnv'].sort_values('propagated_genes')
    targets = seeds[seeds['kind'] != 'cnv'].sort_values('propagated_genes')
    rows = pd.concat([cnvs, targets])
    ypos = list(range(len(cnvs))) + [len(cnvs) + 0.8 + i for i in range(len(targets))]
    for y, (_, r) in zip(ypos, rows.iterrows()):
        color = BLUE if r['kind'] == 'cnv' else ORANGE
        ax.plot([r['retained_seeds'], r['propagated_genes']], [y, y], color=GRID, lw=3, zorder=1,
                solid_capstyle='round')
        ax.scatter(r['retained_seeds'], y, s=90, color=color, ec=SURFACE, lw=1.5, zorder=3)
        ax.scatter(r['propagated_genes'], y, s=90, color=INK, ec=SURFACE, lw=1.5, zorder=3)
        ax.text(r['retained_seeds'] / 1.18, y, f"{r['retained_seeds']}", ha='right', va='center',
                fontsize=10.5, color=INK2)
        ax.text(r['propagated_genes'] * 1.15, y, f"{r['propagated_genes']:,}", ha='left', va='center',
                fontsize=10.5, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SET_LABELS.get(e, e) for e in rows['entity']], fontsize=12)
    ax.set_xscale('log')
    ax.set_xlim(2, 12000)
    ax.set_ylim(-0.7, ypos[-1] + 0.7)
    ax.set_xticks([3, 10, 100, 1000, 10000])
    ax.set_xticklabels(['3', '10', '100', '1,000', '10,000'])
    ax.set_xlabel('Genes (log scale)')
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    clean(ax)
    # Group brackets: high-risk CNVs, rare-variant and common-variant schizophrenia genes.
    trans = ax.get_yaxis_transform()
    n = len(cnvs)
    groups = [(0, n - 1, 'High-risk\nSCZ CNVs', BLUE)]  # labels stay horizontal
    groups += [(y, y, 'rare variants' if e == 'schema' else 'common variants', ORANGE)
               for y, e in zip(ypos[n:], targets['entity'])]
    for y0, y1, label, color in groups:
        ax.plot([-0.5, -0.5], [y0 - 0.35, y1 + 0.35], transform=trans, color=color, lw=4,
                solid_capstyle='round', clip_on=False)
        ax.text(-0.54, (y0 + y1) / 2, label, transform=trans, ha='right', va='center', fontsize=12,
                fontweight='bold', color=color, linespacing=1.15)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', ms=9, color=BLUE, label='CNV seed genes'),
                       Line2D([], [], marker='o', ls='', ms=9, color=ORANGE, label='SCZ seed genes'),
                       Line2D([], [], marker='o', ls='', ms=9, color=INK, label='after propagation (NPS > 1)')],
              loc='upper center', bbox_to_anchor=(0.25, -0.13), ncol=3, frameon=False, fontsize=11,
              handletextpad=0.1, columnspacing=0.8)


def toy_network(n=16, bridges=3):
    """Two small hub-and-spoke modules (A left, B right) joined by a few bridge genes in the middle.

    Returns the graph, a layout scaled to [-1, 1], and the A and B seeds: the leaf genes on the
    inner edge of each module, paired top-to-bottom by bridge genes between the modules.
    """
    g = nx.disjoint_union(nx.barabasi_albert_graph(n, 1, seed=3), nx.barabasi_albert_graph(n, 1, seed=4))
    rng = np.random.default_rng(2)
    for offset in (0, n):
        for _ in range(3):
            u, v = rng.choice(n, 2, replace=False) + offset
            g.add_edge(int(u), int(v))
    xy = np.zeros((len(g) + bridges, 2))
    seeds = []
    for offset, side in ((0, -1), (n, 1)):
        nodes = list(range(offset, offset + n))
        pos = nx.kamada_kawai_layout(g.subgraph(nodes))
        m = np.array([pos[x] for x in nodes])
        m = (m - m.mean(0)) / np.abs(m - m.mean(0)).max() * 0.8
        xy[nodes] = m + [side * 1.0, 0]
        leaves = [x for x in nodes if g.degree(x) == 1]
        inner = sorted(leaves, key=lambda x: side * xy[x, 0])[:bridges]   # closest to the middle
        seeds.append(sorted(inner, key=lambda x: -xy[x, 1]))
    seeds_a, seeds_b = seeds
    for i in range(bridges):
        g.add_edge(2 * n + i, seeds_a[i])
        g.add_edge(2 * n + i, seeds_b[i])
        xy[2 * n + i] = (xy[seeds_a[i]] + xy[seeds_b[i]]) / 2
    xy /= np.abs(xy).max()
    return g, xy, sorted(seeds_a), sorted(seeds_b)


def separate(xy, radii, gap=0.02, iters=400):
    """Push overlapping nodes apart until every pair is at least radii_i + radii_j + gap apart."""
    xy = xy.copy()
    need = radii[:, None] + radii[None, :] + gap
    np.fill_diagonal(need, 0)
    for _ in range(iters):
        d = xy[:, None] - xy[None, :]
        dist = np.linalg.norm(d, axis=2) + np.eye(len(xy))
        push = np.clip(need - dist, 0, None) / 2
        if push.max() < 1e-4:
            break
        xy += (d / dist[..., None] * push[..., None]).sum(1)
    return xy


def toy_nps(g, seeds):
    """Illustrative NPS for the schematic: high at the seeds, decaying with network distance,
    with a small penalty for hubs (standing in for the degree-matched hub correction)."""
    dist = nx.multi_source_dijkstra_path_length(g, set(seeds))
    by_step = {0: 3.0, 1: 2.0, 2: 0.9, 3: 0.2}
    deg = np.array([g.degree(x) for x in g])
    return np.array([by_step.get(dist.get(x, 9), -0.8) for x in g]) - 0.15 * np.maximum(deg - 3, 0)


def draw_graph(ax, g, xy, center, radius, colors, sizes, ring=(), nodes=None, edge_color='#b8b7b1'):
    p = center + radius * xy
    keep = set(g) if nodes is None else set(nodes)
    segs = [(p[u], p[v]) for u, v in g.edges() if u in keep and v in keep]
    ax.add_collection(LineCollection(segs, colors=edge_color, linewidths=1.1, zorder=1))
    idx = sorted(keep)
    ax.scatter(p[idx, 0], p[idx, 1], s=np.asarray(sizes)[idx], c=np.asarray(colors)[idx], ec='#6f6e69',
               lw=0.7, zorder=2)
    if len(ring):
        ax.scatter(p[list(ring), 0], p[list(ring), 1], s=np.asarray(sizes)[list(ring)] * 2.2,
                   facecolor='none', ec=INK, lw=2.2, zorder=3)


def rounded(pts, radius=14, n=24):
    """Polyline with each corner replaced by a quadratic Bezier of the given radius."""
    out = [pts[0]]
    for prev, corner, nxt in zip(pts[:-2], pts[1:-1], pts[2:]):
        d0, d1 = corner - prev, nxt - corner
        r = min(radius, np.linalg.norm(d0) / 2, np.linalg.norm(d1) / 2)
        p0, p2 = corner - d0 / np.linalg.norm(d0) * r, corner + d1 / np.linalg.norm(d1) * r
        t = np.linspace(0, 1, n)[:, None]
        out.extend((1 - t) ** 2 * p0 + 2 * (1 - t) * t * corner + t ** 2 * p2)
    out.append(pts[-1])
    return np.asarray(out)


def lane(ax, points, color, lw=16):
    """Thick flow band through the given points (corners rounded), ending in an arrow head."""
    # Opaque colour pre-blended with the panel band, so band and head never overlap darker.
    color = 0.85 * np.array(matplotlib.colors.to_rgb(color)) + 0.15 * np.array(matplotlib.colors.to_rgb(BAND))
    pts = rounded(np.asarray(points, dtype=float))
    unit = ax.transData.transform([(1, 0)])[0, 0] - ax.transData.transform([(0, 0)])[0, 0]
    width = lw * ax.figure.dpi / 72 / unit  # band width in data units
    tip = pts[-1] + (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2]) * 6
    d = (tip - pts[-2]) / np.linalg.norm(tip - pts[-2])
    hl, hw = max(1.1 * width, 7), max(1.9 * width, 9)
    base = tip - d * hl
    ax.plot(*np.vstack([pts[:-1], base + d * 0.3]).T, color=color, lw=lw,
            solid_capstyle='butt', solid_joinstyle='round', zorder=0)
    normal = np.array([-d[1], d[0]])
    ax.add_patch(plt.Polygon([tip, base + normal * hw / 2, base - normal * hw / 2], closed=True,
                             fc=color, ec=color, lw=0.6, joinstyle='round', zorder=0))


def box(ax, x, y, text, color=INK, fs=11.5, weight='normal', fc=SURFACE):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=color, fontweight=weight,
            linespacing=1.3, zorder=5,
            bbox=dict(boxstyle='round,pad=0.45,rounding_size=0.6', fc=fc, ec=color, lw=1.6))


def step_methods(ax):
    """Propagation (random walk with restart + hub correction) and colocalisation, on a toy network."""
    W, H = 424, 100
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    g, xy, seeds_a, seeds_b = toy_network()
    za, zb = toy_nps(g, seeds_a), toy_nps(g, seeds_b)
    coloc = np.flatnonzero((za > 1) & (zb > 1) & (za * zb >= 3))
    deg = np.array([g.degree(n) for n in g])
    sizes = 40 + 30 * deg
    ya, yb, xg, r = 75, 22, 82, 20
    # Largest drawn marker per node (colocalised x1.3, seed rings x2.2 area), in layout units.
    unit = ax.transData.transform([(1, 0)])[0, 0] - ax.transData.transform([(0, 0)])[0, 0]
    area = sizes * np.where(np.isin(np.arange(len(g)), seeds_a + seeds_b), 2.2 * 1.3, 1.3)
    xy = separate(xy, np.sqrt(area) / 2 * ax.figure.dpi / 72 / unit / r)

    def nps_cmap(color):
        return LinearSegmentedColormap.from_list('nps', [NEGATIVE, '#f0efec', color])

    norm = TwoSlopeNorm(vcenter=0, vmin=-9, vmax=2.5)
    # Lanes: seeds -> propagation (A on top, B below).
    lane(ax, [(0, ya), (50, ya)], BLUE)
    lane(ax, [(0, yb), (50, yb)], SET_B)
    ax.text(1, ya + 7, 'CNV A seeds', fontsize=12.5, fontweight='bold', color=BLUE, va='bottom')
    ax.text(1, yb + 7, 'set B seeds', fontsize=12.5, fontweight='bold', color=SET_B, va='bottom')
    ax.text(1, ya - 7, 'e.g. 22q11.2', fontsize=10.5, color=INK2, va='top')
    ax.text(1, yb - 7, 'another CNV,\nrare or common variant\ngene sets', fontsize=10.5, color=INK2, va='top')
    draw_graph(ax, g, xy, np.array([xg, ya]), r, nps_cmap(BLUE)(norm(za)), sizes, ring=seeds_a)
    draw_graph(ax, g, xy, np.array([xg, yb]), r, nps_cmap(SET_B)(norm(zb)), sizes, ring=seeds_b)
    ax.text(xg, H + 1, 'Network propagation on Brain interactome', ha='center', va='bottom', fontsize=13.5, fontweight='bold',
            color=INK)
    ax.text(xg, H - 1.5, f'random walk with restart (α = {ALPHA})', ha='center', va='top', fontsize=11,
            color=INK2)
    box(ax, xg, (ya + yb) / 2 + 0.5, f'hub correction: heat vs {NPS_REPS:,} degree-matched\nrandom seed sets → NPS (z-score)',
        fs=10.5, color=INK2)

    # Merge both profiles into the product.
    xp, yp = 190, 60
    lane(ax, [(xg + r + 4, ya), (xp - r - 10, ya - 6)], BLUE, lw=12)
    lane(ax, [(xg + r + 4, yb), (140, yb), (140, yp - 14), (xp - r - 8, yp - 6)], SET_B, lw=12)
    ax.text(137, ya + 5, 'Network\ncolocalisation', ha='center', va='bottom', fontsize=12.5,
            fontweight='bold', color=COLOC)
    prod = za * zb
    pnorm = TwoSlopeNorm(vcenter=0, vmin=-12, vmax=6)
    pcmap = LinearSegmentedColormap.from_list('prod', ['#eda100', '#f0efec', COLOC])
    draw_graph(ax, g, xy, np.array([xp, yp]), r, pcmap(pnorm(prod)), sizes)
    ax.text(xp, H + 1, 'NPS$_{A}$ × NPS$_{B}$', ha='center', va='bottom', fontsize=14, fontweight='bold',
            color=INK)

    # Thresholds -> colocalised network (drawn over a faded copy of the full network).
    xc, yc = 312, 64
    lane(ax, [(xp + r + 4, yp), (xc - r - 8, yp)], COLOC, lw=12)
    box(ax, (xp + xc) / 2, yp + 17, 'NPS$_A$ > 1 and NPS$_B$ > 1\nNPS$_A$ × NPS$_B$ ≥ 3', color=COLOC, fs=11)
    # Genes reached by propagation from one set only are tinted by that set; colocalised genes are
    # highlighted in purple with a halo; seeds keep their ring.
    reached = np.full(len(g), '#ebeae6', dtype=object)
    reached[(za > 1) & ~(zb > 1)] = '#a9cbf3'
    reached[(zb > 1) & ~(za > 1)] = '#9fd29f'
    draw_graph(ax, g, xy, np.array([xc, yc]), r, reached, sizes * 0.8, edge_color='#dcdbd6')
    pc = np.array([xc, yc]) + r * xy[coloc]
    ax.scatter(pc[:, 0], pc[:, 1], s=sizes[coloc] * 5, color=COLOC, alpha=0.18, lw=0, zorder=1.5)
    draw_graph(ax, g, xy, np.array([xc, yc]), r, [COLOC] * len(g), sizes * 1.3, nodes=coloc, edge_color=COLOC,
               ring=[n for n in coloc if n in seeds_a + seeds_b])
    for i, (label, color) in enumerate([('colocalised', COLOC), ('reached from A only', '#a9cbf3'),
                                        ('reached from B only', '#9fd29f')]):
        ax.scatter(xc + r + 7, yc - 6 - 5 * i, s=60, color=color, ec='#6f6e69', lw=0.7)
        ax.text(xc + r + 10, yc - 6 - 5 * i, label, va='center', fontsize=10, color=INK2)
    ax.text(xc, H + 1, 'Colocalised network', ha='center', va='bottom', fontsize=13.5, fontweight='bold',
            color=COLOC)
    ax.text(xc, H - 1.5, 'genes close to both seed sets', ha='center', va='top', fontsize=11, color=INK2)
    ax.text(xc + r + 5, yc + 6, f'observed size\n= {len(coloc)} genes', ha='left', va='center', fontsize=11,
            color=COLOC, fontweight='bold')

    # Null: shuffle one profile's gene labels, apply the same thresholds, count.
    lane(ax, [(140, yb), (196, yb)], '#b8b7b1', lw=5)
    box(ax, 226, yb - 6, f'gene labels shuffling of\nNPS profile ({PERM_REPS:,}×)\n→ same thresholds',
        color=INK2, fs=10.5)
    x0, x1, base = 262, 352, 4
    xs = np.linspace(x0, x1, 200)
    mu, sd = x0 + 26, 8.5
    ys = base + 22 * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
    ax.fill_between(xs, base, ys, color='#d6d5d0', lw=0, zorder=1)
    ax.plot([x0, x1], [base, base], color=INK2, lw=1.2)
    ax.plot([mu, mu], [base, base + 24], color=INK, lw=1.4, ls='--')
    obs = x0 + 72
    ax.plot([obs, obs], [base, base + 24], color=COLOC, lw=2.5)
    ax.annotate('', xy=(obs - 1, base + 12), xytext=(mu + 1, base + 12),
                arrowprops=dict(arrowstyle='<->', color=COLOC, lw=1.4))
    ax.text(mu, base + 25, 'expected', ha='center', va='bottom', fontsize=10.5, color=INK)
    ax.text(obs + 1.5, base + 25, 'observed', ha='left', va='bottom', fontsize=10.5, color=COLOC,
            fontweight='bold')
    ax.text((x0 + x1) / 2, base - 1.5, 'colocalised network size', ha='center', va='top', fontsize=10.5,
            color=INK2)
    ax.add_patch(FancyArrowPatch((xc + 10, yc - r - 1), (obs, base + 31), arrowstyle='-|>', mutation_scale=20,
                                 lw=2.2, color=COLOC, connectionstyle='arc3,rad=-0.2'))
    ax.add_patch(FancyArrowPatch((250, yb - 6), (x0 + 4, yb - 10), arrowstyle='-|>', mutation_scale=18,
                                 lw=2, color='#b8b7b1'))
    box(ax, 396, yb - 4, 'NetColoc score\n= obs / exp', color=COLOC, fs=11.5, weight='bold')
    ax.add_patch(FancyArrowPatch((x1 + 1, yb - 4), (371, yb - 4), arrowstyle='-|>', mutation_scale=18,
                                 lw=2.2, color=COLOC))


# ================================================================= results
def circle_order(nodes, edges):
    """Order nodes on a circle so the fewest edges cross (brute force; fine for <= 10 nodes)."""
    def crossings(order):
        rank = {n: i for i, n in enumerate(order)}
        chords = [tuple(sorted((rank[a], rank[b]))) for a, b in edges]
        return sum(a < c < b < d or c < a < d < b for i, (a, b) in enumerate(chords) for c, d in chords[i + 1:])

    first, rest = nodes[0], nodes[1:]
    return min(([first, *p] for p in permutations(rest)), key=crossings)


def result_pairs(ax, seeds, sig):
    cnvs = circle_order(list(seeds.query("kind == 'cnv'")['entity']), list(zip(sig['a'], sig['b'])))
    angle = {c: np.pi / 2 - 2 * np.pi * i / len(cnvs) for i, c in enumerate(cnvs)}
    pos = {c: (np.cos(a), np.sin(a)) for c, a in angle.items()}
    # NetColoc score = observed / expected colocalised network size.
    sig = sig.assign(score=sig['observed'] / sig['null_mean'])
    smin, smax = sig['score'].min(), sig['score'].max()
    for _, r in sig.sort_values('score').iterrows():
        (x0, y0), (x1, y1) = pos[r['a']], pos[r['b']]
        ax.plot([x0, x1], [y0, y1], color=EDGE_BLUE, lw=2 + 9 * (r['score'] - smin) / (smax - smin),
                solid_capstyle='round', zorder=1)
    for c, (x, y) in pos.items():
        ax.scatter(x, y, s=650, color=BLUE, ec=SURFACE, lw=2.5, zorder=3)
        ax.text(1.3 * x, 1.19 * y, c, ha='center', va='center', fontsize=13, fontweight='bold', color=INK)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.35, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(handles=[Line2D([], [], color=EDGE_BLUE, lw=2, label=f'min = {smin:.2f}'),
                       Line2D([], [], color=EDGE_BLUE, lw=11, label=f'max = {smax:.2f}')],
              title='line width = NetColoc score', title_fontsize=11,
              loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=11)


def result_landscape(ax, names):
    run = LANDSCAPE / '01_schema_gwas_landscape'
    genes = tsv(run / 'gene_modules.tsv')[['entrez', 'leiden_cluster']].merge(tsv(run / 'layout.tsv'), on='entrez')
    genes = genes.dropna(subset=['leiden_cluster'])
    genes = genes[genes['leiden_cluster'] <= 5]
    xy = genes.set_index('entrez')[['x', 'y']]
    edges = tsv(run / 'edges.tsv')
    edges = edges[edges['source'].isin(xy.index) & edges['target'].isin(xy.index)]
    segs = np.stack([xy.loc[edges['source']].to_numpy(), xy.loc[edges['target']].to_numpy()], axis=1)
    ax.add_collection(LineCollection(segs, colors='#b8b7b1', linewidths=0.3, alpha=0.18, zorder=1))
    for m, sub in genes.groupby('leiden_cluster'):
        ax.scatter(sub['x'], sub['y'], s=70, color=MODULE_COLORS[int(m)], ec=SURFACE, lw=0.5, zorder=2)
    ax.set_xlim(genes['x'].min() - 0.05, genes['x'].max() + 0.05)
    ax.set_ylim(genes['y'].min() - 0.05, genes['y'].max() + 0.05)
    ax.set_aspect('equal')
    ax.axis('off')
    handles = [Line2D([], [], marker='o', ls='', ms=10, color=MODULE_COLORS[m],
                      label=f"C{m} {short_name(names[m])} ({n})")
               for m, n in genes['leiden_cluster'].astype(int).value_counts().sort_index().items()]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=1, frameon=False,
              fontsize=11, handletextpad=0.1, labelspacing=0.3)


def landscape_distances():
    """Shortest path (steps) from each landscape gene to the nearest seed gene of every CNV / SCZ set."""
    spec, root, inputs, provenance = load_inputs(NETWORK_CONFIG)
    A, nodes, symbols, entities, coordinates, regions = inputs
    index = {g: i for i, g in enumerate(nodes)}
    genes = tsv(LANDSCAPE / '01_schema_gwas_landscape/gene_modules.tsv')['entrez']
    cols = np.array([index[g] for g in genes])
    sets = [e for e, v in entities.items() if v['kind'] == 'cnv' or e in SET_LABELS]
    rows = []
    for name in sets:
        seeds = [index[g] for g in entities[name]['seeds'] if g in index]
        d = shortest_path(A, unweighted=True, directed=False, indices=seeds)[:, cols].min(axis=0)
        counts = pd.Series(np.minimum(d, 3).astype(int)).value_counts()
        rows.append({'set': name, 'seeds_in_network': len(seeds), 'landscape_genes': len(cols),
                     **{f'steps_{k}' if k < 3 else 'steps_3plus': int(counts.get(k, 0)) for k in range(4)},
                     'mean_steps': float(d.mean())})
    return pd.DataFrame(rows)


def result_reach(fig, ax, bar_ax, dist):
    run = LANDSCAPE / 'figures/cnv_module_roads'
    prof = tsv(run / 'cnv_module_profile.tsv')
    summary = tsv(run / 'cnv_summary.tsv').sort_values('reached_landscape_genes', ascending=False)
    modules = [f'C{m}' for m in range(1, 6)]
    share = prof.pivot(index='cnv', columns='module', values='share_of_cnv_reach').loc[summary['cnv'], modules]
    n = len(summary)
    targets = ['schema', 'gwaspri']
    ypos = list(range(n)) + [n + 0.6 + i for i in range(len(targets))]
    labels = [f"{c}  ({k})" for c, k in zip(summary['cnv'], summary['reached_landscape_genes'])]
    labels += [SET_LABELS[t] for t in targets]

    # Left: share of each CNV's reached landscape genes per module.
    im = ax.pcolormesh(np.arange(len(modules) + 1) - 0.5, np.arange(n + 1) - 0.5, share.to_numpy() * 100,
                       cmap=BLUES, vmin=0, vmax=70, edgecolors=SURFACE, linewidth=2)
    for i in range(n):
        for j in range(len(modules)):
            v = share.iat[i, j] * 100
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=11,
                    color='white' if v >= 40 else INK, fontweight='bold' if v >= 40 else 'normal')
    ax.set_ylim(ypos[-1] + 0.5, -0.5)
    ax.set_xlim(-0.5, len(modules) - 0.5)
    ax.patch.set_alpha(0)
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules, fontsize=13, fontweight='bold')
    ax.xaxis.tick_top()
    for j, lab in enumerate(ax.get_xticklabels()):
        lab.set_color(MODULE_COLORS[j + 1])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=12)
    for lab in ax.get_yticklabels()[n:]:
        lab.set_color(ORANGE)
        lab.set_fontweight('bold')
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.text(0.5, -0.5 - 0.9, 'modules reached', ha='center', va='bottom', fontsize=12, color=INK2,
            transform=ax.transData, fontweight='bold')
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + 0.01, box.y0 - 0.025, box.width - 0.02, 0.008])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_label("% of the CNV's reached genes (n)", fontsize=10.5)
    cb.ax.tick_params(labelsize=10)

    # Right: shortest path from the landscape genes to the nearest seed of each set.
    dist = dist.set_index('set').loc[list(summary['cnv']) + targets]
    steps = ['steps_0', 'steps_1', 'steps_2', 'steps_3plus']
    frac = dist[steps].div(dist['landscape_genes'], axis=0) * 100
    for y, (name, row) in zip(ypos, frac.iterrows()):
        left = 0
        for k, col in enumerate(steps):
            v = row[col]
            bar_ax.barh(y, v, left=left, height=0.72, color=DIST_COLORS[k], ec=SURFACE, lw=1.5)
            if v >= 9:
                bar_ax.text(left + v / 2, y, f'{v:.0f}%', ha='center', va='center', fontsize=10.5,
                            color='white' if k < 2 else INK)
            left += v
    bar_ax.set_ylim(ypos[-1] + 0.5, -0.5)
    bar_ax.set_xlim(0, 100)
    bar_ax.set_yticks([])
    bar_ax.set_xticks([0, 50, 100])
    bar_ax.set_xticklabels(['0', '50', '100%'])
    bar_ax.xaxis.tick_top()
    bar_ax.tick_params(labelsize=10.5, length=0)
    for side in bar_ax.spines.values():
        side.set_visible(False)
    bar_ax.text(50, -0.5 - 0.9, 'shortest path to seeds', ha='center', va='bottom', fontsize=12,
                color=INK2, fontweight='bold')
    bar_ax.legend(handles=[Line2D([], [], marker='s', ls='', ms=11, color=DIST_COLORS[k], label=DIST_LABELS[k])
                           for k in range(4)],
                  title=f'{int(dist["landscape_genes"].iloc[0])} landscape genes', title_fontsize=10.5,
                  loc='upper center', bbox_to_anchor=(0.45, -0.01), ncol=2, frameon=False, fontsize=10.5,
                  handletextpad=0.1, columnspacing=0.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=OUT, help='output path without extension')
    args = parser.parse_args()

    seeds = tsv(SEEDS)
    sig = tsv(SIG_PAIRS)
    tests = tsv(PAIR_TESTS)
    n_pairs = int(((tests['family'] == 'region_pair') & tests['primary']).sum())
    n_cnv = int((seeds['kind'] == 'cnv').sum())
    names = tsv(LANDSCAPE / '01_schema_gwas_landscape/module_names.tsv').set_index('leiden_cluster')['biological_process']
    n_landscape = len(tsv(LANDSCAPE / '01_schema_gwas_landscape/gene_modules.tsv'))
    n_edges = len(tsv(LANDSCAPE / '01_schema_gwas_landscape/edges.tsv'))

    fig = plt.figure(figsize=(28, 17), facecolor=SURFACE)
    fig.text(0.02, 0.972, 'Do schizophrenia-linked CNVs converge on the same biology?',
             fontsize=30, fontweight='bold', color=INK, va='center')
    fig.text(0.02, 0.945, 'Network propagation of CNV and schizophrenia genes on the PCNet 2.2 interactome, '
             'restricted to brain-expressed genes', fontsize=16, color=INK2, va='center')

    # ---- workflow band
    band(fig, 0.535, 0.92, '')
    hy = 0.885                               # header line
    ax1 = fig.add_axes([0.14, 0.6, 0.11, 0.215])
    ax2 = fig.add_axes([0.285, 0.555, 0.695, 0.265])
    step_seeds(ax1, seeds)
    step_methods(ax2)
    header(fig, 0.058, hy, '1', 'Seed genes',
           f'genes inside {n_cnv} high-risk schizophrenia CNVs; SCHEMA\n(rare variants) and GWAS (common variants) genes')
    header(fig, 0.3, hy, '2', 'Network propagation and colocalisation',
           'NetColoc Python package (Rosenthal et al., Nature Protocols 2023); toy network with illustrative scores')

    # ---- results band
    band(fig, 0.02, 0.515, '')
    ry, rh = 0.11, 0.29
    ry_head = 0.48
    rcols = [0.05, 0.28, 0.62]
    axB = fig.add_axes([rcols[0] - 0.01, ry + 0.04, 0.22, rh - 0.03])
    axC = fig.add_axes([rcols[1] - 0.005, 0.135, 0.34, 0.29])
    axD = fig.add_axes([rcols[2] + 0.065, ry + 0.04, 0.12, rh - 0.06])
    axD2 = fig.add_axes([rcols[2] + 0.195, ry + 0.04, 0.15, rh - 0.06])

    result_pairs(axB, seeds, sig)
    result_landscape(axC, names)
    dist = landscape_distances()
    result_reach(fig, axD, axD2, dist)

    header(fig, rcols[0] + 0.008, ry_head, 'A', 'CNV × CNV pairwise comparison',
           f'{len(sig)} of {n_pairs} CNV pairs colocalise more than\nexpected by chance (FDR < 5%)', square=True)
    header(fig, rcols[1] + 0.008, ry_head, 'B', 'Schizophrenia landscape of rare and common variants',
           f'Colocalised network size:\n{n_landscape} nodes and {n_edges:,} edges', square=True)
    header(fig, rcols[2] + 0.008, ry_head, 'C', 'CNVs inside the landscape',
           'each CNV reaches the landscape through different modules (left);\n'
           'landscape genes sit 1–2 steps from the seeds of every set (right)', square=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dist.to_csv(args.out.parent / f'{args.out.name}_shortest_paths.tsv', sep='\t', index=False)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(args.out.with_suffix(f'.{ext}'), dpi=300 if ext == 'png' else None, facecolor=SURFACE)
    print('\n'.join(str(args.out.with_suffix(f'.{e}')) for e in ('png', 'pdf', 'svg')))


if __name__ == '__main__':
    main()
