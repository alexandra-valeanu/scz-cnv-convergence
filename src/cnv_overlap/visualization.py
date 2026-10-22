"""Plot pair-test results and descriptive gene rankings."""
from cnv_overlap.results import resolve_path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_CATEGORIES = {
    "schema": "Schizophrenia", "gwaspri": "Schizophrenia",
    "t2d": "Metabolic", "obesity": "Metabolic",
    "cad": "Cardiovascular",
    "crohn": "Immune", "ra": "Immune", "t1d": "Immune", "thyroid": "Immune",
    "crc": "Cancer", "lung": "Cancer",
}
CATEGORY_ORDER = ["Schizophrenia", "Metabolic", "Cardiovascular", "Immune", "Cancer"]
DISPLAY_NAMES = {"schema": "SCHEMA", "gwaspri": "GWAS"}


def display_name(label):
    """Figure label for a gene set; unknown names are shown as given."""
    return DISPLAY_NAMES.get(label, label)


def _category_order(labels):
    """Return labels grouped by disorder category, or None if any label is unknown."""
    if not all(label in TARGET_CATEGORIES for label in labels):
        return None
    ordered = [label for category in CATEGORY_ORDER
               for label in sorted(labels) if TARGET_CATEGORIES[label] == category]
    return ordered


def _annotate_categories(ax, labels, n_rows):
    """Draw category separators and headers above the grouped columns."""
    categories = [TARGET_CATEGORIES[label] for label in labels]
    start = 0
    for index in range(len(labels) + 1):
        if index == len(labels) or categories[index] != categories[start]:
            centre = (start + index - 1)/2
            if start:
                ax.axvline(start - .5, color="white", linewidth=3, clip_on=False, zorder=3)
            ax.plot([start - .45, index - .55], [-.62, -.62], color="0.25", linewidth=1.2,
                    clip_on=False, zorder=3)
            ax.text(centre, -.72, categories[start], ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", clip_on=False)
            start = index
    ax.set_ylim(n_rows - .5, -1.4)


def pair_heatmap(run_dir, null="netcoloc", family="locus_target", fdr=.05, metric="ratio", exclude=()):
    """Pair heatmap; ``ratio`` is NetColoc's COLOC score (observed / expected network size).

    ``z`` colours cells by the size z-score against the null and labels them with the
    observed number of colocalised genes. ``ratio_genes`` colours cells by the COLOC score
    and labels them with the number of colocalised network genes.
    ``exclude`` drops gene sets from the columns.
    """
    if metric not in ("ratio", "ratio_genes", "excess", "observed", "z"):
        raise ValueError("Heatmap metric must be ratio, ratio_genes, excess, observed or z")
    with_genes = metric == "ratio_genes"
    if with_genes:
        metric = "ratio"
    d = pd.read_csv(resolve_path(Path(run_dir)/"pair_tests.tsv"), sep="\t")
    d = d[(d["null"] == null)&(d.family == family)&(d.statistic == "size")&
          ~d.a.isin(exclude)&~d.b.isin(exclude)].copy()
    if d.empty:
        raise ValueError(f"No {null} / {family} primary tests in this run")
    if "size_ratio" not in d:  # runs written before the ratio column existed
        d["size_ratio"] = d.observed / d.null_mean.where(d.null_mean > 0)
    if family == "region_pair":
        d = pd.concat([d, d.rename(columns={"a": "b", "b": "a"})], ignore_index=True)
    effect = d.pivot(index="a", columns="b", values="size_ratio" if metric == "ratio" else metric)
    grouped = _category_order(list(effect.columns))
    if grouped is not None:
        effect = effect[grouped]
    q = d.pivot(index="a", columns="b", values="q").reindex_like(effect)
    text = d.pivot(index="a", columns="b", values="observed" if metric == "z" or with_genes else
                   "size_ratio" if metric == "ratio" else metric).reindex_like(effect)
    # Ratios are coloured on a log2 scale so that 2x and 0.5x are equally saturated around 1.
    shown = np.log2(effect.where(effect > 0)) if metric == "ratio" else effect
    column_width = 1.45 if grouped is not None else .7
    fig, ax = plt.subplots(figsize=(max(7, column_width*len(effect.columns)), max(4, .45*len(effect))))
    finite = np.abs(shown.to_numpy(dtype=float))
    limit = max(float(finite[np.isfinite(finite)].max(initial=0.)), 1.)
    diverging = metric in ("ratio", "excess", "z")
    cmap = plt.get_cmap("RdBu_r" if diverging else "viridis").with_extremes(bad="#eeeeee")
    im = ax.imshow(shown, cmap=cmap, vmin=-limit if diverging else 0, vmax=limit, aspect="auto")
    fmt = "{:.2f}" if metric == "ratio" and not with_genes else "{:.0f}"
    for i in range(len(effect)):
        for j in range(len(effect.columns)):
            if np.isfinite(shown.iloc[i, j]):
                red, green, blue, _ = im.cmap(im.norm(shown.iloc[i, j]))
                color = 'white' if .2126*red + .7152*green + .0722*blue < .5 else 'black'
                ax.text(j, i, fmt.format(text.iloc[i,j])+("*" if q.iloc[i,j]<fdr else ""), ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(effect.columns)), [display_name(c) for c in effect.columns], rotation=50, ha="right")
    ax.set_yticks(range(len(effect)), [display_name(r) for r in effect.index])
    title, label = {"ratio": ("Network colocalization (COLOC score)" +
                              ("\nnumbers: colocalised network genes" if with_genes else ""),
                              "Observed / expected network size"),
                    "excess": ("Network overlap beyond chance", "Observed − expected shared genes"),
                    "observed": ("Observed network overlap", "Shared network genes"),
                    "z": ("Network overlap beyond chance (numbers: colocalised genes)",
                          "Size z-score vs null")}[metric]
    if grouped is not None:
        _annotate_categories(ax, grouped, len(effect))
    ax.set_title(f"{title}\n* FDR < {fdr}" + ("" if null == "netcoloc" else f" ({null} null)"), pad=22)
    bar = fig.colorbar(im, ax=ax, label=label)
    if metric == "ratio":
        ticks = np.arange(-np.floor(limit), np.floor(limit)+1)
        bar.set_ticks(ticks, labels=[f"{2**t:g}" for t in ticks])
    fig.tight_layout()
    return fig


def enrichment_dotplot(annotation, fdr=.05, max_terms=20, max_pairs=20):
    """Display a bounded subset; correction always comes from the full test table."""
    d = annotation[annotation.q_all_pair_terms < fdr].copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(.5, .5, "No pathways pass the across-pair-term FDR threshold", ha="center", va="center")
        ax.axis("off")
        return fig
    d["pair"] = d.a.map(display_name) + " × " + d.b.map(display_name)
    terms = d.groupby('term').q_all_pair_terms.min().nsmallest(max_terms).index
    pairs = d.groupby('pair').q_all_pair_terms.min().nsmallest(max_pairs).index
    d = d[d.term.isin(terms) & d.pair.isin(pairs)]
    fig, ax = plt.subplots(figsize=(max(8, len(pairs)*.6), max(4, len(terms)*.35)))
    points = ax.scatter(d.pair.map({p: i for i, p in enumerate(pairs)}),
        d.term.map({t: i for i, t in enumerate(terms)}),
        s=25+12*np.sqrt(d.overlap), c=-np.log10(d.q_all_pair_terms.clip(lower=1e-300)), cmap='viridis')
    ax.set_xticks(range(len(pairs)), pairs, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(terms)), terms, fontsize=8)
    ax.invert_yaxis()
    sizes = np.unique(np.quantile(d.overlap, [0, .5, 1]).round().astype(int))
    ax.legend([ax.scatter([], [], s=25+12*np.sqrt(k), color="0.5") for k in sizes], sizes,
              title="Shared genes", loc="upper left", bbox_to_anchor=(1.25, 1), frameon=False)
    ax.set_title(f'Pathway enrichment of shared network genes\nFDR < {fdr}')
    fig.colorbar(points, ax=ax, label='−log10 FDR')
    fig.tight_layout()
    return fig
