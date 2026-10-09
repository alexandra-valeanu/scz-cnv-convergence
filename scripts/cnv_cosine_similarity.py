"""Descriptive cosine comparisons, including the pair-union method in Wright et al.

The paper (10.1038/s44319-026-00733-4, Fig. 6B and Methods p. 1934)
compares joint NPS products on each pair's union of selected network genes.
Here CNV x SCHEMA and CNV x GWAS products follow that construction. Single-CNV
NPS comparisons are explicitly adaptations, not common-rare trait networks.
Scores at union genes are retained even outside an individual selected network;
no zero filling, clipping, centering, or absolute values are applied.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def cosine_matrix(profiles, selections=None):
    """Pair-specific unions; empty union/zero norms yield NaN, never false similarity."""
    names = sorted(profiles)
    arrays = {n: np.asarray(profiles[n], dtype=float) for n in names}
    if not arrays or any(v.ndim != 1 or v.shape != arrays[names[0]].shape
                         or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Require aligned, finite, one-dimensional profiles')
    masks = ({n: np.ones(arrays[n].size, dtype=bool) for n in names}
             if selections is None else {n: np.asarray(selections[n], dtype=bool) for n in names})
    if any(masks[n].shape != arrays[n].shape for n in names):
        raise ValueError('Selection masks must align with profiles')
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    rows = []
    for i, a in enumerate(names):
        for b in names[i:]:
            union = masks[a] | masks[b]
            x, y = arrays[a][union], arrays[b][union]
            norm = np.linalg.norm(x)*np.linalg.norm(y)
            score = float(np.clip(np.dot(x, y)/norm, -1, 1)) if norm > 0 else np.nan
            matrix.loc[a, b] = matrix.loc[b, a] = score
            rows.append({'a': a, 'b': b, 'cosine_similarity': score,
                         'union_genes': int(union.sum()), 'selected_a': int(masks[a].sum()),
                         'selected_b': int(masks[b].sum()),
                         'status': 'ok' if norm > 0 else 'empty_union_or_zero_norm'})
    matrix.index.name = 'cnv'
    return matrix, pd.DataFrame(rows)


def plot_cosine(matrix, stem, title, subtitle):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#dddddd')
    im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(matrix)), matrix.index)
    for i, row in enumerate(matrix.to_numpy()):
        for j, value in enumerate(row):
            ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else 'NA', ha='center', va='center',
                    fontsize=9, color='white' if abs(value) > .65 else '#222222')
    fig.colorbar(im, ax=ax, shrink=.7, label='Cosine similarity (descriptive)')
    ax.set_title(f'{title}\n{subtitle}', fontsize=12, pad=18)
    for suffix in ('png', 'pdf'):
        fig.savefig(f'{stem}.{suffix}', dpi=180)
    plt.close(fig)


def export_cosines(profiles, cnv_names, pair_masks, landscape, nodes, threshold, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cnv = {n: profiles[n] for n in cnv_names}
    definitions = {
        'cnv_nps_pair_union': (cnv, {n: profiles[n] > threshold for n in cnv_names},
            'CNV propagation profile similarity', f'Pairwise union of CNV neighbourhoods (NPS > {threshold:g}); adapted method'),
        'cnv_nps_full_network': (cnv, None, 'CNV propagation profile similarity',
            'All tested interactome genes; sensitivity analysis'),
        'cnv_nps_scz_landscape': (cnv, {n: landscape for n in cnv_names},
            'CNV propagation similarity within the schizophrenia landscape',
            'Fixed SCHEMA × GWAS gene universe'),
    }
    for target, label in [('schema', 'SCHEMA'), ('gwaspri', 'GWAS')]:
        definitions[f'cnv_{target}_joint_pair_union'] = (
            {n: profiles[n]*profiles[target] for n in cnv_names},
            {n: pair_masks[n, target] for n in cnv_names},
            f'CNV × {label} subnetwork similarity',
            f'Joint NPS products on each pair’s union; paper-style construction')
    rows = []
    for name, (values, masks, title, subtitle) in definitions.items():
        matrix, pairs = cosine_matrix(values, masks)
        matrix.to_csv(output/f'{name}.tsv', sep='\t')
        pairs.to_csv(output/f'{name}_pairs.tsv', sep='\t', index=False)
        # Full scores and selection flags make every pair-specific union reproducible.
        table = pd.DataFrame({'entrez': nodes})
        for n in cnv_names:
            table[f'score_{n}'] = values[n]
            table[f'selected_{n}'] = True if masks is None else masks[n]
        table.to_csv(output/f'{name}_profiles.tsv', sep='\t', index=False)
        plot_cosine(matrix, output/name, title, subtitle)
        rows.append({'analysis': name, 'title': title, 'method': subtitle,
                     'matrix': f'{name}.tsv', 'figure': f'{name}.pdf'})
    pd.DataFrame(rows).to_csv(output/'index.tsv', sep='\t', index=False)
    (output/'README.md').write_text(
        '# CNV cosine similarities\n\n'
        'Reference: Wright et al., DOI 10.1038/s44319-026-00733-4, Fig. 6B and Methods '
        '“Similarity of trait-specific networks” (p. 1934). Joint NPS is a product, not a square root. '
        'The union is recomputed for each pair, not pooled over all CNVs.\n\n'
        'The two joint-product panels implement this construction for CNV × SCHEMA and CNV × GWAS. '
        'Single-CNV NPS panels are adaptations; the primary neighbourhood union uses the configured '
        'individual NPS threshold. Full-interactome and fixed-landscape panels show sensitivity to the universe.\n\n'
        'Both profiles retain their actual signed scores at every union gene, even when that gene is '
        'outside one selected subnetwork. The paper does not state a zero-fill or clipping step. '
        'Negative scores are retained; matching negative scores can contribute to similarity. '
        'The joint panels also share a target profile, which can itself increase similarity.\n\n'
        'All panels use alphabetical CNV order and a fixed −1 to 1 scale. No similarity cutoff or '
        'significance is inferred. Fig. 6B’s 0.05 edge-display and 0.10 edge-label cutoffs are graphical '
        'choices, not hypothesis tests, and are unnecessary for these complete heatmaps. '
        'Pair-dependent universes do not define a common embedding; no Ward clustering is applied. '
        'Empty unions and zero norms are reported as NA.\n')
