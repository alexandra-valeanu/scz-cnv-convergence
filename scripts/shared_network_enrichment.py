"""Offline descriptive enrichment of the shared CNV–SCHEMA–GWAS core."""
import json
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from cnv_overlap.enrichment import load_gmt, ora
from cnv_overlap.provenance import file_hash, run_dir, write_json
from cnv_overlap.statistics import bh

LIBRARIES = {
    'Reactome pathways': 'ReactomePathways.gmt',
    'GO Cellular Component': 'GO_Cellular_Component_2026.gmt',
    'GO Biological Process': 'GO_Biological_Process_2026.gmt',
    'GO Molecular Function': 'GO_Molecular_Function_2026.gmt',
}


def enrich_shared_network(network_dir, *, annotation_dir, output_root, min_size=5, fdr=.05):
    """One hypergeometric tail per eligible term; BH over all configured libraries.

    The background is the network universe eligible for the same shared-core
    selection, including unannotated genes. All eligible terms, including those
    with zero hits, enter correction. This does not calibrate network selection.
    """
    if min_size < 1 or not 0 < fdr < 1:
        raise ValueError('Require min_size >= 1 and 0 < fdr < 1')
    network_dir = Path(network_dir)
    manifest = json.loads((network_dir/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Shared-network export must be complete')
    for name, checksum in manifest['artifacts'].items():
        if not (network_dir/name).exists() or file_hash(network_dir/name) != checksum:
            raise ValueError(f'Network artifact changed: {name}')
    membership = pd.read_csv(network_dir/'gene_membership.tsv', sep='\t')
    if 'shared_core' not in membership:
        raise ValueError('Shared core requires both SCHEMA and GWAS targets')
    paths = {name: Path(annotation_dir)/file for name, file in LIBRARIES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing annotation snapshots: ' + ', '.join(missing))
    identity = {'network_manifest_sha256': file_hash(network_dir/'manifest.json'),
                'code': file_hash(__file__), 'gmt': {n: file_hash(p) for n, p in paths.items()},
                'min_size': min_size, 'fdr': fdr}
    out = run_dir(output_root, identity, 'enrichment', f'fdr{fdr}', f'min{min_size}')
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        if old.get('status') == 'complete':
            for name, checksum in old['artifacts'].items():
                if not (out/name).exists() or file_hash(out/name) != checksum:
                    raise ValueError(f'Enrichment artifact changed: {name}')
            return out
    eligible = np.ones(len(membership), dtype=bool)
    network_identity = manifest['identity']
    if network_identity['overlap'] == 'remove':
        cnvs = [n for n, e in network_identity['entities'].items() if e['kind'] == 'cnv']
        counts = np.zeros(len(membership), dtype=int)
        for name in cnvs:
            counts += ~(membership[f'seed_{name}'] & (membership.seed_schema | membership.seed_gwaspri))
        eligible = counts >= network_identity['min_cnvs']
    background = set(membership.loc[eligible, 'symbol'].dropna().astype(str))
    query_rows = membership.loc[membership.shared_core].copy()
    query = set(query_rows.symbol.dropna().astype(str))
    if not query <= background:
        raise ValueError('Core query includes ineligible background genes')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', {'status': 'running', 'identity': identity})
    frames, coverage = [], []
    for library, path in paths.items():
        terms = load_gmt(path)
        covered = set().union(*(genes for _, genes in terms.values())) if terms else set()
        result = ora(query, background, terms, min_size=min_size).rename(columns={'q': 'q_within_library'})
        result.insert(0, 'library', library)
        frames.append(result)
        coverage.append({'library': library, 'query_genes': len(query), 'query_annotated': len(query & covered),
                         'query_unannotated': ','.join(sorted(query-covered)), 'background_genes': len(background),
                         'background_annotated': len(background & covered), 'eligible_terms': len(result)})
    results = pd.concat(frames, ignore_index=True)
    results['q_all_libraries'] = bh(results.p)
    results = results.sort_values(['q_all_libraries', 'p', 'library', 'term'])
    results.to_csv(out/'enrichment_all_terms.tsv', sep='\t', index=False)
    results.loc[results.q_all_libraries < fdr].to_csv(out/'enrichment_significant.tsv', sep='\t', index=False)
    pd.DataFrame(coverage).to_csv(out/'annotation_coverage.tsv', sep='\t', index=False)
    query_rows.to_csv(out/'query_genes.tsv', sep='\t', index=False)
    pd.DataFrame({'symbol': sorted(background)}).to_csv(out/'background_genes.tsv', sep='\t', index=False)
    plot_enrichment(results, out, fdr=fdr)
    lines = ['# Shared-network enrichment', '',
             f'{len(query)} unique query symbols; {len(background)} eligible network background symbols.', '',
             f'One-sided hypergeometric tests; BH correction across {len(results)} eligible terms from all configured libraries.',
             f'Terms with fewer than {min_size} background genes are excluded; zero-hit terms remain in correction.', '',
             'Descriptive ORA after network selection; the q-values do not account for selecting genes through propagation.', '']
    for library in LIBRARIES:
        selected = results[(results.library == library) & (results.q_all_libraries < fdr)]
        lines += [f'## {library}', '', f'{len(selected)} terms with combined BH q < {fdr}.', '']
        lines += [f'- {r.term}: {r.overlap}/{r.query_size} genes, q={r.q_all_libraries:.3g}; {r.genes}'
                  for r in selected.head(10).itertuples()]
        lines += ['']
    (out/'summary.md').write_text('\n'.join(lines))
    write_json(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'network_dir': str(network_dir.resolve()), 'libraries': {n: str(p.resolve()) for n, p in paths.items()},
        'background_policy': 'unique eligible network symbols; for remove require eligibility in both masks for min_cnvs CNVs',
        'interpretation': 'descriptive ORA; BH over all eligible terms in all libraries; not adjusted for network selection',
        'artifacts': {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}})
    return out


def plot_enrichment(results, out, *, fdr=.05, top=12, libraries=None, prefix="enrichment_summary", collapsed=False):
    """Horizontal bars: combined-FDR evidence, with hit counts and fold enrichment."""
    import matplotlib.pyplot as plt
    import re

    libraries = list(libraries or LIBRARIES)
    selections = []
    totals = []
    for library in libraries:
        passing = results[(results.library == library) & (results.q_all_libraries < fdr)]
        totals.append(len(passing))
        selections.append(passing.sort_values(['q_all_libraries', 'p', 'term']).head(top).copy())
    columns = 2 if len(libraries) > 1 else 1
    rows = (len(libraries) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(22 if columns == 2 else 12, 7*rows),
                             constrained_layout=True, squeeze=False)
    colors = {'Reactome pathways': '#4477AA', 'GO Cellular Component': '#228877',
              'GO Biological Process': '#AA6677', 'GO Molecular Function': '#997700'}
    max_score = max([1.] + [(-np.log10(d.q_all_libraries.clip(lower=1e-300))).max()
                            for d in selections if not d.empty])
    for library, selected, total, ax in zip(libraries, selections, totals, axes.ravel()):
        ax.set_title(f'{library}\n{len(selected)} of {total} ' +
                     ('representatives' if collapsed else 'significant terms'),
                     loc='left', fontsize=13, fontweight='bold', pad=12)
        if selected.empty:
            ax.text(.5, .5, 'No terms pass the combined FDR threshold', ha='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        y = np.arange(len(selected))
        scores = -np.log10(selected.q_all_libraries.clip(lower=1e-300))
        ax.barh(y, scores, height=.65, color=colors.get(library, '#4477AA'), zorder=3)
        # Stable term IDs remain in the complete tables; use readable names here.
        labels = [textwrap.fill(re.sub(r'\s*\(GO:\d+\)$', '', t), 40) for t in selected.term]
        ax.set_yticks(y, labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('−log₁₀(combined BH q-value)', fontsize=11)
        ax.axvline(-np.log10(fdr), color='#888888', linestyle=':', linewidth=1, zorder=2)
        ax.grid(axis='x', color='#dddddd', linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#bbbbbb')
        ax.tick_params(axis='y', length=0, pad=7)
        for score, yi, record in zip(scores, y, selected.itertuples()):
            ax.text(score + max_score*.015, yi,
                    f'{record.overlap}/{record.query_size} genes · {record.fold_enrichment:.1f}×',
                    va='center', fontsize=9, color='#333333')
        ax.set_xlim(0, max_score*1.55)
        ax.set_ylim(len(selected)-.4, -.6)
    for ax in axes.ravel()[len(libraries):]:
        ax.axis('off')
    fig.suptitle('Shared CNV × SCHEMA × GWAS network' +
        (' — orsum representatives' if collapsed else ' — enrichment') +
        f'\nLonger bars = stronger statistical evidence; labels = hit genes and fold enrichment; dotted line: q = {fdr}' +
        '\nDescriptive analysis after network selection; BH correction across all eligible terms', fontsize=14)
    for suffix in ('png', 'pdf'):
        fig.savefig(Path(out)/f'{prefix}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    pd.concat(selections, ignore_index=True).to_csv(Path(out)/'plotted_terms.tsv', sep='\t', index=False)


def plot_gene_term_heatmap(representatives, query, out):
    """Display representative-term membership and evidence without new inference."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    import re

    libraries = ['Reactome pathways', 'GO Cellular Component', 'GO Biological Process']
    ordered = pd.concat([representatives[representatives.library == name].sort_values(
        ['q_all_libraries', 'p', 'term']) for name in libraries], ignore_index=True)
    genes = sorted(set(query.symbol.dropna().astype(str)))
    memberships = [set(str(s).split(',')) if pd.notna(s) else set() for s in ordered.genes]
    matrix = np.asarray([[gene in members for gene in genes] for members in memberships], dtype=int).reshape(len(ordered), len(genes))
    full_table = ordered[['library', 'term', 'q_all_libraries']].copy()
    for j, gene in enumerate(genes):
        full_table[gene] = matrix[:, j]
    full_table.to_csv(Path(out)/'gene_term_membership.tsv', sep='\t', index=False)
    active = np.flatnonzero(matrix.any(axis=0))
    if len(active) > 1:
        distances = pdist(matrix[:, active].T, metric='jaccard')
        active = active[leaves_list(linkage(distances, method='average', optimal_ordering=True))]
    fig = plt.figure(figsize=(max(15, len(active)*.34 + 9), max(5, len(ordered)*.45 + 2)))
    if not len(ordered) or not len(active):
        ax = fig.add_subplot()
        ax.text(.5, .5, 'No representative term memberships to display', ha='center')
        ax.axis('off')
    else:
        grid = fig.add_gridspec(1, 2, width_ratios=[max(4, len(active)), 6], wspace=.12)
        ax = fig.add_subplot(grid[0])
        evidence = fig.add_subplot(grid[1], sharey=ax)
        colors = ['#f5f5f5', '#4477AA', '#228877', '#AA6677']
        categories = np.array([libraries.index(n)+1 for n in ordered.library])
        ax.imshow(matrix[:, active]*categories[:, None], cmap=ListedColormap(colors),
                  vmin=0, vmax=3, aspect='auto', interpolation='nearest')
        names = [textwrap.fill(re.sub(r'\s*\(GO:\d+\)$', '', t), 53) for t in ordered.term]
        ax.set_yticks(range(len(ordered)), names, fontsize=9)
        seed_genes = set(query.loc[query.is_any_seed, 'symbol']) if 'is_any_seed' in query else set()
        shown_genes = [genes[i] for i in active]
        ax.set_xticks(range(len(active)), [g + (' *' if g in seed_genes else '') for g in shown_genes], rotation=90, fontsize=9)
        ax.set_xticks(np.arange(-.5, len(active), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(ordered), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.7)
        ax.tick_params(which='both', length=0)
        for tick, gene in zip(ax.get_xticklabels(), shown_genes):
            if gene in seed_genes:
                tick.set_fontweight('bold')
        scores = -np.log10(ordered.q_all_libraries.clip(lower=1e-300))
        evidence.barh(range(len(ordered)), scores, color=[colors[i] for i in categories], height=.72)
        evidence.tick_params(axis='y', left=False, labelleft=False)
        evidence.set_xlabel('−log₁₀(BH q)', fontsize=9)
        evidence.set_xlim(0, max(1, scores.max())*1.12)
        evidence.grid(axis='x', alpha=.2)
        evidence.set_axisbelow(True)
        for spine in ('top', 'right', 'left'):
            evidence.spines[spine].set_visible(False)
        boundaries = np.flatnonzero(categories[1:] != categories[:-1])+.5
        for boundary in boundaries:
            ax.axhline(boundary, color='#555555', linewidth=1)
            evidence.axhline(boundary, color='#555555', linewidth=1)
        ax.set_xlabel(f'{len(active)} of {len(genes)} core genes occur in these terms; * = input seed\nColumns grouped by similarity of term memberships', fontsize=10)
        fig.legend(handles=[Patch(facecolor=colors[i+1], label=name) for i,name in enumerate(libraries)],
                   loc='lower center', ncol=3, frameon=False, fontsize=10)
        fig.subplots_adjust(left=.40, right=.97, top=.92, bottom=.15)
    fig.suptitle('Which genes support the enriched biology?\nOrsum representatives · filled cell = annotated query gene · bars = original enrichment evidence', fontsize=14)
    for suffix in ('png', 'pdf'):
        fig.savefig(Path(out)/f'gene_term_heatmap.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
