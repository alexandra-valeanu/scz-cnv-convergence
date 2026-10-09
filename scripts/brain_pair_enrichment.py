"""Brain-focused descriptive ORA for NetColoc size pairs."""
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from cnv_overlap.enrichment import load_gmt, ora
from cnv_overlap.provenance import file_hash, run_dir, write_json
from cnv_overlap.statistics import bh

LIBRARIES = {
    'GO Biological Process': 'GO_Biological_Process_2026.gmt',
    'SynGO synaptic function': 'SynGO_2024.gmt',
    'Reactome pathways': 'ReactomePathways.gmt',  # official June 2026 release; not on Enrichr
}
# First three categorical slots of the validated reference palette (all-pairs safe).
LIBRARY_COLORS = dict(zip(LIBRARIES, ('#2a78d6', '#1baf7a', '#eb6834')))
LIBRARY_SHORT = dict(zip(LIBRARIES, ('GO:BP', 'SynGO', 'Reactome')))
TARGET_LABELS = {'schema': 'SCHEMA', 'gwaspri': 'GWAS'}
FIGURES = {'CNV × SCHEMA': 'enrichment_schema', 'CNV × GWAS': 'enrichment_gwas',
           'CNV × CNV': 'enrichment_cnv_cnv'}


def locus_key(name):
    """Chromosome order for names like 15q11.2-q13.1; others sort last by name."""
    match = re.match(r'(\d+)([pq])([\d.]+)', name)
    return (int(match[1]), match[2], float(match[3].rstrip('.')), name) if match else (999, '', 0., name)


def enrich_pairs(source, *, annotation_dir, output_root, fdr=.05, min_size=5, backend="enrichr",
                 collapse=True, orsum_executable='orsum.py'):
    """Test every CNV × SCHEMA/GWAS pair and only significant CNV × CNV pairs.

    BH across all tested pair × library × eligible term tests, including zero hits.
    Use the actual propagated nodes as background; remove shared seeds only when
    the source run did so. Pair significance uses the original size-endpoint BH q.
    With collapse, orsum summarises each plot's significant terms into representatives;
    q-values are never recomputed after collapsing.
    """
    if not 0 < fdr < 1 or min_size < 1:
        raise ValueError('Require 0 < fdr < 1 and min_size >= 1')
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text())
    identity = manifest['identity']
    if manifest['status'] != 'complete':
        raise ValueError('Source run is incomplete')
    if not identity['inputs'].get('network', '').endswith('_brain'):
        raise ValueError('A brain-filtered source run is required')
    for name in ('pair_tests.tsv', 'candidate_genes.tsv', 'nps.tsv'):
        if file_hash(source/name) != manifest['artifacts'][name]:
            raise ValueError(f'Source artifact changed: {name}')
    paths = {name: Path(annotation_dir)/filename for name, filename in LIBRARIES.items()}
    if backend not in ('enrichr', 'offline'):
        raise ValueError('backend must be enrichr or offline')
    if backend == 'enrichr':
        from enrichr_background import LIBRARY_IDS, prepare_libraries, enrichr_ora
        paths.update(prepare_libraries(Path(output_root)/'enrichr_cache'))
    orsum = None
    if collapse:
        from shared_network_orsum import find_orsum
        binary = find_orsum(orsum_executable)
        orsum = {'version': subprocess.check_output([str(binary), '--version'], text=True).strip(),
                 'files': {n: file_hash(binary.parent/n) for n in ('orsum.py', 'termCombinationLib.py', 'plotFunctions.py')}}
    meta = {'source_manifest': file_hash(source/'manifest.json'), 'code': file_hash(__file__),
            'ora_code': file_hash(Path(__import__('cnv_overlap.enrichment', fromlist=['ora']).__file__)),
            'libraries': {name: file_hash(p) for name, p in paths.items()},
            'fdr': fdr, 'min_size': min_size, 'backend': backend, 'orsum': orsum,
            'backend_code': file_hash(Path(__file__).with_name('enrichr_background.py')) if backend == 'enrichr' else None}
    out = run_dir(output_root, meta, 'brain-pair-enrichment')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', {'status': 'running', 'identity': meta})
    tests = pd.read_csv(source/'pair_tests.tsv', sep='\t')
    tested = tests.loc[(tests['null'] == 'netcoloc') & (tests.statistic == 'size') &
                       (tests.status == 'ok')].copy()
    entities = identity['entities']
    def pair_group(row):
        kinds = [entities[name]['kind'] for name in (row.a, row.b)]
        if kinds == ['cnv', 'cnv']:
            return 'CNV × CNV'
        if 'cnv' in kinds:
            target = row.b if kinds[0] == 'cnv' else row.a
            return {'schema': 'CNV × SCHEMA', 'gwaspri': 'CNV × GWAS'}.get(target)
        return None
    def locus(row):
        """Plot row: the CNV for CNV × target pairs, the pair for CNV × CNV."""
        if row.pair_group == 'CNV × CNV':
            return ' × '.join(sorted((row.a, row.b), key=locus_key))
        return row.a if entities[row.a]['kind'] == 'cnv' else row.b
    tested['pair_group'] = [pair_group(r) for r in tested.itertuples()]
    tested = tested.dropna(subset=['pair_group'])
    tested['significant'] = tested.q < fdr
    # Every CNV is shown against SCHEMA and GWAS; CNV × CNV keeps significant pairs only.
    tested = tested.loc[(tested.pair_group != 'CNV × CNV') | tested.significant].copy()
    tested['locus'] = [locus(r) for r in tested.itertuples()]
    tested['pair'] = [f'{r.a} × ' + TARGET_LABELS.get(r.b, r.b) for r in tested.itertuples()]
    order = sorted(range(len(tested)), key=lambda i: (tested.pair_group.iloc[i],
        [locus_key(x) for x in tested.locus.iloc[i].split(' × ')]))
    tested = tested.iloc[order]
    candidates = pd.read_csv(source/'candidate_genes.tsv', sep='\t')
    nodes = pd.read_csv(source/'nps.tsv', sep='\t', usecols=['entrez']).entrez.astype(str)
    symbol_map = identity['symbols']
    background = {symbol_map.get(g, g) for g in nodes}
    libraries = {name: load_gmt(path) for name, path in paths.items()}
    frames, coverage, queries, backgrounds = [], [], [], []
    for r in tested.itertuples():
        bg = background.copy()
        if identity['config']['overlap'] == 'remove':
            shared = set(entities[r.a]['seeds']) & set(entities[r.b]['seeds'])
            bg -= {symbol_map.get(str(g), str(g)) for g in shared}
        query_rows = candidates.loc[(candidates.a == r.a) & (candidates.b == r.b)].copy()
        query = set(query_rows.symbol.dropna().astype(str))
        if not query <= bg:
            raise ValueError(f'Query outside eligible background: {r.pair}')
        queries.append(query_rows.assign(pair=r.pair, pair_group=r.pair_group))
        backgrounds.extend({'pair': r.pair, 'symbol': g} for g in sorted(bg))
        for library, terms in libraries.items():
            # Reactome 2026 is not an Enrichr library: same one-sided hypergeometric test, run locally.
            if backend == 'enrichr' and library in LIBRARY_IDS:
                result = enrichr_ora(query, bg, terms, library=library,
                    cache_dir=Path(output_root)/'enrichr_cache', min_size=min_size)
            else:
                result = ora(query, bg, terms, min_size=min_size).rename(columns={'q': 'q_within_pair_library'})
            result['test_source'] = 'enrichr' if backend == 'enrichr' and library in LIBRARY_IDS else 'local'
            result = result.assign(a=r.a, b=r.b, pair=r.pair, pair_group=r.pair_group, locus=r.locus,
                                   pair_q=r.q, pair_significant=r.significant, library=library)
            frames.append(result)
            covered = set().union(*(genes for _, genes in terms.values()))
            coverage.append({'pair': r.pair, 'library': library, 'query_genes': len(query),
                             'query_annotated': len(query & covered), 'background_genes': len(bg),
                             'background_annotated': len(bg & covered), 'eligible_terms': len(result)})
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        'a', 'b', 'pair', 'pair_group', 'locus', 'pair_q', 'pair_significant', 'library', 'term', 'p',
        'overlap', 'query_size', 'fold_enrichment', 'genes'])
    results['q_all_pair_terms'] = bh(results.p)
    results = results.sort_values(['q_all_pair_terms', 'p'])
    tested.to_csv(out/'tested_pairs.tsv', sep='\t', index=False)
    results.to_csv(out/'enrichment_all_terms.tsv', sep='\t', index=False)
    results.loc[results.q_all_pair_terms < fdr].to_csv(out/'enrichment_significant.tsv', sep='\t', index=False)
    pd.DataFrame(coverage, columns=['pair', 'library', 'query_genes', 'query_annotated',
        'background_genes', 'background_annotated', 'eligible_terms']).to_csv(out/'annotation_coverage.tsv', sep='\t', index=False)
    (pd.concat(queries, ignore_index=True) if queries else candidates.iloc[:0]).to_csv(out/'query_genes.tsv', sep='\t', index=False)
    pd.DataFrame(backgrounds, columns=['pair', 'symbol']).to_csv(out/'background_genes.tsv', sep='\t', index=False)
    network = identity['inputs']['network']
    groups = ['CNV × CNV'] if identity['config']['comparisons'] == 'cnv_cnv' else ['CNV × SCHEMA', 'CNV × GWAS']
    for group in groups:
        group_results, rows = results.loc[results.pair_group == group], tested.loc[tested.pair_group == group]
        stem = out/FIGURES[group]
        options = dict(group=group, network=network, fdr=fdr)
        if collapse:
            mapping = collapse_terms(group_results, libraries, out/'orsum'/FIGURES[group],
                                     binary=binary, fdr=fdr, min_size=min_size)
            mapping.to_csv(stem.with_name(stem.name + '_term_mapping.tsv'), sep='\t', index=False)
            plot_panel(group_results, rows, stem, mapping=mapping, **options)
            plot_panel(group_results, rows, stem.with_name(stem.name + '_all_terms'), **options)
        else:
            plot_panel(group_results, rows, stem, **options)
    summary = (f'{network}: ' + ' and '.join(groups) + '\n\n'
               f'{len(tested)} tested pairs ({int(tested.significant.sum())} with NetColoc size BH q < {fdr}); '
               f'{len(results)} eligible pair–term tests; '
               f'{int((results.q_all_pair_terms < fdr).sum())} enriched pair–terms (BH q < {fdr}).\n\n'
               'Pairs: every CNV × SCHEMA and CNV × GWAS pair (significant pairs marked * in plots); '
               'CNV × CNV only for significant pairs.\n'
               'Queries: thresholded shared candidate genes for each tested pair.\n'
               'Background: actual propagated brain-network genes, excluding shared seeds for overlap=remove.\n'
               f'One-sided over-representation analysis; minimum {min_size} background genes per term. '
               'BH across all tested pairs and all three libraries, including zero-hit terms.\n'
               f'Backend: {backend}; Enrichr (Speedrichr custom-background API) for GO BP 2026 and SynGO 2024; '
               'Reactome uses the official June 2026 GMT with the same test computed locally.\n'
               'SynGO provides brain-focused context; GO and Reactome use the same brain background.\n'
               'Descriptive enrichment after network and pair selection; q-values do not account for that selection.\n'
               'Plots show, per library, the 2 strongest significant terms of every row, filled to 12 with the '
               'strongest terms overall (ordered by best combined q); full tables retain all tests.\n'
               + (f'Collapsed plots: orsum {orsum["version"]} per plot and library, one ranked list per row, on the '
                  'full annotation sets; each column is a representative term, (+n) counts the terms it represents, '
                  'and a dot shows the best original q among them. *_all_terms plots are uncollapsed; '
                  '*_term_mapping.tsv lists every representative and represented term.\n' if collapse else ''))
    (out/'summary.md').write_text(summary)
    write_json(out/'manifest.json', {'status': 'complete', 'identity': meta, 'source': str(source.resolve()),
        'interpretation': summary, 'artifacts': {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*')
            if p.is_file() and p.name != 'manifest.json'}})
    return out


def collapse_terms(results, libraries, directory, *, binary, fdr, min_size):
    """Run orsum per library with each row's significant terms as one ranked list.

    Representatives are chosen jointly across rows, so a column means the same
    biology in every row. Returns library, representative and represented terms.
    """
    significant = results.loc[results.q_all_pair_terms < fdr]
    frames = [pd.DataFrame(columns=['library', 'representative', 'represented'])]
    for library, terms in libraries.items():
        hits = significant.loc[significant.library == library].sort_values(['q_all_pair_terms', 'p'], kind='stable')
        if hits.empty:
            continue
        folder = directory/re.sub(r'\W', '', LIBRARY_SHORT.get(library, library))
        folder.mkdir(parents=True, exist_ok=True)
        ids = {term: f'T{i:05d}' for i, term in enumerate(sorted(terms))}
        names = {i: term for term, i in ids.items()}
        with (folder/'annotations.gmt').open('w') as f:
            for term, (_, genes) in sorted(terms.items()):
                f.write('\t'.join([ids[term], term, *sorted(genes)]) + '\n')
        files, aliases = [], []
        for i, (locus, ranked) in enumerate(hits.groupby('locus', sort=False)):
            path = folder/f'ranked_row{i:02d}.txt'
            path.write_text(''.join(ids[t] + '\n' for t in ranked.term))
            files.append(str(path.resolve()))
            aliases.append(f'row{i:02d}')
        pd.DataFrame({'alias': aliases, 'locus': hits.locus.drop_duplicates().tolist()}).to_csv(
            folder/'rows.tsv', sep='\t', index=False)
        command = [str(binary), '--gmt', str((folder/'annotations.gmt').resolve()), '--files', *files,
                   '--fileAliases', *aliases, '--outputFolder', str((folder/'upstream').resolve()),
                   '--minTermSize', str(min_size), '--maxRepSize', '1000000', '--maxTermSize', '1000000',
                   '--numberOfTermsToPlot', '50']
        result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, 'MPLBACKEND': 'Agg'})
        (folder/'command.json').write_text(json.dumps(command, indent=2) + '\n')
        (folder/'stdout.log').write_text(result.stdout)
        (folder/'stderr.log').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'orsum failed for {library}; see {folder}/stderr.log')
        mapping = pd.read_csv(folder/'upstream/filteredResultIDMapping.tsv', sep='\t')
        if set(mapping.Represented) != {ids[t] for t in hits.term}:
            raise ValueError(f'orsum did not account for every significant term: {library}')
        frames.append(pd.DataFrame({'library': library, 'representative': mapping.Representative.map(names),
                                    'represented': mapping.Represented.map(names)}))
    return pd.concat(frames, ignore_index=True)


def term_label(term, width=70):
    """Drop GO identifiers, keep SynGO's BP/CC tag, and shorten long names."""
    term = re.sub(r'\s*\(GO:\d+\)\s*(BP|CC)?$', lambda m: f' [{m[1]}]' if m[1] else '', term)
    return term if len(term) <= width else term[:width-1].rstrip() + '…'


def plot_panel(results, rows, stem, *, group, network, fdr, top=12, per_row=2, mapping=None, styles=None):
    """One dot plot: rows are terms grouped under library headers, columns are loci.

    Terms run horizontally so long names stay readable on slides; significant
    loci are starred and bold. ``styles`` maps library → (short label, colour);
    defaults to the brain libraries.

    With an orsum mapping, rows are representatives and each dot is the best
    original q among the terms that representative covers in that locus.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory
    styles = styles or {l: (LIBRARY_SHORT[l], LIBRARY_COLORS[l]) for l in LIBRARIES}
    target = group.split(' × ')[1]
    loci = rows.locus.tolist()
    if target == 'CNV':
        labels = loci
        bold = [False]*len(loci)
        title = f'What each significant CNV × CNV overlap is enriched for ({network})'
        note = ''
    else:
        labels = [f'{r.locus} *' if r.significant else r.locus for r in rows.itertuples()]
        bold = rows.significant.astype(bool).tolist()
        title = f"What each CNV's {target} overlap is enriched for ({network})"
        note = f'* significant pair (NetColoc q < {fdr:g})   ·   '
    note += f'dots: enrichment BH q < {fdr:g}'
    significant = results.loc[results.q_all_pair_terms < fdr].copy()
    represented = {}
    if mapping is not None:
        note += '   ·   (+n) = similar terms merged by orsum'
        # A term orsum places under several representatives joins the most significant one.
        best = significant.groupby(['library', 'term']).q_all_pair_terms.min()
        ranked = mapping.assign(rep_q=[best.get((l, r), np.inf) for l, r in zip(mapping.library, mapping.representative)])
        ranked = ranked.sort_values('rep_q', kind='stable').drop_duplicates(['library', 'represented'])
        lookup = {(r.library, r.represented): r.representative for r in ranked.itertuples()}
        represented = ranked.groupby(['library', 'representative']).size().to_dict()
        significant['represented_term'] = significant.term
        significant['term'] = [lookup[k] for k in zip(significant.library, significant.term)]
        significant = significant.sort_values('q_all_pair_terms', kind='stable').drop_duplicates(['library', 'term', 'locus'])
    blocks = {}
    for library in styles:
        # A strong pair must not crowd out every other locus: keep each locus's best terms first.
        hits = significant.loc[(significant.library == library) & significant.locus.isin(loci)]
        best = hits.groupby('term').q_all_pair_terms.min().sort_values(kind='stable')
        chosen = set(hits.sort_values('q_all_pair_terms', kind='stable').groupby('locus').head(per_row).term)
        chosen |= set(best.index[:max(0, top-len(chosen))])
        if chosen:
            blocks[library] = [term for term in best.index if term in chosen]
    # Long lists split into two side-by-side columns at a library boundary, so the figure stays slide-shaped.
    height = lambda libs: sum(len(blocks[l]) + 1.4 for l in libs)
    order = list(blocks)
    split = min(range(1, len(order)), key=lambda i: max(height(order[:i]), height(order[i:])), default=len(order))
    panels = [order[:split], order[split:]] if height(order) > 24 and len(order) > 1 else [order]
    x = {locus: i for i, locus in enumerate(loci)}
    keys = {(l, t) for l in order for t in blocks[l]}
    plotted = significant.loc[[(r.library, r.term) in keys and r.locus in x for r in significant.itertuples()]]
    strength = -np.log10(plotted.q_all_pair_terms.astype(float).clip(lower=1e-300))
    size = lambda s: 30 + 16*np.minimum(s, 25)  # area ∝ −log10 q, capped at q = 1e-25
    def label(library, term):
        extra = represented.get((library, term), 1) - 1
        return term_label(term, 55) + (f' (+{extra})' if extra else '')
    rows_high = max([height(libs) for libs in panels] + [1])
    fig_h = max(4.5, .31*rows_high + 2.6)
    with plt.rc_context({'font.size': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 11.5,
                         'legend.fontsize': 11, 'legend.title_fontsize': 12}):
        fig, axes = plt.subplots(1, len(panels), figsize=(len(panels)*(6.8 + .5*len(loci)) + 1.8, fig_h),
                                 squeeze=False)
        for ax, libs in zip(axes[0], panels):
            # Each library block gets a tinted header row, then its terms, strongest first.
            y, pos = {}, 0.
            for library in libs:
                short, color = styles[library]
                ax.axhspan(pos - .5, pos + .45, color=color, alpha=.14, zorder=0, linewidth=0)
                ax.text(-.02, pos, short, transform=blended_transform_factory(ax.transAxes, ax.transData),
                        ha='right', va='center', fontweight='bold', fontsize=12.5)
                for term in blocks[library]:
                    pos += 1
                    y[library, term] = pos
                pos += 1.4
            for library in libs:
                keep = plotted.library == library
                ax.scatter(plotted.locus[keep].map(x), [y[k] for k in zip(plotted.library[keep], plotted.term[keep])],
                           s=size(strength[keep]), color=styles[library][1], edgecolors='white', linewidths=1, zorder=3)
            ax.set_yticks(list(y.values()), [label(*k) for k in y])
            ax.set_ylim(rows_high - .9, -.9)  # same row pitch in both columns
            ax.set_xticks(range(len(loci)), labels, rotation=40, ha='left', rotation_mode='anchor')
            ax.xaxis.tick_top()
            for tick, strong in zip(ax.xaxis.get_major_ticks(), bold):
                tick.label2.set_fontweight('bold' if strong else 'normal')
            ax.tick_params(length=0, pad=6)
            ax.set_xlim(-.6, max(len(loci), 1) - .4)
            ax.grid(color='#e5e4df', linewidth=.6, zorder=0)
            ax.set_axisbelow(True)
            ax.spines[:].set_visible(False)
            if not libs:
                ax.set_yticks([])
                ax.text(.5, .5, f'No enriched terms at combined BH q < {fdr:g}', transform=ax.transAxes,
                        ha='center', va='center', color='#52514e')
        ax = axes[0, -1]
        handles = [Line2D([], [], marker='o', linestyle='', markersize=10, color=color, label=short)
                   for short, color in styles.values()]
        first = ax.legend(handles=handles, title='Library', loc='upper left', bbox_to_anchor=(1.02, 1),
                          frameon=False, alignment='left')
        ax.add_artist(first)
        sizes = [ax.scatter([], [], s=size(v), color='#8a8984', label=text)
                 for v, text in ((-np.log10(fdr), f'{fdr:g}'), (5, '1e-5'), (10, '1e-10'), (25, '≤1e-25'))]
        ax.legend(handles=sizes, title='Enrichment q', loc='upper left', bbox_to_anchor=(1.02, .78 if fig_h > 7 else .6),
                  frameon=False, labelspacing=1.4, borderpad=.8, alignment='left')
        # Title and note sit at fixed inch offsets from the top so they never collide.
        top = lambda inches: 1 - inches/fig_h
        fig.text(.01, top(.15), title, ha='left', va='top', fontsize=17, fontweight='bold')
        fig.text(.01, top(.55), note, ha='left', va='top', fontsize=11.5, color='#52514e')
        fig.tight_layout(rect=(0, 0, 1, top(.85)), w_pad=3)
        for ext in ('png', 'pdf'):
            fig.savefig(stem.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    plotted.to_csv(stem.with_name(stem.name + '_plotted_terms.tsv'), sep='\t', index=False)
