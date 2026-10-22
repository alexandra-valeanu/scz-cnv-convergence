"""Count-based shared CNV network and its descriptive functional context."""
from cnv_overlap.results import finalize_manifest, resolve_path
import argparse
from pathlib import Path
import re
import shutil
import sys
import textwrap

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT/'scripts'))

from cnv_overlap.pipeline import AnalysisConfig
from cnv_overlap.provenance import file_hash, graph_hash, run_dir, write_json
from cnv_overlap.statistics import bh
from network_inputs import load_network_inputs
from results_layout import results_dir
from notebook_analysis import SCZ_TARGETS, export_scz_networks
from shared_core_figures import _colors, shared_core_figures
from shared_core_sensitivity import run_core_sensitivity
from shared_network_enrichment import enrich_shared_network

MAX_TERMS = 30


def load_inputs(network, min_seeds, convergence_config=None, disorder_config=None):
    spec, root, inputs, _ = load_network_inputs(ROOT/(convergence_config or 'configs/cnv_convergence.json'), network, min_seeds=min_seeds)
    _, _, scz_inputs, _ = load_network_inputs(ROOT/(disorder_config or 'configs/cnv_disorder_overlap.json'), network, min_seeds=min_seeds)
    if graph_hash(scz_inputs[0], scz_inputs[1]) != graph_hash(inputs[0], inputs[1]):
        raise ValueError('SCZ targets require the same network universe as CNV convergence')
    targets = {n: e for n, e in scz_inputs[3].items() if n in SCZ_TARGETS and e['kind'] == 'target'}
    if set(targets) != set(SCZ_TARGETS):
        raise ValueError('Both SCHEMA and GWAS-prioritized targets are required')
    A, nodes, symbols, entities, coordinates, regions = inputs
    return spec, root, (A, nodes, symbols, entities | targets, coordinates, regions)


def cnv_processes(genes, enrichment, cnv_names, *, output_root, fdr, min_cnvs):
    """CNV x process contribution inside the shared network.

    For each CNV and each enriched term, x = core genes of the term reached by that CNV
    (with both SCHEMA and GWAS). The tail P(X >= x) draws the CNV's core genes from the
    whole core, i.e. does this CNV contribute more of the term than its share of the
    core predicts. BH across all CNV x term cells. Descriptive, conditional on the core.
    """
    enrichment = Path(enrichment)
    terms = pd.read_csv(resolve_path(enrichment/'enrichment_all_terms.tsv'), sep='\t')
    identity = {'enrichment_manifest_sha256': file_hash(enrichment/'manifest.json'),
                'code': file_hash(__file__), 'fdr': fdr, 'max_terms': MAX_TERMS}
    out = run_dir(output_root, identity, 'cnv-processes', f'min{min_cnvs}cnvs')
    from cnv_overlap.results import reuse_completed
    if reuse_completed(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    selected = terms[terms.q_all_libraries < fdr]
    basis = f'core enrichment q < {fdr}'
    if selected.empty:
        selected, basis = terms[terms.overlap >= 2].nsmallest(MAX_TERMS, 'p'), 'top core terms by P (none at FDR)'
    selected = selected.sort_values(['q_all_libraries', 'p']).head(MAX_TERMS)
    reached = {c: set(genes.loc[genes.cnvs_both.fillna('').str.split(',').apply(lambda s: c in s), 'symbol'])
               for c in cnv_names}
    N = len(genes)
    rows = []
    for t in selected.itertuples():
        members = set(str(t.genes).split(',')) if isinstance(t.genes, str) else set()
        for c in cnv_names:
            x = len(members & reached[c])
            n = len(reached[c])
            rows.append({'cnv': c, 'library': t.library, 'term': t.term, 'core_term_q': t.q_all_libraries,
                         'term_core_genes': len(members), 'cnv_core_genes': n, 'core_genes': N,
                         'cnv_term_genes': x, 'share_of_term': x/len(members) if members else np.nan,
                         'fold_vs_cnv_share': x/(n*len(members)/N) if n and members else np.nan,
                         'p_within_core': float(hypergeom.sf(x-1, N, len(members), n)) if x else 1.,
                         'genes': ','.join(sorted(members & reached[c]))})
    table = pd.DataFrame(rows)
    if not table.empty:
        table['q_within_core'] = bh(table.p_within_core)
    table.to_csv(resolve_path(out/'cnv_term_contribution.tsv'), sep='\t', index=False)
    functions = (genes.assign(cnv=genes.cnvs_both.fillna('').str.split(',')).explode('cnv')
                 .query('cnv in @cnv_names').groupby(['cnv', 'function']).size().unstack(fill_value=0))
    functions.to_csv(resolve_path(out/'cnv_function_counts.tsv'), sep='\t')
    plot_cnv_processes(table, functions, cnv_names, out, fdr=fdr, min_cnvs=min_cnvs, basis=basis)
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity, 'term_basis': basis,
        'test': 'hypergeometric within the core: CNV core genes drawn from all core genes; BH over CNV x term',
        'interpretation': 'descriptive; conditional on core membership and on core enrichment'})
    return out


def plot_cnv_processes(table, functions, cnv_names, out, *, fdr, min_cnvs, basis):
    import matplotlib.pyplot as plt

    colors = _colors()
    order = functions.sum(axis=1).sort_values(ascending=False).index.tolist()
    order += [c for c in cnv_names if c not in order]
    terms = table.drop_duplicates('term')[['library', 'term']] if not table.empty else pd.DataFrame(columns=['library', 'term'])
    fig_h = max(5, .32*len(terms) + 3)
    fig, (bar, ax) = plt.subplots(1, 2, figsize=(9 + .55*len(order), fig_h),
                                  gridspec_kw={'width_ratios': [4, max(3, .55*len(order))], 'wspace': .9})
    # Left: function-category composition of each CNV's core genes.
    left = np.zeros(len(order))
    frame = functions.reindex(order).fillna(0)
    for category, color in colors.items():
        if category in frame:
            bar.barh(range(len(order)), frame[category], left=left, color=color, label=category, height=.7)
            left += frame[category].to_numpy()
    bar.set_yticks(range(len(order)), order)
    bar.invert_yaxis()
    bar.set_xlabel('Shared-network genes reached (SCHEMA + GWAS)')
    bar.set_title('Function of each CNV’s shared-network genes', loc='left', fontsize=10)
    bar.spines[['top', 'right']].set_visible(False)
    bar.legend(loc='upper center', bbox_to_anchor=(.5, -.08), ncol=2, frameon=False, fontsize=7.5)
    # Right: CNV x enriched term, colour = share of the term's core genes reached by the CNV.
    if terms.empty:
        ax.text(.5, .5, 'No core terms to display', ha='center', transform=ax.transAxes)
        ax.axis('off')
    else:
        share = table.pivot(index='term', columns='cnv', values='share_of_term').loc[terms.term, order]
        count = table.pivot(index='term', columns='cnv', values='cnv_term_genes').loc[terms.term, order]
        q = table.pivot(index='term', columns='cnv', values='q_within_core').loc[terms.term, order]
        image = ax.imshow(share.to_numpy(dtype=float), aspect='auto', cmap='Blues', vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(count.to_numpy()):
            if v:
                ax.text(j, i, f'{int(v)}{"*" if q.iat[i, j] < fdr else ""}', ha='center', va='center', fontsize=7,
                        color='white' if share.iat[i, j] > .6 else '#1a1a19')
        labels = [textwrap.shorten(re.sub(r'\s*\(GO:\d+\)$', '', t), 60) + f'  [{lib.split()[0]}]'
                  for lib, t in terms.itertuples(index=False)]
        ax.set_yticks(range(len(terms)), labels, fontsize=7.5)
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        cbar = fig.colorbar(image, ax=ax, fraction=.04, pad=.02)
        cbar.set_label('Share of the term’s shared-network genes reached by the CNV', fontsize=8)
        ax.set_title(f'Enriched processes by CNV ({basis})\ncell = genes; * = CNV over-contributes '
                     f'vs its share of the network (BH q < {fdr})', fontsize=9, pad=40)
    fig.suptitle(f'Which CNVs drive which processes in the SCZ shared network (≥{min_cnvs} CNVs)', fontsize=12)
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'cnv_processes.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--network', default='PCNET2.2_brain')
    parser.add_argument('--min-cnvs', type=int, nargs='+', default=[3, 2])
    parser.add_argument('--min-seeds', type=int, default=3)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    spec, root, inputs = load_inputs(args.network, args.min_seeds)
    config = AnalysisConfig.from_dict(spec['analysis'])
    cache_dir = root/spec['cache_dir']
    results_root = root/spec['results_root']
    annotation_dir = root/'input/annotations'
    base = results_dir(results_root, 'scz_shared_network')
    ks = sorted(set(args.min_cnvs), reverse=True)
    # One joint null covers every threshold; same location/identity as the notebook so it is reused.
    sensitivity = run_core_sensitivity(inputs, config=config, cache_dir=cache_dir,
        output_root=results_dir(results_root, 'scz_core_sensitivity'), thresholds=tuple(sorted({2, 3, 4} | set(ks))),
        primary_min_cnvs=ks[0], annotation_dir=annotation_dir)
    tests = pd.read_csv(resolve_path(sensitivity/'core_size_tests.tsv'), sep='\t').query("mode == 'both'").set_index('min_cnvs')
    cnv_names = sorted(n for n, e in inputs[3].items() if e['kind'] == 'cnv')
    for k in ks:
        bundle_identity = {'network': args.network, 'min_cnvs': k, 'sensitivity': file_hash(sensitivity/'manifest.json'), 'code': file_hash(__file__)}
        out = run_dir(base, bundle_identity, 'shared-network', f'min{k}cnvs')
        if (out/'manifest.json').exists():
            from cnv_overlap.results import verify_run
            verify_run(out)
            continue
        out.mkdir(parents=True, exist_ok=True)
        networks = export_scz_networks(inputs, config=config, cache_dir=cache_dir, output_root=results_dir(results_root, 'scz_colocalised_networks'), min_cnvs=k)
        enrichment = enrich_shared_network(networks, annotation_dir=annotation_dir, output_root=results_dir(results_root, 'scz_network_enrichment'), fdr=config.fdr)
        figures = shared_core_figures(networks, sensitivity_dir=sensitivity, annotation_dir=annotation_dir,
                                      output_root=results_dir(results_root, 'scz_core_figures'))
        index = pd.read_csv(resolve_path(networks/'network_index.tsv'), sep='\t')
        core_dir = networks/index.loc[index.kind == 'shared_core', 'directory'].iloc[0]
        for suffix in ('png', 'pdf'):
            shutil.copyfile(core_dir/f'network.{suffix}', out/f'cnv_pies_network.{suffix}')
        genes = pd.read_csv(resolve_path(core_dir/'genes.tsv'), sep='\t').astype({'symbol': str}).merge(
            pd.read_csv(resolve_path(figures/'gene_functions.tsv'), sep='\t')[['symbol', 'function', 'evidence_term']].astype({'symbol': str}),
            on='symbol', how='left')
        processes = cnv_processes(genes, enrichment, cnv_names, output_root=results_root/'3_shared_scz_networks/reports/cnv_processes',
                                  fdr=config.fdr, min_cnvs=k)
        genes.sort_values(['n_cnvs_both', 'symbol'], ascending=[False, True])[
            ['entrez', 'symbol', 'function', 'evidence_term', 'n_cnvs_both', 'cnvs_both',
             'n_cnvs_schema', 'n_cnvs_gwaspri', 'gene_origin', 'seed_cnvs']].to_csv(resolve_path(out/'genes.tsv'), sep='\t', index=False)
        significant = pd.read_csv(resolve_path(enrichment/'enrichment_significant.tsv'), sep='\t')
        contribution = pd.read_csv(resolve_path(processes/'cnv_term_contribution.tsv'), sep='\t')
        leaning = contribution[contribution.get('q_within_core', pd.Series(dtype=float)) < config.fdr]
        t = tests.loc[k]
        lines = [f'# SCZ shared network, ≥{k} CNVs: {args.network}', '',
                 'Gene kept if ≥ k CNVs each colocalise with both SCHEMA and GWAS at that gene. '
                 'CNV × target pair significance is NOT required; descriptive network.', '',
                 f'{len(genes)} genes: ' + ', '.join(f'{s} ({n})' for s, n in zip(genes.symbol, genes.n_cnvs_both)), '',
                 f'Core size vs joint null (reported, not used for selection): observed {t.observed_genes} vs '
                 f'null mean {t.null_mean:.2f}; P = {t.p_empirical:.3g}, Q = {t.q_sensitivity:.3g}.', '',
                 f'Enrichment: {len(significant)} terms at combined BH q < {config.fdr}.']
        lines += [f'- {r.term} [{r.library}]: {r.overlap}/{r.query_size}, q = {r.q_all_libraries:.3g}'
                  for r in significant.head(10).itertuples()]
        lines += ['', f'CNV over-contribution to enriched processes (within-core BH q < {config.fdr}): '
                  + ('; '.join(f'{r.cnv} → {r.term} ({r.cnv_term_genes}/{r.term_core_genes})' for r in leaning.itertuples())
                     or 'none; see cnv_processes.png for the descriptive pattern'), '',
                 f'- networks: `{networks}`', f'- enrichment: `{enrichment}`', f'- figures: `{figures}`',
                 f'- cnv_processes: `{processes}`', f'- core-size null: `{sensitivity}`']
        (out/'summary.md').write_text('\n'.join(lines) + '\n')
        finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': bundle_identity,
            'sources': {key: {'path': str(value.resolve()), 'manifest_sha256': file_hash(value/'manifest.json')}
                        for key, value in {'networks': networks, 'enrichment': enrichment, 'figures': figures, 'processes': processes, 'sensitivity': sensitivity}.items()}})
        print((out/'summary.md').read_text(), flush=True)
    return base


if __name__ == '__main__':
    main()
