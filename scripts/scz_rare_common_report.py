"""Summarize separate CNV–SCHEMA and CNV–GWAS pathway tests."""
from cnv_overlap.results import finalize_manifest, resolve_path
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cnv_overlap.provenance import file_hash, run_dir, write_json
from cnv_overlap.statistics import bh


def run(sources, output):
    sources = [Path(s).resolve() for s in sources]
    identity = {'sources': {str(s): file_hash(s/'manifest.json') for s in sources},
                'code': file_hash(__file__),
                'statistics_code': file_hash(Path(__file__).resolve().parents[1]/'src/cnv_overlap/statistics.py')}
    out = run_dir(Path(output), identity, 'rare-common-review')
    from cnv_overlap.results import reuse_completed
    if reuse_completed(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for source in sources:
        manifest = json.loads((source/'manifest.json').read_text())
        if manifest.get('status') != 'complete':
            raise ValueError('Require completed source runs')
        if file_hash(resolve_path(source/'pathway_tests.tsv')) != manifest['artifacts']['pathway_tests.tsv']:
            raise ValueError('Changed source test table')
        frozen = Path(manifest['identity']['source'])
        network = json.loads((frozen/'manifest.json').read_text())['identity']['settings']['network']
        frames.append(pd.read_csv(resolve_path(source/'pathway_tests.tsv'), sep='\t').assign(network=network,
                      source_run=str(source), draws=manifest['identity']['reps']))
    all_tests = pd.concat(frames, ignore_index=True)
    for _, indexes in all_tests.groupby(['network', 'endpoint']).groups.items():
        pvalues = all_tests.loc[indexes, 'p'].to_numpy()
        order = np.argsort(pvalues, kind='stable')
        holm = np.empty(len(pvalues))
        holm[order] = np.minimum(1, np.maximum.accumulate(pvalues[order]*np.arange(len(pvalues), 0, -1)))
        all_tests.loc[indexes, 'p_holm_both_branches'] = holm
        all_tests.loc[indexes, 'q_by_both_branches'] = np.minimum(1,
            all_tests.loc[indexes, 'q_both_branches']*np.sum(1/np.arange(1, len(pvalues)+1)))
    for endpoint, indexes in all_tests.groupby('endpoint').groups.items():
        all_tests.loc[indexes, 'q_both_networks_and_branches'] = bh(all_tests.loc[indexes, 'p'])
    all_tests.to_csv(resolve_path(out/'all_pathway_tests.tsv'), sep='\t', index=False)
    main = all_tests[all_tests.endpoint == 'pair_coverage'].copy()
    main.to_csv(resolve_path(out/'main_convergence.tsv'), sep='\t', index=False)
    discoveries = main[main.q_both_branches < .05].sort_values(['network', 'mode', 'q_both_branches', 'p'])
    discoveries.to_csv(resolve_path(out/'bh_discoveries_both_branches.tsv'), sep='\t', index=False)
    top = main.sort_values(['p', 'pathway']).groupby(['network', 'mode']).head(10)
    top.to_csv(resolve_path(out/'top10_per_branch.tsv'), sep='\t', index=False)
    summary = all_tests.groupby(['network', 'mode', 'endpoint']).agg(tests=('p', 'size'),
        min_q_both_branches=('q_both_branches', 'min'),
        discoveries_both_branches=('q_both_branches', lambda x: int((x < .05).sum())),
        discoveries_all_networks_branches=('q_both_networks_and_branches', lambda x: int((x < .05).sum())))
    summary.to_csv(resolve_path(out/'summary.tsv'), sep='\t')
    lines = ['# Rare/common Reactome convergence', '',
        'Separate full-interactome tests; no direct rare/common difference test was performed.',
        'A detection in one branch does not establish specificity or preferential support.', '',
        '| Network | Branch | Pathways | Main discoveries, BH across both branches | Minimum q |',
        '|---|---|---:|---:|---:|']
    for (network, mode), group in main.groupby(['network', 'mode']):
        lines.append(f'| {network} | {mode} | {len(group)} | {int((group.q_both_branches < .05).sum())} | {group.q_both_branches.min():.4g} |')
    lines += ['', 'All raw tests, effects, Monte Carlo intervals and correction columns are retained.',
        'The extra q_both_networks_and_branches column guards against selecting either network after inspection.',
        'p_holm_both_branches and q_by_both_branches additionally correct across both targets, allowing arbitrary pathway dependence.',
        'Top-ranked pathways are not discoveries unless their correction criterion is met.',
        'Reactome pathways overlap and are hierarchical; related names are not independent replications.', '',
        '## Source runs', ''] + [f'- {s}' for s in sources]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'artifacts': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.name != 'manifest.json'}})
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--output', default='results/publication/rare_common_reactome')
    args = parser.parse_args()
    print(run(args.sources, args.output))
