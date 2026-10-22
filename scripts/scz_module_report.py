"""Archive main module results and leave-one-CNV-out sensitivity from saved draws."""
from cnv_overlap.results import finalize_manifest, resolve_path
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cnv_overlap.provenance import file_hash, run_dir, write_json
from cnv_overlap.statistics import bh, tail
from scz_module_inference import module_statistics


def leave_one_out(data):
    names, modules = data['cnvs'], data['modules']
    rows = []
    for omit, name in enumerate(names):
        keep = np.arange(len(names)) != omit
        observed = module_statistics(data['observed_counts'][keep], data['observed_totals'][keep],
                                     data['module_sizes'])['pair_coverage']
        null = module_statistics(data['counts'][:, keep], data['totals'][:, keep],
                                 data['module_sizes'])['pair_coverage']
        for j, module in enumerate(modules):
            rows.append({'omitted_cnv': str(name), 'module': str(module),
                         **tail(observed[j], null[:, j])})
    result = pd.DataFrame(rows)
    result['q_all_omissions_modules'] = bh(result.p)
    return result


def run(sources, output):
    sources = [Path(s).resolve() for s in sources]
    manifests = [json.loads((s/'manifest.json').read_text()) for s in sources]
    for source, manifest in zip(sources, manifests):
        if manifest.get('status') != 'complete':
            raise ValueError('Require completed module runs')
        for f in ('module_tests.tsv', 'module_draws_direct.npz'):
            if file_hash(resolve_path(source/f)) != manifest['artifacts'][manifest.get('artifact_paths', {}).get(f, f)]:
                raise ValueError(f'Changed source artifact: {source/f}')
    identity = {'sources': {str(s): file_hash(s/'manifest.json') for s in sources},
                'code': file_hash(__file__),
                'statistics_code': file_hash(Path(__file__).resolve().parents[1]/'src/cnv_overlap/statistics.py'),
                'module_code': file_hash(Path(__file__).with_name('scz_module_inference.py'))}
    out = run_dir(Path(output), identity, 'module-review')
    from cnv_overlap.results import reuse_completed
    if reuse_completed(out):
        return out
    out.mkdir(parents=True, exist_ok=True)
    main, loo = [], []
    for source, manifest in zip(sources, manifests):
        frozen = Path(manifest['identity']['source'])
        network = json.loads((frozen/'manifest.json').read_text())['identity']['settings']['network']
        table = pd.read_csv(resolve_path(source/'module_tests.tsv'), sep='\t')
        main.append(table.query("mode == 'direct' and endpoint == 'pair_coverage'").assign(network=network))
        with np.load(resolve_path(source/'module_draws_direct.npz')) as data:
            loo.append(leave_one_out(data).assign(network=network))
    summary = pd.concat(main, ignore_index=True)
    summary['q_both_networks'] = bh(summary.p)
    summary.to_csv(resolve_path(out/'main_module_tests.tsv'), sep='\t', index=False)
    pd.concat(loo, ignore_index=True).to_csv(resolve_path(out/'leave_one_cnv_out.tsv'), sep='\t', index=False)
    lines = ['# Fixed-landscape module convergence', '',
             'Exploratory direct-reach/pair-coverage endpoint; 5% BH threshold.', '',
             '| Network | Tested modules | BH-significant modules | Minimum q |',
             '|---|---:|---:|---:|']
    for network, group in summary.groupby('network'):
        lines.append(f'| {network} | {len(group)} | {int((group.q < .05).sum())} | {group.q.min():.4g} |')
    lines += ['', 'No statistically established module convergence may be claimed from a family with no discoveries.',
              'The combined-network q column guards against selecting either network after inspection.',
              'Leave-one-CNV-out tests reuse the same joint draws and frozen modules; correction spans all',
              'omissions × modules within each network. These are dependent sensitivity analyses, not replication.',
              'See docs/molecular_psychiatry_analysis.md for null scope and outstanding validation.', '']
    (out/'README.md').write_text('\n'.join(lines))
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'artifacts': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.name != 'manifest.json'}})
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--output', default='results/publication/scz_module_review')
    args = parser.parse_args()
    print(run(args.sources, args.output))
