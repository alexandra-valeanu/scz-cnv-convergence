"""Index existing runs without choosing analyses by modification time."""
import argparse
import json
from pathlib import Path

import pandas as pd

from cnv_overlap.provenance import file_hash
from network_inputs import NETWORKS

ROOT = Path(__file__).resolve().parents[1]


def build_index(root=ROOT, verify=False):
    results = Path(root)/'results'
    results.mkdir(exist_ok=True)
    rows = []
    for manifest in sorted(results.glob('**/manifest.json')):
        try:
            data = json.loads(manifest.read_text())
            status = data.get('status', 'unspecified')
            integrity = 'not checked'
            if verify and status == 'complete':
                bad = [f for f, h in data.get('artifacts', {}).items()
                       if not (manifest.parent/f).is_file() or file_hash(manifest.parent/f) != h]
                integrity = 'failed: '+', '.join(bad) if bad else 'verified' if data.get('artifacts') else 'no artifact checksums'
            rows.append({'run': str(manifest.parent.relative_to(results)), 'status': status,
                         'integrity': integrity, 'manifest_sha256': file_hash(manifest)})
        except (ValueError, KeyError, OSError) as error:
            rows.append({'run': str(manifest.parent.relative_to(results)), 'status': 'unreadable', 'integrity': str(error)})
    pd.DataFrame(rows, columns=['run', 'status', 'integrity', 'manifest_sha256']).to_csv(
        results/'run_inventory.tsv', sep='\t', index=False)
    text = ['# Results', '', 'Open the publication report for the network of interest. '
            'A missing report means the new publication workflow has not completed.', '',
            '| Network | Publication report | Supplementary stability |', '| --- | --- | --- |']
    for network in ('PCNET2.2', 'PCNET2.2_brain', 'STRING_high_conf', 'STRING_high_conf_brain'):
        current = results/'publication'/network/'current.json'
        link = 'Not completed'
        if current.exists():
            data = json.loads(current.read_text())
            run = current.parent/data['run']
            manifest = run/'manifest.json'
            if manifest.exists() and file_hash(manifest) == data['manifest_sha256']:
                if json.loads(manifest.read_text()).get('status') == 'complete':
                    link = f'[Report]({run.relative_to(results)}/README.md)'
        supplements = []
        for name in ('nps_stability', 'pvalue_stability'):
            directory = results/network/name
            n = sum(json.loads(p.read_text()).get('status') == 'complete' for p in directory.glob('*/manifest.json'))
            supplements.append(f'[{name}: {n} complete]({network}/{name}/)' if directory.exists() else f'{name}: absent')
        text.append(f'| {network} | {link} | {"; ".join(supplements)} |')
    text.extend(['', '- [All run manifests](run_inventory.tsv): includes previous, running and incomplete analyses.',
                 '- [Executed notebooks](notebooks/): inspection records, separate from the publication workflow.',
                 '- Previous result directories are retained; their presence does not select them for the manuscript.',
                 '', 'Refresh: `.venv/bin/python scripts/index_results.py --verify`.'])
    (results/'README.md').write_text('\n'.join(text)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Check all declared result artifact hashes')
    build_index(verify=parser.parse_args().verify)
