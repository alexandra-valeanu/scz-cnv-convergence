"""Build result inventories and verify registered artifacts."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from cnv_overlap.provenance import file_hash
from cnv_overlap.results import GROUPS, FOLDERS, verify_run

ROOT = Path(__file__).resolve().parents[1]
NETWORK_ORDER = ('PCNET2.2_brain', 'PCNET2.2', 'STRING_high_conf_brain', 'STRING_high_conf')


def publication_link(results, network):
    home = Path(results)/'publication'/network
    pointer = home/'selections.json'
    if not pointer.exists():
        pointer = home/'current.json'
    if not pointer.exists():
        return None
    data = json.loads(pointer.read_text())
    record = data.get('default', data)
    manifest = home/record['run']/'manifest.json'
    if manifest.is_file() and file_hash(manifest) == record['manifest_sha256'] and json.loads(manifest.read_text()).get('status') == 'complete':
        return f'[Report]({manifest.parent.relative_to(results)}/README.md)'
    return None


def network_readme(results, network, rows=None):
    base = Path(results)/network
    rows = rows if rows is not None else []
    role = 'Article analysis' if network.startswith('PCNET') else 'Supporting interactome check'
    text = [f'# {network}', '', f'{role}. Runs record their settings and source identities.', '']
    statuses = []
    selected = set()
    for pointer in base.rglob('selections.json'):
        records = json.loads(pointer.read_text())
        if 'default' in records:
            selected.add((pointer.parent/records['default']['run']).resolve())
    for group, question in GROUPS.items():
        members = [r for r in rows if r['run'].startswith(f'{network}/{group}/')]
        statuses.append({'analysis': group, 'state': 'not run' if not members else 'recorded', 'runs': len(members)})
        text += [f'## {group}', '', question, '']
        group_text = [f'# {group}', '', question, '',
                      'Generated from run manifests. Selected runs are marked explicitly; older runs are retained.', '']
        if not members:
            text += ['Not run.', '']
        for record in members:
            run = Path(record['run']).relative_to(network)
            target = run/'README.md' if (base/run/'README.md').exists() else run
            label = 'selected; ' if (base/run).resolve() in selected else ''
            text += [f'- [{run}]({target}) — {label}{record["status"]}; {record["integrity"]}']
            group_text += [f'- [{run.relative_to(group)}]({target.relative_to(group)}) — '
                           f'{label}{record["status"]}; {record["integrity"]}']
        text.append('')
        if (base/group).is_dir() and not (base/group/'manifest.json').exists():
            (base/group/'README.md').write_text('\n'.join(group_text)+'\n')
    pd.DataFrame(statuses).to_csv(base/'analysis_status.tsv', sep='\t', index=False)
    (base/'README.md').write_text('\n'.join(text)+'\n')


def build_index(root=ROOT, verify=False, *, results_root=None, strict=False):
    results = Path(results_root) if results_root is not None else Path(root)/'results'
    results.mkdir(parents=True, exist_ok=True)
    rows, artifacts, failures = [], [], []
    registered = set()
    manifests = sorted(set(results.rglob('manifest.json')) | set(results.rglob('report_manifest.json')))
    for manifest in manifests:
        if manifest.name == 'report_manifest.json' and (manifest.parent/'manifest.json').exists():
            continue
        try:
            data = json.loads(manifest.read_text())
            status = data.get('status', 'legacy/unknown')
            bad = []
            for name, checksum in data.get('artifacts', {}).items():
                path = manifest.parent/name
                registered.add(path)
                good = path.is_file()
                if verify and good: good = file_hash(path) == checksum
                if not good: bad.append(name)
                artifacts.append({'run': str(manifest.parent.relative_to(results)), 'path': name,
                                  'sha256': checksum, 'integrity': 'verified' if verify and good else 'present' if good else 'failed'})
            for source in data.get('sources', {}).values():
                path = Path(source['path'])
                if not path.is_absolute(): path = Path(root)/path
                source_manifest = path/'manifest.json'
                if not source_manifest.exists() or (verify and file_hash(source_manifest) != source['manifest_sha256']):
                    bad.append(f'source:{path}')
            if verify and status == 'complete':
                try:
                    verify_run(manifest.parent)
                except (ValueError, KeyError, OSError) as error:
                    bad.append(str(error))
            integrity = ('failed: '+', '.join(bad)) if bad else ('verified' if verify and data.get('artifacts') else 'not checked')
            if bad: failures.append(str(manifest))
            rows.append({'run': str(manifest.parent.relative_to(results)), 'status': status,
                         'integrity': integrity, 'manifest_sha256': file_hash(manifest),
                         'layout_version': data.get('layout_version', 1)})
        except (ValueError, KeyError, OSError) as error:
            failures.append(str(manifest))
            rows.append({'run': str(manifest.parent.relative_to(results)), 'status': 'unreadable', 'integrity': str(error)})
    # Unregistered scientific artifacts are visible even when all declared hashes pass.
    for path in results.rglob('*'):
        if path.is_file() and path not in registered and path.suffix in ('.pdf', '.png', '.npz', '.graphml', '.tsv') and path.name not in ('run_inventory.tsv', 'artifact_inventory.tsv', 'analysis_status.tsv'):
            artifacts.append({'run': '', 'path': str(path.relative_to(results)), 'sha256': '', 'integrity': 'unregistered'})
    for pointer in results.rglob('selections.json'):
        try:
            for record in json.loads(pointer.read_text()).values():
                manifest = pointer.parent/record['run']/'manifest.json'
                if file_hash(manifest) != record['manifest_sha256'] or json.loads(manifest.read_text()).get('status') != 'complete':
                    failures.append(str(pointer))
        except (OSError, ValueError, KeyError): failures.append(str(pointer))
    pd.DataFrame(rows, columns=['run', 'status', 'integrity', 'manifest_sha256', 'layout_version']).to_csv(results/'run_inventory.tsv', sep='\t', index=False)
    pd.DataFrame(artifacts, columns=['run', 'path', 'sha256', 'integrity']).to_csv(results/'artifact_inventory.tsv', sep='\t', index=False)
    text = ['# Results', '', 'PCNet full and brain networks provide the article analyses. STRING networks provide supporting interactome checks.', '']
    for network in NETWORK_ORDER:
        if (results/network).exists():
            network_readme(results, network, rows)
            text += [f'- [{network}]({network}/README.md)']
    text += ['', '- [Run inventory](run_inventory.tsv)', '- [Artifact inventory and unregistered files](artifact_inventory.tsv)', '- [Execution records and notebooks](executions/)', '']
    for network in NETWORK_ORDER:
        link = publication_link(results, network)
        if link: text.append(f'- Publication {network}: {link}')
    (results/'README.md').write_text('\n'.join(text)+'\n')
    if strict and failures:
        raise ValueError('Result verification failed: '+', '.join(sorted(set(failures))))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--results-root', type=Path, default=os.environ.get('CNV_RESULTS_ROOT') or ROOT/'results')
    args = parser.parse_args()
    build_index(verify=args.verify, results_root=args.results_root, strict=args.strict)
