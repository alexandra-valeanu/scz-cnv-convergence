"""Plan or export a completed legacy run into the v2 layout without changing its source."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

from cnv_overlap.provenance import file_hash, run_dir, write_json
from cnv_overlap.results import _destination, finalize_manifest, verify_run


def migration_plan(source):
    source = Path(source)
    verify_run(source)
    kind = ('landscape' if (source/'01_schema_gwas_landscape').exists() else
            'pairs' if (source/'nps.tsv').exists() and (source/'hypotheses.tsv').exists() else
            'pair_network' if source.name.startswith('significant-pairs-') else
            'stability' if source.name.startswith(('nps-stability-', 'pvalue-stability-')) else None)
    return [{'old_path': str(p.relative_to(source)),
             'new_path': 'legacy_manifest.json' if p == source/'manifest.json' else _destination(str(p.relative_to(source)), kind),
             'sha256': file_hash(p)}
            for p in sorted(source.rglob('*')) if p.is_file()]


def export_run(source, destination):
    source = Path(source).resolve()
    plan = migration_plan(source)
    identity = {'migration': 'results-layout-v2', 'source': str(source),
                'source_manifest': file_hash(source/'manifest.json')}
    out = run_dir(destination, identity, 'migrated-'+source.name)
    if out.resolve() == source or source in out.resolve().parents:
        raise ValueError('Migration destination must be outside its immutable source')
    if out.exists():
        verify_run(out)
        return out
    staging = out.with_name('.staging-'+out.name)
    if staging.exists():
        raise ValueError(f'Unfinished migration exists: {staging}')
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    (staging/'manifest.json').rename(staging/'legacy_manifest.json')
    write_json(staging/'manifest.json', {'status': 'running', 'identity': identity})
    write_json(staging/'migration_map.json', plan)
    # Finalize in its permanent name so published selection paths are correct.
    staging.rename(out)
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'sources': {'legacy': {'path': str(source), 'manifest_sha256': identity['source_manifest']}}})
    verify_run(out)
    for row in plan:
        if file_hash(source/row['old_path']) != row['sha256']:
            raise ValueError('Source changed during migration')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, help='Canonical analysis home for the export')
    parser.add_argument('--apply', action='store_true', help='Create a new export; default prints a file mapping only')
    args = parser.parse_args()
    if args.apply:
        if args.destination is None: parser.error('--apply requires --destination')
        print(export_run(args.source_run, args.destination))
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=['old_path', 'new_path', 'sha256'], delimiter='\t')
        writer.writeheader()
        writer.writerows(migration_plan(args.source_run))


if __name__ == '__main__':
    main()
