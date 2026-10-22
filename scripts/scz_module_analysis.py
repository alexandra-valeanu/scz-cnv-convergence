"""Add module inference to a frozen landscape without modifying its artifacts."""
from cnv_overlap.results import analysis_from_source, resolve_artifact, finalize_manifest
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from cnv_overlap.pipeline import AnalysisConfig
from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import file_hash, run_dir, software, write_json
from scz_landscape_pipeline import conditional_null, landscape_membership
from scz_shared_network import load_inputs

ROOT = Path(__file__).resolve().parents[1]


def run(source, reps=5000, min_module_genes=5, preserve_target_overlap=True):
    source = Path(source).resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Require a completed source landscape')
    settings = manifest['identity']['settings']
    config = AnalysisConfig.from_dict(manifest['identity']['config'])
    files = ['all_gene_membership.tsv', '01_schema_gwas_landscape/gene_modules.tsv']
    for f in files:
        if file_hash(resolve_artifact(source, f)) != manifest['artifacts'][manifest.get('artifact_paths', {}).get(f, f)]:
            raise ValueError(f'Source artifact changed: {f}')
    spec, root, inputs = load_inputs(settings['network'], settings.get('min_seeds', 3),
        settings.get('convergence_config'), settings.get('disorder_config'))
    A, nodes, symbols, entities, *_ = inputs
    if entities != manifest['identity']['entities']:
        raise ValueError('Current inputs differ from frozen source entities')
    prop = Propagator(A, nodes, root/spec['cache_dir'], config.propagation)
    seeds = {n: sorted(set(e['seeds']) & prop.pos.keys()) for n, e in entities.items()}
    profiles = {n: np.asarray(prop.nps(s)) for n, s in seeds.items()}
    frozen = pd.read_csv(resolve_artifact(source, files[0]), sep='\t').set_index('entrez')
    if set(frozen.index) != set(prop.nodes):
        raise ValueError('Network gene universe differs from source')
    frozen = frozen.loc[prop.nodes]
    for n in entities:
        if not np.allclose(profiles[n], frozen[f'nps_{n}'], rtol=1e-10, atol=1e-10):
            raise ValueError(f'Propagation differs from source: {n}')
    table, _, both = landscape_membership(prop.nodes, symbols, entities, profiles, config,
                                         settings.get('primary_min_cnvs', 3))
    landscape = table.in_scz_landscape.to_numpy()
    genes = pd.read_csv(resolve_artifact(source, files[1]), sep='\t').set_index('entrez')
    if set(genes.index) != set(prop.nodes[landscape]):
        raise ValueError('Landscape differs from source')
    labels = genes.loc[prop.nodes[landscape], 'leiden_cluster']
    identity = {'algorithm': 'fixed_landscape_module_inference_v1', 'source': str(source),
                'source_manifest': file_hash(source/'manifest.json'), 'reps': reps,
                'min_module_genes': min_module_genes, 'preserve_target_overlap': preserve_target_overlap, 'config': asdict(config), 'software': software(),
                'code': {p.name: file_hash(p) for p in [Path(__file__),
                    ROOT/'scripts/scz_module_inference.py', ROOT/'scripts/scz_landscape_pipeline.py',
                    ROOT/'scripts/scz_restricted_core_null.py']},
                'package_code': {p.name: file_hash(p) for p in sorted((ROOT/'src/cnv_overlap').glob('*.py'))}}
    out = run_dir(analysis_from_source(source, 'landscape_modules'), identity, 'modules')
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        if old.get('status') == 'complete':
            for f, checksum in old['artifacts'].items():
                if file_hash(out/f) != checksum:
                    raise ValueError(f'Completed artifact changed: {f}')
            return out
    out.mkdir(parents=True, exist_ok=True)
    finalize_manifest(out/'manifest.json', {'status': 'running', 'identity': identity})
    conditional_null(prop, profiles, seeds, sorted(n for n in entities if entities[n]['kind'] == 'cnv'),
                     landscape, both, config, tuple(settings.get('core_thresholds', [1, 2, 3, 4])), reps,
                     out, symbols, module_labels=labels, min_module_genes=min_module_genes,
                     preserve_target_overlap=preserve_target_overlap)
    (out/'README.md').write_text(
        '# Conditional module inference\n\n'
        f'Frozen source: {source}\n\n'
        'Main endpoint: direct/pair_coverage in module_tests.tsv. Joint reach is sensitivity. '
        'Each mode/endpoint has its own BH and Holm family; do not pool discoveries across families. '
        'All endpoints are exploratory extensions of previously examined data. See '
        'docs/molecular_psychiatry_analysis.md for definitions, null limitations and required robustness. '
        'These results alone do not establish disease specificity or dosage direction.\n')
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity,
        'artifacts': {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*'))
                      if p.is_file() and p.name != 'manifest.json'}})
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--reps', type=int, default=5000)
    parser.add_argument('--randomize-target-overlap', action='store_true',
                        help='Sensitivity: replace even seeds shared with SCHEMA/GWAS')
    parser.add_argument('--min-module-genes', type=int, default=5)
    args = parser.parse_args()
    if args.reps < 2 or args.min_module_genes < 1:
        parser.error('Require >=2 draws and positive minimum module size')
    print(run(args.source_run, args.reps, args.min_module_genes, not args.randomize_target_overlap))
