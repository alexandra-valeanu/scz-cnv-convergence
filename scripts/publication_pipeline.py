"""Run the publication analyses on explicitly selected networks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

import pandas as pd

from cnv_overlap.pipeline import AnalysisConfig, run
from cnv_overlap.provenance import file_hash, run_dir, software, write_json
from cnv_overlap.visualization import pair_heatmap
from network_inputs import NETWORKS, load_network_inputs
from scz_landscape_pipeline import enrich, run_pipeline, plot_cnv_pairs
from cnv_pair_network import significant_pairs
from shared_network_cluster_context import _context_paths
from publication_enrichment import analysis_tools

ROOT = Path(__file__).resolve().parents[1]


def verify_run(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        return False
    for name, expected in manifest.get('artifacts', {}).items():
        if file_hash(path/name) != expected:
            raise ValueError(f'Completed artifact changed: {path/name}')
    for source in manifest.get('sources', {}).values():
        source_path = ROOT/source['path']
        if file_hash(source_path/'manifest.json') != source['manifest_sha256'] or not verify_run(source_path):
            raise ValueError(f'Source run changed: {source_path}')
    return True


def export_disease_evidence(source, evidence_path, out, fdr):
    tests = pd.read_csv(source/'pair_tests.tsv', sep='\t')
    tests = tests[(tests.statistic == 'size') & (tests['null'] == 'netcoloc')
                  & ~tests.b.isin(['schema', 'gwaspri'])].copy()
    tests['significant'] = tests.q < fdr
    evidence = pd.read_csv(evidence_path, sep='\t', dtype=str).fillna('')
    merged = tests.merge(evidence, left_on=['a', 'b'], right_on=['cnv', 'target'], how='left')
    merged['evidence_status'] = merged.evidence_status.fillna('not reviewed; no absence-of-evidence claim')
    merged['dosage_match'] = 'unresolved: propagated CNV is unsigned'
    merged.to_csv(out/'all_disease_comparisons.tsv', sep='\t', index=False)
    merged[merged.significant].to_csv(out/'significant_disease_comparisons.tsv', sep='\t', index=False)
    merged[merged.significant & merged.evidence_status.str.contains('not reviewed|lead', case=False)].to_csv(
        out/'literature_review_queue.tsv', sep='\t', index=False)
    tested_keys = set(zip(tests.a, tests.b))
    evidence['tested_in_this_network'] = [(a, b) in tested_keys for a, b in zip(evidence.cnv, evidence.target)]
    evidence.to_csv(out/'literature_to_analysis.tsv', sep='\t', index=False)


def _execute_network(network, settings, progress=print):
    import matplotlib.pyplot as plt
    paths = _context_paths(ROOT/'input/annotations')
    identity = {'network': network, 'settings': settings,
        'analysis_tools': analysis_tools(),
        'software': software(),
        'configs': {k: file_hash(ROOT/settings[k]) for k in ('convergence_config', 'disorder_config')},
        'code': {str(p.relative_to(ROOT)): file_hash(p) for folder in ('scripts', 'src/cnv_overlap')
                 for p in sorted((ROOT/folder).glob('*.py'))},
        'annotations': {k: file_hash(p) for k, p in paths.items()},
        'disease_evidence': file_hash(ROOT/settings['disease_evidence'])}
    out = run_dir(ROOT/'results/publication'/network, identity, 'publication')
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and verify_run(out):
        write_json(out.parent/'current.json', {'run': out.name, 'manifest_sha256': file_hash(manifest_path)})
        return out
    write_json(manifest_path, {'status': 'running', 'identity': identity})
    try:
        sources = {}
        for stage, key in [('01_cnv_pairs', 'convergence_config'), ('02_cnv_targets', 'disorder_config')]:
            progress(f'{network}: {stage}', flush=True)
            spec, root, inputs, provenance = load_network_inputs(ROOT/settings[key], network,
                min_seeds=settings['min_seeds'], target_scope=settings['disease_target_scope'])
            config = AnalysisConfig.from_dict(spec['analysis'])
            if config.overlap != 'bin':
                raise ValueError('Publication enrichment requires overlap=bin')
            source = run(*inputs, config=config, output_root=root/spec['output_root'],
                         cache_dir=root/spec['cache_dir'], provenance=provenance, progress=progress)
            folder = out/stage
            folder.mkdir(exist_ok=True)
            sources[stage] = {'path': str(source.relative_to(ROOT)),
                              'manifest_sha256': file_hash(source/'manifest.json')}
            write_json(folder/'source.json', sources[stage])
            relative_source = os.path.relpath(source, folder)
            (folder/'README.md').write_text('# Pair analysis\n\n'
                f'- [Propagation NPS profiles]({relative_source}/nps.tsv)\n'
                f'- [Colocalized candidate genes]({relative_source}/candidate_genes.tsv)\n'
                f'- [Seed QC]({relative_source}/seed_qc.tsv)\n'
                f'- [Source manifest]({relative_source}/manifest.json)\n')
            shutil.copy2(source/'seed_qc.tsv', folder/'seed_qc.tsv')
            qc = pd.read_csv(folder/'seed_qc.tsv', sep='\t')
            qc['sparse_mapping'] = qc.retained_seeds < 3
            qc.to_csv(folder/'seed_qc.tsv', sep='\t', index=False)
            pd.DataFrame([{'entity': n, 'mapped_seeds': count, 'reason': 'below min_seeds'}
                          for n, count in provenance.get('excluded_below_min_seeds', {}).items()],
                         columns=['entity', 'mapped_seeds', 'reason']).to_csv(folder/'excluded_seed_sets.tsv', sep='\t', index=False)
            figure = pair_heatmap(source, family='region_pair' if key == 'convergence_config' else 'locus_target',
                                  metric='z', fdr=config.fdr)
            for ext in ('pdf', 'png'):
                figure.savefig(folder/f'pair_z_scores.{ext}', dpi=180, bbox_inches='tight')
            plt.close(figure)
            tests = pd.read_csv(source/'pair_tests.tsv', sep='\t')
            tests.to_csv(folder/'pair_tests.tsv', sep='\t', index=False)
            if key == 'convergence_config':
                pairs = significant_pairs(source, fdr=config.fdr)
                pairs.to_csv(folder/'significant_pairs.tsv', sep='\t', index=False)
                plot_cnv_pairs(pairs, sorted(inputs[3]), folder)
            else:
                candidates = pd.read_csv(source/'candidate_genes.tsv', sep='\t')
                nps = pd.read_csv(source/'nps.tsv', sep='\t')
                background = {inputs[2].get(int(g), str(g)) for g in nps.entrez}
                queries = {}
                for cnv, entity in sorted(inputs[3].items()):
                    if entity['kind'] != 'cnv':
                        continue
                    for target in ('schema', 'gwaspri'):
                        if target not in inputs[3]:
                            raise ValueError(f'Missing required target {target} in {network}')
                        rows = candidates[(candidates.a == cnv) & (candidates.b == target)]
                        queries[f'{cnv} × {target}'] = set(rows.symbol.astype(str))
                enrich(queries, background, paths, folder/'enrichment', config.fdr, 'CNV × SCHEMA and CNV × GWAS')
                disease = out/'04_cnv_disease'
                disease.mkdir(exist_ok=True)
                export_disease_evidence(source, ROOT/settings['disease_evidence'], disease, config.fdr)
                if (~tests.b.isin(['schema', 'gwaspri'])).any():
                    figure = pair_heatmap(source, family='locus_target', metric='z', fdr=config.fdr,
                                          exclude=['schema', 'gwaspri'])
                    for ext in ('png', 'pdf'):
                        figure.savefig(disease/f'disease_z_scores.{ext}', dpi=180, bbox_inches='tight')
                    plt.close(figure)
            write_json(manifest_path, {'status': 'running', 'identity': identity, 'sources': sources})
        landscape = run_pipeline({**{k: settings[k] for k in ('min_seeds', 'core_thresholds', 'primary_min_cnvs',
            'null_reps', 'leiden_resolution', 'min_cluster_genes', 'convergence_config', 'disorder_config')}, 'network': network,
            'cnv_pair_run': sources['01_cnv_pairs']['path'],
            'output_root': str((out/'03_schizophrenia_landscape').relative_to(ROOT))}, progress=progress)
        sources['03_schizophrenia_landscape'] = {'path': str(landscape.relative_to(ROOT)),
                                               'manifest_sha256': file_hash(landscape/'manifest.json')}
        supplementary = out/'05_supplementary'
        supplementary.mkdir(exist_ok=True)
        rows = []
        for analysis in ('nps_stability', 'pvalue_stability'):
            for manifest in sorted((ROOT/'results'/network/analysis).glob('**/manifest.json')):
                data = json.loads(manifest.read_text())
                rows.append({'analysis': analysis, 'status': data.get('status', 'unknown'),
                             'manifest': str(manifest.relative_to(ROOT)), 'sha256': file_hash(manifest)})
        pd.DataFrame(rows, columns=['analysis', 'status', 'manifest', 'sha256']).to_csv(
            supplementary/'stability_runs.tsv', sep='\t', index=False)
        (out/'README.md').write_text(f'# {network}: publication results\n\n'
            '1. `01_cnv_pairs/`: z-score heatmap and significant-pair circle; width is observed-minus-expected overlap.\n'
            '2. `02_cnv_targets/`: CNV × target tests; SCHEMA/GWAS enrichment, separate annotation panels and orsum.\n'
            '3. `03_schizophrenia_landscape/`: SCHEMA × GWAS landscape, BP-named Leiden modules and CNV coverage.\n'
            '4. `04_cnv_disease/`: all disease tests joined to dosage-specific literature evidence.\n'
            '5. `05_supplementary/`: references to preserved NPS and p-value stability runs.\n\n'
            'Exact source paths and checksums are in `manifest.json`. Missing stability runs are not completed by this workflow.\n')
        pd.DataFrame([{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size}
                      for p in sorted(out.rglob('*')) if p.is_file()]).to_csv(out/'results_index.tsv', sep='\t', index=False)
        write_json(manifest_path, {'status': 'complete', 'identity': identity, 'sources': sources,
            'artifacts': {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file() and p != manifest_path}})
        write_json(out.parent/'current.json', {'run': out.name, 'manifest_sha256': file_hash(manifest_path)})
        return out
    except Exception as error:
        write_json(manifest_path, {'status': 'failed', 'identity': identity,
                                  'error': f'{type(error).__name__}: {error}'})
        raise


def execute_network(network, settings, progress=print):
    directory = ROOT/'results/publication'/network
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _execute_network(network, settings, progress)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/publication.json')
    parser.add_argument('--networks', nargs='+', choices=NETWORKS)
    parser.add_argument('--plan', action='store_true', help='Print selected networks and settings without computation')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    networks = args.networks or settings['networks']
    if args.plan:
        print(json.dumps({'networks': networks, 'settings': settings}, indent=2))
        return
    import matplotlib
    matplotlib.use('Agg')
    for network in networks:
        try:
            print(execute_network(network, settings), flush=True)
        finally:
            from index_results import build_index
            build_index()


if __name__ == '__main__':
    main()
