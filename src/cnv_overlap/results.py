"""Result locations, artifact verification, and named run selections."""
import json
import os
import csv
import fcntl
from pathlib import Path

GROUPS = {
    '0_inputs': 'Seed sets: how many genes each CNV / disorder set has, before and after propagation',
    '1_cnv_vs_cnv': 'Do CNVs converge on shared network neighbourhoods? (CNV x CNV pairs)',
    '2_cnv_vs_disorders': 'Which CNVs sit close to disorder gene sets? (CNV x SCHEMA / GWAS / other disorders)',
    '3_scz_landscape': 'SCHEMA x GWAS landscape: functional modules and which CNVs touch them',
    '4_cnv_reactome_profiles': 'CNV-first Reactome profiles, matched-seed calibration and SCZ overlay',
    '5_cilia': 'Do CNVs and SCZ genes sit close to cilia genes?',
    '6_robustness': 'Stability of network scores and p-values across random seeds',
    '3_shared_scz_networks': 'Archived shared-core workflow (superseded in the main sequence; files preserved)',
    '4_scz_landscape': 'Legacy landscape runs (new runs use 3_scz_landscape; sources preserved)',
}

LEGACY_FOLDERS = {
    'cnv_convergence': '1_cnv_vs_cnv/convergence',
    'cnv_cnv_report': '1_cnv_vs_cnv/report',
    'multi_cnv_convergence': '1_cnv_vs_cnv/multi_cnv_recurrence',
    'cnv_pair_network': '1_cnv_vs_cnv/pair_network',
    'enrichr_cnv_cnv': '1_cnv_vs_cnv/enrichr',
    'cnv_disorder_overlap': '2_cnv_vs_disorders/overlap',
    'cnv_disorder_report': '2_cnv_vs_disorders/report',
    'enrichr_cnv_disorder': '2_cnv_vs_disorders/enrichr',
    'scz_colocalised_networks': '3_shared_scz_networks/networks',
    'scz_network_enrichment': '3_shared_scz_networks/enrichment',
    'scz_network_orsum': '3_shared_scz_networks/enrichment_collapsed',
    'scz_core_sensitivity': '3_shared_scz_networks/core_sensitivity',
    'scz_core_figures': '3_shared_scz_networks/core_figures',
    'scz_network_leiden': '3_shared_scz_networks/leiden',
    'scz_network_cluster_context': '3_shared_scz_networks/cluster_context',
    'scz_shared_network': '3_shared_scz_networks/per_threshold',
    'scz_core_gene_test': '3_shared_scz_networks/core_gene_test',
    'scz_target_pair': '3_shared_scz_networks/target_pair',
    'schizophrenia_landscape': '4_scz_landscape',
    'cilia_colocalisation': '5_cilia',
    'nps_stability': '6_robustness/nps_stability',
    'pvalue_stability': '6_robustness/pvalue_stability',
}


FOLDERS = {**LEGACY_FOLDERS,
    'cnv_reactome_profiles': '4_cnv_reactome_profiles',
    'seed_sets': '0_inputs/seed_sets',
    'cnv_convergence': '1_cnv_vs_cnv/pair_tests',
    'cnv_cnv_report': '1_cnv_vs_cnv/reports',
    'enrichr_cnv_cnv': '1_cnv_vs_cnv/pair_enrichment',
    'cnv_disorder_overlap': '2_cnv_vs_disorders/pair_tests',
    'cnv_disorder_report': '2_cnv_vs_disorders/reports',
    'enrichr_cnv_disorder': '2_cnv_vs_disorders/pair_enrichment',
    'rare_common_pathways': '2_cnv_vs_disorders/rare_common_pathways',
    'disease_evidence': '2_cnv_vs_disorders/disease_evidence',
    'scz_colocalised_networks': '3_shared_scz_networks/shared_networks',
    'scz_network_orsum': '3_shared_scz_networks/enrichment/collapsed',
    'scz_network_leiden': '3_shared_scz_networks/modules',
    'scz_network_cluster_context': '3_shared_scz_networks/annotation_context',
    'scz_shared_network': '3_shared_scz_networks/reports/shared_networks',
    'scz_core_figures': '3_shared_scz_networks/reports/core_figures',
    'scz_core_gene_test': '3_shared_scz_networks/recurrent_core',
    'scz_pathway_convergence': '3_shared_scz_networks/pathway_convergence',
    'schizophrenia_landscape': '3_scz_landscape/landscape',
    'landscape_modules': '3_scz_landscape/module_inference',
    'landscape_pathways': '3_scz_landscape/pathway_convergence',
    'landscape_resolution': '3_scz_landscape/leiden_resolution',
    'landscape_reports': '3_scz_landscape/reports',
    'cilia_comparison': '5_cilia/comparison',
}
LAYOUT_VERSION = 2


def results_dir(network_root, analysis):
    """Canonical analysis home; run_dir adds the runs collection."""
    return Path(network_root) / FOLDERS[analysis]


def resolve_path(path):
    """Resolve an artifact path using the nearest run manifest.

    Supports original and reorganized result layouts, including unfinished runs."""
    path = Path(path)
    for parent in path.parents:
        manifest = parent / 'manifest.json'
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            relative = path.relative_to(parent).as_posix()
            mapped = data.get('artifact_paths', {}).get(relative)
            if mapped is not None:
                return parent / mapped
    return path


def resolve_artifact(run, key):
    return resolve_path(Path(run) / key)


def verify_run(run, _seen=None):
    from .provenance import file_hash
    run = Path(run).resolve()
    seen = set() if _seen is None else _seen
    if run in seen:
        raise ValueError(f'Cyclic source dependency: {run}')
    seen = seen | {run}
    data = json.loads((run / 'manifest.json').read_text())
    if data.get('status') != 'complete':
        raise ValueError(f'Incomplete source run: {run}')
    for name, checksum in data.get('artifacts', {}).items():
        path = run / name
        if not path.is_file() or file_hash(path) != checksum:
            raise ValueError(f'Completed artifact changed: {path}')
    for source in data.get('sources', {}).values():
        path = Path(source['path'])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2]/path
        if file_hash(path/'manifest.json') != source['manifest_sha256']:
            raise ValueError(f'Source manifest changed: {path}')
        verify_run(path, seen)
    return data


def select_run(home, *, name='default', source=None):
    """Resolve a named selection, or an unambiguous complete legacy run."""
    from .provenance import file_hash
    home = Path(home)
    if source is not None:
        verify_run(source)
        return Path(source)
    selections = home / 'selections.json'
    if selections.exists():
        record = json.loads(selections.read_text()).get(name)
        if record is None:
            raise ValueError(f'No selection {name!r} in {home}')
        run = home / record['run']
        if file_hash(run / 'manifest.json') != record['manifest_sha256']:
            raise ValueError(f'Selection manifest changed: {run}')
        verify_run(run)
        return run
    candidates = []
    for manifest in home.glob('**/manifest.json'):
        if any((p / 'manifest.json').exists() for p in manifest.parent.parents if p != home and p.is_relative_to(home)):
            continue
        if json.loads(manifest.read_text()).get('status') == 'complete':
            candidates.append(manifest.parent)
    if not candidates:
        raise FileNotFoundError(f'No complete run under {home}')
    if len(candidates) != 1:
        raise ValueError(f'Ambiguous runs under {home}; supply an explicit source or named selection')
    verify_run(candidates[0])
    return candidates[0]


def publish_selection(run, name='default'):
    from .provenance import file_hash, write_json
    run = Path(run)
    home = run.parent.parent if run.parent.name == 'runs' else run.parent
    # A nested source belongs to its enclosing immutable run, not a mutable collection.
    if any((parent/'manifest.json').is_file() for parent in home.parents):
        return
    path = home / 'selections.json'
    with (home/'.selections.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = json.loads(path.read_text()) if path.exists() else {}
        records[name] = {'run': os.path.relpath(run, home),
                         'manifest_sha256': file_hash(run / 'manifest.json')}
        write_json(path, records)
        lines = ['# Analysis runs', '', 'Named selections point to verified completed runs. Each run records its settings and sources.', '',
                 '| Selection | Results |', '| --- | --- |']
        for label, record in sorted(records.items(), key=lambda item: (item[0] != 'default', item[0])):
            lines.append(f"| {label} | [{record['run']}]({record['run']}/README.md) |")
        (home/'README.md').write_text('\n'.join(lines)+'\n')


def _destination(key, kind):
    p = Path(key)
    if kind == 'landscape':
        if key in ('all_gene_membership.tsv', 'seed_qc.tsv'):
            return '00_provenance_qc/' + key
        if key.startswith('figures/cnv_module_roads/'):
            return key.replace('figures/cnv_module_roads/', '03_module_convergence/cnv_module_roads/', 1)
        if key.startswith('01_schema_gwas_landscape/'):
            tail = key.split('/', 1)[1]
            if '/' in tail:
                return key
            section = ('modules' if p.stem in ('gene_modules', 'modules', 'module_names', 'modules_all_labels')
                       else 'network')
            if p.stem == 'pair_tests': section = 'pair_test'
            if p.stem == 'cnv_recurrence': return '02_cnv_coverage/recurrence/' + p.name
            return f'01_schema_gwas_landscape/{section}/{p.name}'
        if key.startswith('02_cnv_in_landscape/'):
            tail = key.split('/', 1)[1]
            if tail.startswith('conditional_null/'):
                name = p.name
                if name.endswith('.npz'): section = '05_null_model/draws'
                elif name in ('excluded_random_seeds.tsv', 'target_overlap_qc.tsv', 'degree_pool_qc.tsv'): section = '05_null_model/qc'
                elif name.startswith('module_'): section = '03_module_convergence/tests'
                elif name.startswith('cnv_coverage'): section = '02_cnv_coverage/landscape_coverage'
                elif name.startswith('core_size'): section = '04_recurrent_core/core_size_tests'
                else: section = '04_recurrent_core/gene_identity_tests'
                return section + '/' + name
            if tail.startswith(('core_min', 'enrichment/')):
                return '04_recurrent_core/' + tail
            if tail.startswith('gene_cnv_reach/'):
                return '02_cnv_coverage/' + tail
            if p.stem.startswith(('core_gene', 'unrestricted_primary')):
                return '04_recurrent_core/' + p.name
            if p.stem == 'cnv_module_coverage': return '03_module_convergence/coverage/' + p.name
            section = 'recurrence' if p.stem == 'recurrence_distribution' else 'landscape_coverage'
            return f'02_cnv_coverage/{section}/{p.name}'
    elif kind == 'pair_network' and len(p.parts) == 1:
        if p.stem in ('significant_pairs', 'cnv_pair_graph'): return 'significant_pairs/' + key
        if p.stem in ('genes', 'edges', 'network', 'gene_membership'): return 'gene_network/' + key
        if p.stem in ('clusters', 'excluded_singletons', 'leiden_clusters', 'cluster_pair_composition'): return 'modules/' + key
        if p.stem in ('context_enrichment', 'context_enrichment_significant', 'cluster_context', 'tissue_celltype_enrichment', 'background_genes'): return 'enrichment/' + key
    elif kind == 'pairs' and len(p.parts) == 1:
        if p.name in ('seed_qc.tsv', 'seed_overlap.tsv'): return 'qc/' + key
        if p.name in ('nps.tsv', 'candidate_genes.tsv', 'recurrence.tsv'): return 'gene_membership/' + key
        if p.name in ('pair_tests.tsv', 'hypotheses.tsv'): return 'statistics/' + key
        if p.name.startswith('netcoloc-'): return 'null_model/draws/' + key
    elif kind == 'stability' and len(p.parts) == 1:
        if p.name.startswith(('nps_seed', 'membership_seed', 'pair_tests_permutation')): return 'per_setting/' + key
        if p.name in ('seed_qc.tsv', 'reference_counts.tsv', 'recompute_check.tsv'): return 'qc/' + key
        if p.suffix in ('.tsv', '.gz', '.png', '.pdf'): return 'summary/' + key
    return key


def finalize_manifest(path, data):
    """Seal a newly assembled run in v2 layout; retain v1 artifact keys for readers.

    Called explicitly by producers. Completed source runs are never migrated here.
    """
    from .provenance import file_hash, write_json
    path = Path(path)
    if data.get('status') != 'complete':
        return write_json(path, data)
    out = path.parent
    old = json.loads(path.read_text()) if path.exists() else {}
    if old.get('status') == 'complete':
        raise ValueError(f'Cannot reseal a completed run: {out}')
    kind = ('landscape' if (out/'01_schema_gwas_landscape').exists() else
            'pairs' if (out/'nps.tsv').exists() and (out/'hypotheses.tsv').exists() else
            'pair_network' if out.name.startswith('significant-pairs-') else
            'stability' if out.name.startswith(('nps-stability-', 'pvalue-stability-')) else None)
    identity = data.get('identity', {})
    sources = dict(data.get('sources', {}))
    for key in ('source', 'network_dir', 'sensitivity_dir', 'enrichment_dir'):
        reference = identity.get(key, data.get(key))
        if isinstance(reference, str) and (Path(reference)/'manifest.json').is_file():
            source = Path(reference).resolve()
            sources.setdefault(key, {'path': str(source), 'manifest_sha256': file_hash(source/'manifest.json')})
    for group in ('null_dirs', 'pairwise_runs'):
        for label, reference in data.get(group, {}).items():
            source = Path(reference).resolve()
            sources.setdefault(f'{group}.{label}', {'path': str(source), 'manifest_sha256': file_hash(source/'manifest.json')})
    data = {**data, 'sources': sources}
    write_json(out/'resolved_config.json', identity.get('settings', identity.get('config', identity)))
    files = [p for p in out.rglob('*') if p.is_file() and p != path and not p.name.startswith('.writing-')]
    mapping = {p.relative_to(out).as_posix(): _destination(p.relative_to(out).as_posix(), kind) for p in files}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f'Artifact path collision in {out}')
    # Rewrite generated relative links before hashing. Nested manifests stay intact.
    for p in files:
        if p.parent == out and p.name in ('README.md', 'results_index.tsv'):
            text = p.read_text()
            for before, after in sorted(mapping.items(), key=lambda item: -len(item[0])):
                if before != after: text = text.replace(before, after)
            if kind == 'landscape':
                text = text.replace('`02_cnv_in_landscape`', '`02_cnv_coverage`, `03_module_convergence`, `04_recurrent_core`, `05_null_model`')
            p.write_text(text)
    for before, after in mapping.items():
        if before != after:
            dest = out / after
            dest.parent.mkdir(parents=True, exist_ok=True)
            (out / before).rename(dest)
    for directory in sorted((p for p in out.rglob('*') if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(directory.iterdir()): directory.rmdir()
    index = out/'results_index.tsv'
    if not index.exists():
        with index.open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['role', 'result', 'path'])
            for name in sorted(mapping.values()):
                writer.writerow(['figure' if Path(name).suffix in ('.png', '.pdf', '.svg') else 'data', Path(name).stem.replace('_', ' '), name])
        mapping['results_index.tsv'] = 'results_index.tsv'
    readme = out/'README.md'
    if not readme.exists():
        readme.write_text(f'# {out.name}\n\n[Results index](results_index.tsv) · [Resolved configuration](resolved_config.json)\n\n'
                          'Exact artifact checksums and source identities are recorded in `manifest.json`.\n')
        mapping['README.md'] = 'README.md'
    data = {**data, 'layout_version': LAYOUT_VERSION, 'artifact_paths': mapping, 'analysis': data.get('analysis', kind or out.name.rsplit('-', 1)[0]),
            'artifacts': {name: file_hash(out/name) for name in mapping.values()}}
    write_json(path, data)
    publish_selection(out, name=out.name)
    publish_selection(out)


def network_root(network, default='results'):
    """Honour a workflow-wide results override in child processes."""
    return Path(os.environ.get('CNV_RESULTS_ROOT', default)) / network


def notebook_execution_dir(results_root, network=None):
    """Keep interactive reports beside their configured results, including fixtures."""
    from uuid import uuid4
    if os.environ.get('CNV_EXECUTION_DIR'):
        return Path(os.environ['CNV_EXECUTION_DIR']).resolve()
    results_root = Path(results_root).resolve()
    home = results_root.parent if network else results_root
    return home / 'executions' / (network or 'custom') / uuid4().hex


def analysis_from_source(source, analysis):
    """Locate a derived analysis using the source's recorded network."""
    source = Path(source)
    data = json.loads((source/'manifest.json').read_text())
    identity = data.get('identity', {})
    settings = identity.get('settings', {})
    network = settings.get('network') or identity.get('network')
    if network is None:
        network = next((parent.name for parent in source.parents if parent.name in ('PCNET2.2', 'PCNET2.2_brain', 'STRING_high_conf', 'STRING_high_conf_brain')), None)
    if network:
        for parent in source.parents:
            if parent.name == network:
                return results_dir(parent, analysis)
        return results_dir(network_root(network), analysis)
    raise ValueError(f'Source does not identify its network: {source}')


def report_run(source, label, *, settings=None, output_root=None):
    from .provenance import file_hash, run_dir, software
    source = Path(source)
    verify_run(source)
    identity = {'source': str(source.resolve()), 'source_manifest': file_hash(source/'manifest.json'),
                'report': label, 'settings': settings or {}, 'software': software()}
    home = Path(output_root) if output_root else analysis_from_source(source, 'landscape_reports')
    out = run_dir(home, identity, label)
    if (out/'manifest.json').exists():
        verify_run(out)
        return out, identity, True
    out.mkdir(parents=True, exist_ok=True)
    finalize_manifest(out/'manifest.json', {'status': 'running', 'identity': identity})
    return out, identity, False


def reuse_completed(out):
    path = Path(out)/'manifest.json'
    if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
        verify_run(out)
        publish_selection(out)
        return True
    return False


def select_analysis(network, analysis, *, name='default', source=None):
    home = results_dir(network, analysis)
    if home.exists() or source is not None:
        return select_run(home, name=name, source=source)
    # Before the CNV-first workflow, landscape collections already had named
    # selections one level below 4_scz_landscape. Honour those before scanning
    # the older group, which can also contain several historical runs.
    previous = {
        'schizophrenia_landscape': '4_scz_landscape/landscape',
        'landscape_modules': '4_scz_landscape/module_inference',
        'landscape_pathways': '4_scz_landscape/pathway_convergence',
        'landscape_resolution': '4_scz_landscape/leiden_resolution',
        'landscape_reports': '4_scz_landscape/reports',
    }.get(analysis)
    if previous and (Path(network)/previous).exists():
        return select_run(Path(network)/previous, name=name)
    legacy = LEGACY_FOLDERS.get(analysis)
    if legacy:
        return select_run(Path(network)/legacy, name=name)
    raise FileNotFoundError(home)
