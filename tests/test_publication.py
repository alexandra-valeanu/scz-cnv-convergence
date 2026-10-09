"""Publication-specific interpretation and provenance checks."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from prepare_publication_annotations import CELL_TYPES, parse_gmx
from publication_enrichment import cluster_colors, module_names, summarize_enrichment
from publication_pipeline import export_disease_evidence, verify_run
from cnv_overlap.provenance import file_hash
from shared_network_cluster_context import CONTEXT_LIBRARIES, context_ora


def test_module_names_ignore_stronger_pathway_and_cc_hits():
    genes = pd.DataFrame({'leiden_cluster': [1, 1, 2], 'symbol': ['A', 'B', 'C']})
    hits = pd.DataFrame([
        ['C1', 'Reactome pathways', 'pathway', 1e-20, 1e-22],
        ['C1', 'GO Cellular Component', 'compartment', 1e-10, 1e-12],
        ['C1', 'GO Biological Process', 'process', .01, .001],
        ['C2', 'GO Cellular Component', 'compartment', .001, .0001]],
        columns=['cluster', 'library', 'term', 'q_group', 'p'])
    names = module_names(genes, hits).set_index('leiden_cluster')
    assert names.loc[1, 'biological_process'] == 'process'
    assert names.loc[2, 'biological_process'] == 'No enriched biological process'


def test_gmx_preserves_column_membership():
    assert parse_gmx('A\tB\nna\tna\nX\tY\nZ\t\n') == {'A': {'X', 'Z'}, 'B': {'Y'}}
    with pytest.raises(ValueError):
        parse_gmx('A\tA\nna\tna\nX\tY\n')


def test_disease_join_keeps_untested_traits_and_unsigned_dosage(tmp_path):
    pd.DataFrame([['cnv', 't2d', 'size', 'netcoloc', .01],
                  ['cnv', 'schema', 'size', 'netcoloc', .01]],
                 columns=['a', 'b', 'statistic', 'null', 'q']).to_csv(tmp_path/'pair_tests.tsv', sep='\t', index=False)
    evidence = tmp_path/'evidence.tsv'
    pd.DataFrame([['cnv', 't2d', 'deletion', 'cohort association'],
                  ['cnv', 't2d', 'duplication', 'case-level evidence'],
                  ['cnv', 'obesity', 'deletion', 'cohort association']],
                 columns=['cnv', 'target', 'literature_dosage', 'evidence_status']).to_csv(evidence, sep='\t', index=False)
    export_disease_evidence(tmp_path, evidence, tmp_path, .05)
    hits = pd.read_csv(tmp_path/'significant_disease_comparisons.tsv', sep='\t')
    assert len(hits) == 2
    assert set(hits.literature_dosage) == {'deletion', 'duplication'}
    assert hits.dosage_match.str.startswith('unresolved').all()
    review = pd.read_csv(tmp_path/'literature_to_analysis.tsv', sep='\t')
    assert not review.loc[review.target == 'obesity', 'tested_in_this_network'].any()


def test_completed_run_tampering_is_rejected(tmp_path):
    artifact = tmp_path/'result.tsv'
    artifact.write_text('original\n')
    (tmp_path/'manifest.json').write_text(json.dumps({'status': 'complete', 'artifacts': {'result.tsv': file_hash(artifact)}}))
    assert verify_run(tmp_path)
    artifact.write_text('changed\n')
    with pytest.raises(ValueError, match='changed'):
        verify_run(tmp_path)


def test_empty_annotations_keep_separate_panels(tmp_path):
    paths = {}
    for library, (_, name) in CONTEXT_LIBRARIES.items():
        paths[library] = tmp_path/name
        paths[library].write_text('term\tdescription\tA\tB\tC\tD\tE\n')
    results = context_ora({'empty': set()}, set('ABCDE'), paths, 5)
    assert (results.q_group == 1).all()
    summarize_enrichment(results, paths, tmp_path)
    assert (tmp_path/'mitocarta_3_0.pdf').exists()
    assert (tmp_path/'hpa_broad_brain_cell_type.pdf').exists()
    assert pd.read_csv(tmp_path/'orsum_mapping.tsv', sep='\t').empty


def test_broad_cell_snapshot_contains_requested_classes():
    assert {'Brain excitatory neurons', 'Brain inhibitory neurons', 'Astrocytes', 'Oligodendrocytes'} <= set(CELL_TYPES)


def test_module_palette_does_not_repeat_at_twenty():
    colors = cluster_colors(range(1, 41))
    assert len(set(colors.values())) == 40
    assert '#bbbbbb' not in colors.values()


def test_publication_workflow_on_synthetic_network(tmp_path, monkeypatch):
    import publication_pipeline as publication
    import scz_landscape_pipeline as landscape
    from test_scz_core import scz_fixture
    from test_workflow import settings
    actual_root = Path(__file__).resolve().parents[1]
    for directory in ('scripts', 'src', 'input'):
        (tmp_path/directory).symlink_to(actual_root/directory, target_is_directory=True)
    (tmp_path/'configs').mkdir()
    for name in ('cnv_convergence', 'cnv_disorder_overlap'):
        (tmp_path/f'configs/{name}.json').write_text('{}')
    inputs = scz_fixture()
    base = replace(settings(), comparisons='cnv_cnv')
    def load(path, network, **kwargs):
        cnv_only = 'convergence' in Path(path).name
        entities = {n: e for n, e in inputs[3].items() if not cnv_only or e['kind'] == 'cnv'}
        config = base if cnv_only else replace(base, comparisons='cnv_disorder')
        spec = {'analysis': asdict(config), 'cache_dir': 'cache',
                'output_root': 'results/raw/'+('pairs' if cnv_only else 'targets')}
        return spec, tmp_path, (*inputs[:3], entities, *inputs[4:]), {'network': network}
    monkeypatch.setattr(publication, 'ROOT', tmp_path)
    monkeypatch.setattr(landscape, 'ROOT', tmp_path)
    monkeypatch.setattr(publication, 'load_network_inputs', load)
    monkeypatch.setattr(landscape, 'load_inputs', lambda *a: ({'analysis': asdict(base), 'cache_dir': 'cache'}, tmp_path, inputs))
    settings_dict = json.loads((actual_root/'configs/publication.json').read_text()) | {'null_reps': 4}
    result = publication.execute_network('PCNET2.2', settings_dict, progress=lambda *a, **k: None)
    assert verify_run(result)
    assert publication.execute_network('PCNET2.2', settings_dict) == result
    assert (result/'01_cnv_pairs/pair_z_scores.pdf').exists()
    assert (result/'02_cnv_targets/enrichment/mitocarta_3_0.pdf').exists()
    manifests = list((result/'03_schizophrenia_landscape').glob('landscape-*/manifest.json'))
    assert len(manifests) == 1
    assert (manifests[0].parent/'02_cnv_in_landscape/propagation_reached_gene_cnv.tsv').exists()
