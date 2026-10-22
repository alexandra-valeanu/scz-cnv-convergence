"""Exercise the storage contract with real writes and legacy inputs."""
import json
import os
from pathlib import Path

import pytest

from cnv_overlap.provenance import file_hash, run_dir, write_json
from cnv_overlap.results import (finalize_manifest, resolve_artifact, resolve_path,
                                 select_run, verify_run, results_dir, select_analysis)


def pair_run(home, value=1):
    run = run_dir(home, {'value': value}, 'pairs')
    run.mkdir(parents=True)
    for name in ('nps.tsv', 'hypotheses.tsv', 'pair_tests.tsv', 'seed_qc.tsv', 'candidate_genes.tsv'):
        (run/name).write_text(f'value\n{value}\n')
    finalize_manifest(run/'manifest.json', {'status': 'complete', 'identity': {'value': value}})
    return run


def test_new_pair_layout_resolves_old_keys_and_verifies(tmp_path):
    run = pair_run(tmp_path/'pair_tests')
    assert run.parent.name == 'runs'
    assert (run/'statistics/pair_tests.tsv').is_file()
    assert not (run/'pair_tests.tsv').exists()
    assert resolve_artifact(run, 'pair_tests.tsv') == run/'statistics/pair_tests.tsv'
    assert resolve_path(run/'seed_qc.tsv') == run/'qc/seed_qc.tsv'
    manifest = verify_run(run)
    assert manifest['layout_version'] == 2
    assert set(manifest['artifacts']) == {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    assert (run/'resolved_config.json').exists()
    assert (run/'README.md').exists()
    assert (run/'results_index.tsv').exists()


def test_legacy_read_does_not_move_or_modify_files(tmp_path):
    run = tmp_path/'legacy'
    run.mkdir()
    (run/'pair_tests.tsv').write_text('legacy\n')
    write_json(run/'manifest.json', {'status': 'complete', 'artifacts': {'pair_tests.tsv': file_hash(run/'pair_tests.tsv')}})
    before = file_hash(run/'manifest.json')
    assert resolve_artifact(run, 'pair_tests.tsv') == run/'pair_tests.tsv'
    verify_run(run)
    assert file_hash(run/'manifest.json') == before
    with pytest.raises(ValueError, match='reseal'):
        finalize_manifest(run/'manifest.json', {'status': 'complete'})


def test_selection_ignores_mtime_and_rejects_changed_manifest(tmp_path):
    first = pair_run(tmp_path, 1)
    second = pair_run(tmp_path, 2)
    os.utime(first/'manifest.json', (2000000000, 2000000000))
    assert select_run(tmp_path) == second
    assert select_run(tmp_path, name=first.name) == first
    (second/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='Selection manifest changed'):
        select_run(tmp_path)


def test_ambiguous_legacy_sources_are_not_guessed(tmp_path):
    pair_run(tmp_path, 1)
    pair_run(tmp_path, 2)
    (tmp_path/'selections.json').unlink()
    with pytest.raises(ValueError, match='Ambiguous'):
        select_run(tmp_path)


def test_incomplete_run_never_advances_selection(tmp_path):
    good = pair_run(tmp_path, 1)
    failed = run_dir(tmp_path, {'value': 2}, 'pairs')
    failed.mkdir()
    finalize_manifest(failed/'manifest.json', {'status': 'failed'})
    assert select_run(tmp_path) == good
    with pytest.raises(ValueError, match='Incomplete'):
        select_run(tmp_path, source=failed)


def test_landscape_export_moves_tests_and_preserves_registered_addons(tmp_path):
    run = run_dir(tmp_path, {}, 'landscape')
    files = [
        '01_schema_gwas_landscape/gene_modules.tsv',
        '01_schema_gwas_landscape/genes.tsv',
        '02_cnv_in_landscape/conditional_null/module_tests.tsv',
        '02_cnv_in_landscape/conditional_null/draws.npz',
        '02_cnv_in_landscape/gene_cnv_reach/gene_cnv_reach_table.tsv',
        'figures/cnv_module_roads/cnv_module_profile.tsv',
    ]
    for name in files:
        path = run/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    (run/'results_index.tsv').write_text('role\tresult\tpath\nmain\tModules\t'+files[2]+'\n')
    finalize_manifest(run/'manifest.json', {'status': 'complete', 'identity': {}})
    assert resolve_artifact(run, files[2]) == run/'03_module_convergence/tests/module_tests.tsv'
    assert resolve_artifact(run, files[3]) == run/'05_null_model/draws/draws.npz'
    assert resolve_artifact(run, files[4]).is_file()
    assert resolve_artifact(run, files[5]).is_file()
    assert '03_module_convergence/tests/module_tests.tsv' in (run/'results_index.tsv').read_text()
    verify_run(run)


def test_legacy_analysis_home_fallback(tmp_path):
    old = tmp_path/'1_cnv_vs_cnv/convergence'
    run = pair_run(old)
    assert select_analysis(tmp_path, 'cnv_convergence') == run
    assert results_dir(tmp_path, 'cnv_convergence') == tmp_path/'1_cnv_vs_cnv/pair_tests'


def test_resolved_landscape_config_does_not_edit_checked_in_config(tmp_path, monkeypatch):
    import run_full_results as workflow
    root = tmp_path/'project'
    (root/'configs').mkdir(parents=True)
    source = root/'configs/scz_landscape.json'
    source.write_text('{"network":"old", "null_reps":4}')
    before = source.read_bytes()
    execution = tmp_path/'execution'
    execution.mkdir()
    monkeypatch.setattr(workflow, 'ROOT', root)
    snapshot = workflow.landscape_config('PCNET2.2_brain', tmp_path/'source', execution, tmp_path/'custom/PCNET2.2_brain')
    assert source.read_bytes() == before
    settings = json.loads(snapshot.read_text())
    assert settings['null_reps'] == 4
    assert settings['output_root'] == str(tmp_path/'custom/PCNET2.2_brain/3_scz_landscape/landscape')


def test_strict_index_catches_tampering_and_lists_unregistered(tmp_path):
    import index_results
    root = tmp_path/'results'
    run = pair_run(root/'PCNET2.2/1_cnv_vs_cnv/pair_tests')
    (run/'extra.tsv').write_text('unregistered')
    index_results.build_index(results_root=root, verify=True, strict=True)
    assert 'unregistered' in (root/'artifact_inventory.tsv').read_text()
    group_readme = (root/'PCNET2.2/1_cnv_vs_cnv/README.md').read_text()
    assert 'selected; complete' in group_readme
    assert f'pair_tests/runs/{run.name}/README.md' in group_readme
    (run/'statistics/pair_tests.tsv').write_text('changed')
    with pytest.raises(ValueError, match='verification failed'):
        index_results.build_index(results_root=root, verify=True, strict=True)


def test_migration_exports_without_modifying_source(tmp_path):
    from migrate_results import migration_plan, export_run
    source = tmp_path/'old'
    source.mkdir()
    for name in ('nps.tsv', 'hypotheses.tsv', 'pair_tests.tsv'):
        (source/name).write_text('x\n1\n')
    write_json(source/'manifest.json', {'status': 'complete', 'identity': {},
        'artifacts': {p.name: file_hash(p) for p in source.iterdir()}})
    before = {p.name: file_hash(p) for p in source.iterdir()}
    plan = migration_plan(source)
    assert next(row for row in plan if row['old_path'] == 'pair_tests.tsv')['new_path'] == 'statistics/pair_tests.tsv'
    out = export_run(source, tmp_path/'new')
    assert (out/'statistics/pair_tests.tsv').read_bytes() == (source/'pair_tests.tsv').read_bytes()
    assert {p.name: file_hash(p) for p in source.iterdir()} == before
    assert export_run(source, tmp_path/'new') == out
    verify_run(out)


def test_nested_runs_do_not_publish_mutable_selections_inside_parent(tmp_path):
    outer = tmp_path/'outer'
    outer.mkdir()
    write_json(outer/'manifest.json', {'status': 'running'})
    child = pair_run(outer/'pair_test')
    assert not (outer/'pair_test/selections.json').exists()
    finalize_manifest(outer/'manifest.json', {'status': 'complete', 'identity': {}})
    verify_run(outer)
    verify_run(child)


def test_recursive_verification_rejects_modified_source(tmp_path):
    source = pair_run(tmp_path/'source')
    out = tmp_path/'report'
    out.mkdir()
    finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': {}, 'sources': {
        'input': {'path': str(source), 'manifest_sha256': file_hash(source/'manifest.json')}}})
    (source/'statistics/pair_tests.tsv').write_text('corrupt')
    with pytest.raises(ValueError, match='artifact changed'):
        verify_run(out)


def test_orchestrator_records_exact_source_and_failure(tmp_path, monkeypatch):
    import run_full_results as workflow
    root = tmp_path/'project'
    (root/'configs').mkdir(parents=True)
    config = root/'configs/scz_landscape.json'
    config.write_text('{"network":"PCNET2.2_brain", "null_reps":4}')
    results = tmp_path/'outputs'
    source = pair_run(results/'PCNET2.2_brain/1_cnv_vs_cnv/pair_tests')
    monkeypatch.setattr(workflow, 'ROOT', root)
    monkeypatch.setenv('CNV_RESULTS_ROOT', '')
    monkeypatch.setenv('CNV_EXECUTION_DIR', '')
    monkeypatch.setattr('sys.argv', ['run_full_results.py', '--network', 'PCNET2.2_brain',
        '--stages', 'landscape', '--results-root', str(results)])
    def fail(script, *args):
        settings = json.loads(Path(args[1]).read_text())
        assert settings['cnv_pair_run'] == str(source)
        assert settings['output_root'].startswith(str(results))
        raise RuntimeError('simulated stage failure')
    monkeypatch.setattr(workflow, 'python', fail)
    with pytest.raises(RuntimeError, match='simulated'):
        workflow.main()
    execution = next((results/'executions/PCNET2.2_brain').iterdir())
    assert json.loads((execution/'execution.json').read_text())['status'] == 'failed'
    assert json.loads((execution/'stage_runs.json').read_text())['cnv_convergence']['run'] == str(source)
    assert json.loads(config.read_text()) == {'network': 'PCNET2.2_brain', 'null_reps': 4}


@pytest.mark.parametrize('network,results_root', [
    ('STRING_high_conf_brain', 'results'),
    ('PCNET2.2_brain', 'other-results'),
])
def test_resume_rejects_changed_context_before_writing(tmp_path, monkeypatch, network, results_root):
    import run_full_results as workflow
    execution = tmp_path/'execution'
    execution.mkdir()
    snapshot = execution/'resolved_config.json'
    write_json(snapshot, {'network': 'PCNET2.2_brain', 'results_root': str(tmp_path/'results')})
    before = snapshot.read_bytes()
    monkeypatch.setattr('sys.argv', ['run_full_results.py', '--network', network,
        '--stages', 'index', '--resume', str(execution), '--results-root', str(tmp_path/results_root)])
    with pytest.raises(SystemExit) as error:
        workflow.main()
    assert error.value.code == 2
    assert snapshot.read_bytes() == before
    assert not (execution/'execution.json').exists()


def test_resume_requires_recorded_execution(tmp_path):
    from run_full_results import validate_resume
    with pytest.raises(ValueError, match='configuration is missing'):
        validate_resume(tmp_path/'missing', 'PCNET2.2', tmp_path/'results')
    execution = tmp_path/'execution'
    execution.mkdir()
    write_json(execution/'resolved_config.json', {'network': 'PCNET2.2', 'results_root': str(tmp_path/'results')})
    validate_resume(execution, 'PCNET2.2', tmp_path/'results')


def test_pipeline_stage_selection():
    from run_full_results import DEFAULT_STAGES, select_stages
    assert select_stages() == list(DEFAULT_STAGES)
    assert select_stages(['index', 'notebooks']) == ['notebooks', 'index']
    assert select_stages(from_stage='landscape') == ['landscape', 'cnv_reactome', 'index']
    assert select_stages(from_stage='nps_stability') == ['nps_stability', 'pvalue_stability', 'index']
    with pytest.raises(ValueError, match='cannot be combined'):
        select_stages(['index'], from_stage='landscape')
    with pytest.raises(ValueError, match='comes after'):
        select_stages(from_stage='index', to_stage='notebooks')


def test_interactive_notebook_reports_follow_results_root(tmp_path, monkeypatch):
    from cnv_overlap.results import notebook_execution_dir
    monkeypatch.delenv('CNV_EXECUTION_DIR', raising=False)
    home = tmp_path/'custom-results'
    assert notebook_execution_dir(home/'PCNET2.2', 'PCNET2.2').parent == home/'executions/PCNET2.2'
    assert notebook_execution_dir(home).parent == home/'executions/custom'
    monkeypatch.setenv('CNV_EXECUTION_DIR', str(tmp_path/'attempt'))
    assert notebook_execution_dir(home, 'PCNET2.2') == tmp_path/'attempt'


def test_seed_overview_reads_original_disorder_layout(tmp_path, monkeypatch):
    import seed_set_overview
    from test_workflow import fixture, settings
    from cnv_overlap.pipeline import run
    from dataclasses import replace
    results = tmp_path/'results'
    source = run(*fixture(), config=replace(settings(), comparisons='cnv_disorder'),
        output_root=results/'PCNET2.2/2_cnv_vs_disorders/overlap/combined',
        cache_dir=tmp_path/'cache', progress=lambda _: None)
    monkeypatch.setenv('CNV_RESULTS_ROOT', str(results))
    out = seed_set_overview.overview('PCNET2.2')
    record = json.loads((out/'source.json').read_text())
    assert record['run'] == str(source)
    verify_run(out)
