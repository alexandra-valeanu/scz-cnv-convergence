"""Run network notebooks and downstream analyses with execution records."""
import argparse
import json
import os
import fcntl
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
import subprocess
import sys

from network_inputs import NETWORKS
from results_layout import results_dir
from cnv_overlap.results import select_run, select_analysis
from cnv_overlap.provenance import write_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGES = ('notebooks', 'seed_sets', 'pair_network', 'landscape', 'cnv_reactome', 'index')
STAGES = (*DEFAULT_STAGES[:-1], 'cilia', 'nps_stability', 'pvalue_stability', 'index')
NOTEBOOKS = ('cnv_convergence', 'cnv_disorder_overlap')


def select_stages(stages=None, from_stage=STAGES[0], to_stage=STAGES[-1]):
    """Return requested stages in execution order, validating range options."""
    if stages:
        if (from_stage, to_stage) != (STAGES[0], STAGES[-1]):
            raise ValueError('--stages cannot be combined with --from-stage/--to-stage')
        return [stage for stage in STAGES if stage in stages]
    if STAGES.index(from_stage) > STAGES.index(to_stage):
        raise ValueError(f'--from-stage {from_stage} comes after --to-stage {to_stage}')
    sequence = DEFAULT_STAGES
    if from_stage not in DEFAULT_STAGES or to_stage not in DEFAULT_STAGES:
        sequence = STAGES
    start = sequence.index(from_stage)
    stop = sequence.index(to_stage) + 1
    return list(sequence[start:stop])


def validate_resume(execution, network, results_root):
    """Require an existing execution with the same network and result root."""
    path = execution/'resolved_config.json'
    if not path.is_file():
        raise ValueError(f'Resume configuration is missing: {path}')
    settings = json.loads(path.read_text())
    if settings['network'] != network:
        raise ValueError(f'Resume network differs: {settings["network"]} != {network}')
    if Path(settings['results_root']).resolve() != results_root.resolve():
        raise ValueError('Resume results root differs from the recorded execution')


def python(script, *args):
    """Run one project script with the current interpreter; stop the pipeline if it fails."""
    command = [sys.executable, '-u', str(ROOT/'scripts'/script), *map(str, args)]
    print('\n$', ' '.join(command[2:]), flush=True)
    execution = os.environ.get('CNV_EXECUTION_DIR')
    if execution:
        logs = Path(execution)/'logs'
        logs.mkdir(parents=True, exist_ok=True)
        log = logs/(Path(script).stem+'-'+uuid4().hex[:8]+'.log')
        print(f'Log: {log}', flush=True)
        with log.open('w') as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, cwd=ROOT, check=True)


def landscape_config(network, pair_run, execution, base):
    """Save resolved landscape settings in the execution directory."""
    path = ROOT/'configs'/('scz_landscape.json' if network == 'PCNET2.2_brain' else f'scz_landscape_{network}.json')
    settings = json.loads((path if path.is_file() else ROOT/'configs/scz_landscape.json').read_text())
    settings.update(network=network, cnv_pair_run=str(pair_run.resolve()),
                    output_root=str(results_dir(base, 'schizophrenia_landscape')))
    path = execution/'landscape_config.json'
    write_json(path, settings)
    return path


def main():
    """Execute the selected network stages in order, stopping on subprocess failure."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--network', required=True, choices=NETWORKS)
    parser.add_argument('--stages', nargs='+', choices=STAGES, help='Run only these stages (in pipeline order)')
    parser.add_argument('--from-stage', choices=STAGES, default=STAGES[0], help='Resume from this stage')
    parser.add_argument('--to-stage', choices=STAGES, default=STAGES[-1], help='Stop after this stage')
    parser.add_argument('--notebooks', nargs='+', choices=NOTEBOOKS, default=list(NOTEBOOKS),
                        help='Notebooks for the notebooks stage; brain variants are picked for *_brain networks')
    parser.add_argument('--plan', action='store_true', help='Print the selected stages without running them')
    parser.add_argument('--results-root', type=Path, default=os.environ.get('CNV_RESULTS_ROOT') or ROOT/'results')
    parser.add_argument('--resume', type=Path, help='Resume an execution using its exact recorded sources')
    parser.add_argument('--cnv-pair-run', type=Path, help='Explicit source when running downstream stages alone')
    args = parser.parse_args()
    try:
        stages = select_stages(args.stages, args.from_stage, args.to_stage)
        if args.resume:
            validate_resume(args.resume, args.network, args.results_root)
    except ValueError as error:
        parser.error(str(error))
    network, base = args.network, args.results_root.resolve()/args.network
    if args.plan:
        print(json.dumps({'network': network, 'stages': list(stages), 'notebooks': args.notebooks}))
        return
    execution = args.resume or args.results_root.resolve()/'executions'/network/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
    execution.mkdir(parents=True, exist_ok=True)
    os.environ['CNV_RESULTS_ROOT'] = str(args.results_root.resolve())
    os.environ['CNV_EXECUTION_DIR'] = str(execution.resolve())
    records_path = execution/'stage_runs.json'
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    write_json(execution/'resolved_config.json', {'network': network, 'stages': list(stages), 'notebooks': args.notebooks, 'results_root': str(args.results_root.resolve())})
    def selected(analysis):
        if analysis in records:
            record = records[analysis]
            source = Path(record['run'])
            if file_hash(source/'manifest.json') != record['manifest_sha256']:
                raise ValueError(f'Recorded source changed: {source}')
            return select_run(results_dir(base, analysis), source=source)
        explicit = args.cnv_pair_run if analysis == 'cnv_convergence' else None
        source = select_analysis(base, analysis, source=explicit)
        records[analysis] = {'run': str(source.resolve()), 'manifest_sha256': file_hash(source/'manifest.json')}
        write_json(records_path, records)
        return source
    write_json(execution/'execution.json', {'status': 'running', 'network': network, 'stages': list(stages)})
    base.mkdir(parents=True, exist_ok=True)
    with (base/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for stage in stages:
                print(f'\n=== {network}: {stage}', flush=True)
                if stage == 'notebooks':
                    # The runner picks the brain notebooks for *_brain networks.
                    python('run_network_notebooks.py', '--networks', network, '--notebooks', *args.notebooks)
                    for analysis in args.notebooks:
                        records.pop(analysis, None)
                        selected(analysis)
                elif stage == 'seed_sets':
                    python('seed_set_overview.py', '--network', network, '--source', selected('cnv_disorder_overlap'))
                elif stage == 'pair_network' and not network.endswith('_brain'):
                    # The brain convergence notebook already builds the pair network.
                    python('cnv_pair_network.py', '--network', network,
                           '--run', selected('cnv_convergence'))
                elif stage == 'landscape':
                    pair_run = selected('cnv_convergence')
                    python('scz_landscape_pipeline.py', '--config', landscape_config(network, pair_run, execution, base))
                    records.pop('schizophrenia_landscape', None)
                    selected('schizophrenia_landscape')
                elif stage == 'cnv_reactome':
                    python('cnv_reactome_profiles.py', '--source-run', selected('schizophrenia_landscape'),
                           '--output-root', results_dir(base, 'cnv_reactome_profiles'))
                elif stage == 'cilia':
                    python('cilia_colocalization.py', '--network', network)
                    python('plot_cilia_colocalisation.py', '--network', network)
                elif stage in ('nps_stability', 'pvalue_stability'):
                    python(stage+'.py', '--networks', network)
                elif stage == 'index':
                    python('index_results.py', '--results-root', args.results_root.resolve(), '--verify', '--strict')
        except Exception as error:
            write_json(execution/'execution.json', {'status': 'failed', 'network': network, 'stage': stage, 'error': f'{type(error).__name__}: {error}'})
            raise
    write_json(execution/'execution.json', {'status': 'complete', 'network': network, 'stages': list(stages)})
    print(f'\nDone: {base}', flush=True)


if __name__ == '__main__':
    main()
