"""Execute selected network notebooks and retain executed copies and logs."""
import argparse
from datetime import datetime, timezone
from uuid import uuid4
from cnv_overlap.provenance import write_json
import os
from pathlib import Path
import subprocess
import sys

from network_inputs import BASE_NETWORKS, NETWORKS
NOTEBOOKS = ('cnv_convergence', 'cnv_disorder_overlap',
             'cnv_brain_convergence', 'cnv_brain_disorder_overlap')

def resolve_workflow(network, name):
    """Normalize a network and notebook name to the matching brain or full workflow."""
    if network not in NETWORKS or name not in NOTEBOOKS:
        raise ValueError(f'Unknown network/notebook: {network}/{name}')
    if name.startswith('cnv_brain_') and not network.endswith('_brain'):
        network += '_brain'
    if network.endswith('_brain') and not name.startswith('cnv_brain_'):
        name = name.replace('cnv_', 'cnv_brain_', 1)
    return network, name


def main():
    """Execute selected notebook workflows in subprocesses and retain outputs and logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--networks', nargs='+', choices=NETWORKS, default=list(BASE_NETWORKS))
    parser.add_argument('--notebooks', nargs='+', choices=NOTEBOOKS, default=list(NOTEBOOKS[:2]))
    parser.add_argument('--execute-one', nargs=2, metavar=('NETWORK', 'NOTEBOOK'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.execute_one:
        import nbformat
        from nbclient import NotebookClient
        network, name = resolve_workflow(*args.execute_one)
        os.environ['CNV_NETWORK'] = network
        os.environ.pop('CNV_CONFIG', None)
        notebook = nbformat.read(root/'notebooks'/f'{name}.ipynb', as_version=4)
        execution = Path(os.environ['CNV_EXECUTION_DIR'])
        destination = execution/'notebooks'/f'{name}.ipynb'
        destination.parent.mkdir(parents=True, exist_ok=True)
        def report_cell(cell, cell_index, **kwargs):
            """Print completed-cell progress and stream output to the execution log."""
            print(f'Cell {cell_index} completed', flush=True)
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'stream':
                    print(output.get('text', ''), end='', flush=True)
        try:
            NotebookClient(notebook, timeout=None, kernel_name='python3', on_cell_executed=report_cell,
                           resources={'metadata': {'path': str(root)}}).execute()
        finally:
            nbformat.write(notebook, destination)
        return
    failures = []
    results = Path(os.environ.get('CNV_RESULTS_ROOT', root/'results'))
    parent_execution = os.environ.get('CNV_EXECUTION_DIR')
    seen = set()
    for requested_network in args.networks:
        for requested_name in args.notebooks:
            network, name = resolve_workflow(requested_network, requested_name)
            if (network, name) in seen:
                continue
            seen.add((network, name))
            execution = (Path(parent_execution)/'notebook_attempts' if parent_execution else results/'executions'/network) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8])
            out = execution/'logs'
            out.mkdir(parents=True, exist_ok=True)
            log = out/f'{name}.log'
            environment = {**os.environ, 'CNV_EXECUTION_DIR': str(execution.resolve())}
            write_json(execution/'execution.json', {'status': 'running', 'network': network, 'notebook': name})
            print(f'Running {name} on {network}; log: {log}', flush=True)
            with log.open('w') as stream:
                result = subprocess.run([sys.executable, __file__, '--execute-one', network, name],
                                        cwd=root, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            write_json(execution/'execution.json', {'status': 'failed' if result.returncode else 'complete', 'network': network, 'notebook': name, 'returncode': result.returncode})
            if result.returncode:
                failures.append((network, name))
                print(f'FAILED: {network}/{name}; see {log}', flush=True)
            else:
                print(f'Completed {network}/{name}', flush=True)
    if failures:
        raise SystemExit(f'Failed notebook runs: {failures}')


if __name__ == '__main__':
    main()
