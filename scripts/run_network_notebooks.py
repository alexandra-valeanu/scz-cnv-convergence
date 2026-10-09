"""Execute both notebooks on both networks, retaining executed notebooks and logs."""
import argparse
from datetime import date
import os
from pathlib import Path
import subprocess
import sys

from network_inputs import BASE_NETWORKS, NETWORKS
# Executed notebooks are written as executed_<prefix>_<notebook>_<YYYY-MM-DD>.ipynb (date of execution).
PREFIXES = {'PCNET2.2': 'pcnet2.2', 'STRING_high_conf': 'string',
            'PCNET2.2_brain': 'pcnet2.2_brain', 'STRING_high_conf_brain': 'string_brain'}
NOTEBOOKS = ('cnv_convergence', 'cnv_disorder_overlap',
             'cnv_brain_convergence', 'cnv_brain_disorder_overlap')

def resolve_workflow(network, name):
    if network not in NETWORKS or name not in NOTEBOOKS:
        raise ValueError(f'Unknown network/notebook: {network}/{name}')
    if name.startswith('cnv_brain_') and not network.endswith('_brain'):
        network += '_brain'
    if network.endswith('_brain') and not name.startswith('cnv_brain_'):
        name = name.replace('cnv_', 'cnv_brain_', 1)
    return network, name


def main():
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
        destination = root/'results/notebooks'/f'executed_{PREFIXES[network]}_{name}_{date.today().isoformat()}.ipynb'
        destination.parent.mkdir(parents=True, exist_ok=True)
        def report_cell(cell, cell_index, **kwargs):
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
    seen = set()
    for requested_network in args.networks:
        for requested_name in args.notebooks:
            network, name = resolve_workflow(requested_network, requested_name)
            if (network, name) in seen:
                continue
            seen.add((network, name))
            out = root/'results'/network
            out.mkdir(parents=True, exist_ok=True)
            log = out/f'{name}.log'
            print(f'Running {name} on {network}; log: {log}', flush=True)
            with log.open('w') as stream:
                result = subprocess.run([sys.executable, __file__, '--execute-one', network, name],
                                        cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                failures.append((network, name))
                print(f'FAILED: {network}/{name}; see {log}', flush=True)
            else:
                print(f'Completed {network}/{name}', flush=True)
    if failures:
        raise SystemExit(f'Failed notebook runs: {failures}')


if __name__ == '__main__':
    main()
