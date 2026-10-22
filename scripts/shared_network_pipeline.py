"""Build shared CNV networks, core sensitivity tests, and descriptive enrichment."""
import argparse
from pathlib import Path

from cnv_overlap.inputs import load_inputs
from cnv_overlap.pipeline import AnalysisConfig
from cnv_overlap.provenance import graph_hash
from notebook_analysis import SCZ_TARGETS, export_scz_networks
from shared_network_enrichment import LIBRARIES, enrich_shared_network
from shared_network_orsum import collapse_enrichment
from shared_core_sensitivity import run_core_sensitivity
from results_layout import results_dir


def run_shared_network_pipeline(inputs, *, config, cache_dir, output_root,
                                annotation_dir, min_cnvs=3, enrich=True, collapse=True, orsum_executable="orsum.py",
                                sensitivity=True, sensitivity_reps=None, progress=print):
    """Export graphs, core sensitivity with a joint null, and descriptive enrichment.

    Independent content-addressed directories allow changes to enrichment libraries
    to reuse the network exports. No downloads take place during pipeline execution.
    """
    if enrich:
        missing = [str(Path(annotation_dir)/n) for n in LIBRARIES.values()
                   if not (Path(annotation_dir)/n).exists()]
        if missing:
            raise FileNotFoundError('Missing annotation snapshots: ' + ', '.join(missing))
        targets = {n for n, e in inputs[3].items() if e['kind'] == 'target'}
        if not set(SCZ_TARGETS) <= targets:
            raise ValueError('Shared-core enrichment requires both schema and gwaspri targets')
    networks = export_scz_networks(inputs, config=config, cache_dir=cache_dir,
        output_root=results_dir(output_root, 'scz_colocalised_networks'), min_cnvs=min_cnvs, progress=progress)
    enrichment = None
    orsum = None
    core_sensitivity = None
    if sensitivity and all(n in inputs[3] and inputs[3][n]['kind'] == 'target' for n in SCZ_TARGETS):
        core_sensitivity = run_core_sensitivity(inputs, config=config, cache_dir=cache_dir,
            output_root=results_dir(output_root, 'scz_core_sensitivity'), primary_min_cnvs=min_cnvs,
            reps=sensitivity_reps, annotation_dir=annotation_dir if enrich else None, progress=progress)
        progress(f'Core sensitivity and conditional null: {core_sensitivity}')
    if enrich:
        enrichment = enrich_shared_network(networks, annotation_dir=annotation_dir,
            output_root=results_dir(output_root, 'scz_network_enrichment'), fdr=config.fdr)
        progress(f'Shared-core enrichment: {enrichment}')
        if collapse:
            orsum = collapse_enrichment(enrichment, output_root=results_dir(output_root, 'scz_network_orsum'), executable=orsum_executable)
            progress(f'Collapsed enrichment: {orsum}')
    return {'networks': networks, 'enrichment': enrichment, 'orsum': orsum, 'sensitivity': core_sensitivity}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--target-config', help='Optional validated SCHEMA/GWAS snapshot for a CNV-only config')
    parser.add_argument('--output-root', help='Defaults to <config root>/results')
    parser.add_argument('--annotation-dir', help='Defaults to <config root>/input/annotations')
    parser.add_argument('--min-cnvs', type=int, default=3)
    parser.add_argument('--skip-enrichment', action='store_true', help='Skip annotation analyses; core sensitivity still runs unless skipped separately')
    parser.add_argument('--orsum-executable', default='orsum.py')
    parser.add_argument('--skip-orsum', action='store_true')
    parser.add_argument('--skip-sensitivity', action='store_true')
    parser.add_argument('--sensitivity-reps', type=int, default=None)
    parser.add_argument('--network', default='PCNET2.2_brain')
    args = parser.parse_args(argv)
    # CLI exports run without a graphical display; notebook callers keep their backend.
    import matplotlib
    matplotlib.use('Agg')
    from network_inputs import load_network_inputs
    spec, root, inputs, _ = load_network_inputs(args.config, args.network)
    if args.target_config:
        _, _, target_inputs, _ = load_inputs(args.target_config)
        if graph_hash(inputs[0], inputs[1]) != graph_hash(target_inputs[0], target_inputs[1]):
            raise ValueError('Target snapshot must use the same network')
        targets = {n: e for n, e in target_inputs[3].items() if n in SCZ_TARGETS and e['kind'] == 'target'}
        inputs = (*inputs[:3], inputs[3] | targets, *inputs[4:])
    paths = run_shared_network_pipeline(inputs, config=AnalysisConfig.from_dict(spec['analysis']),
        cache_dir=root/spec['cache_dir'], output_root=args.output_root or root/spec['results_root'],
        annotation_dir=args.annotation_dir or root/'input/annotations', min_cnvs=args.min_cnvs,
        enrich=not args.skip_enrichment, collapse=not args.skip_orsum, orsum_executable=args.orsum_executable,
        sensitivity=not args.skip_sensitivity, sensitivity_reps=args.sensitivity_reps)
    for kind, path in paths.items():
        if path is not None:
            print(f'{kind}: {path}')
    return paths


if __name__ == '__main__':
    main()
