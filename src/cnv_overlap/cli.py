"""Command-line entry points for reproducible offline runs."""
import argparse
import json
from pathlib import Path

from .inputs import load_inputs
from .pipeline import AnalysisConfig, run


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate", "rewire"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "rewire":
            p.add_argument("--chains", type=int, default=3)
            p.add_argument("--steps", type=int, default=5)
            p.add_argument("--attempts-per-edge", type=float, default=2.)
    p = sub.add_parser("annotate")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gmt", required=True, nargs="+")
    p.add_argument("--output-dir", required=True)
    p = sub.add_parser("report")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gmt", nargs="*", default=[])
    args = parser.parse_args(argv)
    spec, root, inputs, provenance = load_inputs(args.config)
    config = AnalysisConfig.from_dict(spec.get("analysis", {}))
    if args.command == "validate":
        print(json.dumps({"nodes": len(inputs[1]), "edges": inputs[0].nnz//2,
                          "entities": len(inputs[3]), "provenance": provenance}, indent=2))
    elif args.command == "rewire":
        from .rewiring import RewireConfig, score_ensemble
        print(score_ensemble(inputs, config, output_root=root/"results/rewiring",
            cache_dir=root/spec.get("cache_dir", "data/cache/robust"),
            config=RewireConfig(chains=args.chains, steps=args.steps, attempts_per_edge=args.attempts_per_edge)))
    elif args.command == "report":
        from .reporting import export_report
        print(export_report(args.run_dir, inputs, output_dir=args.output_dir,
                            gmt_paths=args.gmt, fdr=config.fdr))
    elif args.command == "annotate":
        from .enrichment import annotate_run
        _, nodes, symbols, *_ = inputs
        annotate_run(args.run_dir, [Path(p) for p in args.gmt], [symbols[g] for g in nodes], output_dir=args.output_dir)
    else:
        print(run(*inputs, config=config, output_root=root/spec.get("output_root", "results/robust"),
                  cache_dir=root/spec.get("cache_dir", "data/cache/robust"), provenance=provenance))


if __name__ == "__main__":
    main()
