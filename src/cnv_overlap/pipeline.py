"""Offline analysis from explicit, versioned inputs. Every run owns its outputs."""
from cnv_overlap.results import finalize_manifest, publish_selection, resolve_path
from dataclasses import asdict, dataclass, field
import itertools
import json

import numpy as np
import pandas as pd

from .propagation import PropagationConfig, Propagator
from .provenance import atomic_path, digest, file_hash, run_dir, seal, software, valid, write_json
from .statistics import Thresholds, bh, hit_mask, label_null, scores, netcoloc_tail


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings for propagation, pair selection, permutations, and FDR."""
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    permutation_reps: int = 1000
    overlap: str = "bin"
    fdr: float = .05
    comparisons: str = "all"

    def __post_init__(self):
        """Reject unsupported comparison scopes and invalid testing settings."""
        if self.comparisons not in ("all", "cnv_cnv", "cnv_disorder"):
            raise ValueError("comparisons must be all, cnv_cnv, or cnv_disorder")
        if self.permutation_reps < 2 or self.overlap not in ("bin", "remove", "none") or not 0 < self.fdr < 1:
            raise ValueError("Invalid repetitions, overlap policy, or FDR")

    @classmethod
    def from_dict(cls, d):
        """Build nested configuration objects from a mapping; reject obsolete keys."""
        d = dict(d)
        obsolete = set(d) & {"windows", "run_windows", "contrasts", "label_reps"}
        if obsolete:
            raise ValueError(f"Obsolete analysis settings {sorted(obsolete)}; use the NetColoc configs and permutation_reps")
        for key, ctor in [("propagation", PropagationConfig), ("thresholds", Thresholds)]:
            if key in d:
                d[key] = ctor(**d[key])
        return cls(**d)


def pair_scores(a, b, seeds_a, seeds_b, nodes, config):
    """Return overlap statistics and the node mask after shared-seed filtering."""
    keep = np.ones(len(nodes), dtype=bool)
    if config.overlap == "remove":
        keep &= ~np.isin(nodes, list(set(seeds_a)&set(seeds_b)))
    if not keep.any():
        raise ValueError("No eligible nodes after shared-seed removal")
    return scores(np.asarray(a)[keep], np.asarray(b)[keep], config.thresholds), keep


def write_table(frame, path):
    """Atomically write a DataFrame as TSV without its index."""
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, sep="\t", index=False)


def pair_plan(entities, comparisons="all"):
    """Return deterministic (a, b, family) comparisons for the requested scope."""
    loci = sorted(n for n, e in entities.items() if e["kind"] == "cnv")
    regions = sorted(n for n, e in entities.items() if e["kind"] in ("cnv", "comparator"))
    targets = sorted(n for n, e in entities.items() if e["kind"] == "target")
    if comparisons == "cnv_cnv":
        return [(a, b, "region_pair") for a, b in itertools.combinations(loci, 2)]
    if comparisons == "cnv_disorder":
        return [(a, b, "locus_target") for a in loci for b in targets]
    if comparisons != "all":
        raise ValueError(f"Unknown comparison selection: {comparisons}")
    return [(a, b, "locus_target") for a in loci for b in targets] + [
        (a, b, "region_pair") for a, b in itertools.combinations(regions, 2)]


def run(A, nodes, symbols, entities, coords, regions, *, config=AnalysisConfig(),
        output_root="results/robust", cache_dir="data/cache/robust", provenance=None, progress=print):
    """Run or reuse a checksummed analysis and return its output directory.

    Node IDs must align with adjacency rows and columns. Entities map names
    to kind and seeds. Completed runs are reused only after artifact checks;
    failed runs retain a manifest describing the error.
    """
    pairs = pair_plan(entities, config.comparisons)
    active = {n for a, b, _ in pairs for n in (a, b)}
    entities = {n: e for n, e in entities.items() if n in active}
    if not pairs:
        raise ValueError("No comparisons available for the selected analysis")
    prop = Propagator(A, nodes, cache_dir, config.propagation)
    nodes = prop.nodes
    # Capture before mutating/filtering; missing IDs and tissue-induced changes matter.
    entity_input = {n: {**e, "seeds": sorted(set(map(int, e["seeds"])))} for n, e in sorted(entities.items())}
    meta = {"config": asdict(config), "graph": prop.graph_id, "entities": entity_input,
            "symbols": {str(k): v for k, v in sorted(symbols.items())},
            "coordinates": coords.to_json(orient="records"), "regions": regions.to_json(orient="records"),
            "inputs": provenance or {}, "software": software()}
    # Normalize the identity to its JSON representation.
    meta = json.loads(json.dumps(meta))
    out = run_dir(output_root, meta, config.comparisons, f"alpha{config.propagation.alpha}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/"manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["identity"] != meta:
            raise RuntimeError("Run identity collision")
        if previous.get("status") == "complete":
            if not all((out/p).exists() and file_hash(out/p) == h for p, h in previous["artifacts"].items()):
                raise RuntimeError("Completed run artifacts changed; use a new output root or remove corrupted run")
            publish_selection(out)
            return out
    write_json(manifest, {"status": "running", "identity": meta})
    try:
        _execute(prop, symbols, entity_input, coords, regions, config, out, meta, progress)
        artifacts = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*"))
                     if p.is_file() and p != manifest and not p.name.startswith(".writing-")}
        finalize_manifest(manifest, {"status": "complete", "identity": meta, "artifacts": artifacts,
                   "primary": "size", "primary_null": "netcoloc", "p_method": "normal_permutation",
                   "interpretation": "unsigned network colocalization against permuted gene labels"})
    except Exception as ex:
        write_json(manifest, {"status": "failed", "identity": meta, "error": f"{type(ex).__name__}: {ex}"})
        raise
    return out


def _execute(prop, symbols, entities, coords, regions, config, out, meta, progress):
    """Write seed QC, propagation profiles, pair tests, and candidate genes."""
    nodes = prop.nodes
    qc = []
    entities = {n: dict(e) for n, e in entities.items()}
    for name, e in entities.items():
        raw = set(e["seeds"])
        retained = sorted(raw & prop.pos.keys())
        if not retained:
            raise ValueError(f"{name}: zero seeds in selected network")
        e["seeds"] = retained
        qc.append({"entity": name, "kind": e["kind"], "raw_seeds": len(raw), "retained_seeds": len(retained),
            "missing_ids": ",".join(map(str, sorted(raw-set(retained)))),
            "retained_with_coordinates": len(set(retained)&set(coords.entrez)),
            "median_degree": float(np.median(prop.degrees[prop.seed_indices(retained)]))})
    write_table(pd.DataFrame(qc), resolve_path(out/"seed_qc.tsv"))
    names = sorted(entities)
    overlap = pd.DataFrame([[len(set(entities[a]["seeds"]) & set(entities[b]["seeds"])) for b in names] for a in names], index=names, columns=names)
    overlap.rename_axis("entity").reset_index().to_csv(resolve_path(out/"seed_overlap.tsv"), sep="\t", index=False)
    pairs = pair_plan(entities, config.comparisons)
    if not pairs:
        raise ValueError("No comparisons: need CNVs and targets or at least two regions")
    write_table(pd.DataFrame(pairs, columns=["a", "b", "family"]), resolve_path(out/"hypotheses.tsv"))
    zs = {}
    for name, e in entities.items():
        progress(f"NPS: {name} ({len(e['seeds'])} seeds)")
        zs[name] = np.asarray(prop.nps(e["seeds"])).copy()
    write_table(pd.DataFrame({"entrez": nodes, **zs}), resolve_path(out/"nps.tsv"))
    records = []
    gene_records = []
    for a, b, family in pairs:
        pair_id = digest([a, b])[:16]
        progress(f"NetColoc: {a} x {b}")
        path = resolve_path(out/f"netcoloc-{pair_id}.npz")
        pm = {"run": digest(meta), "a": a, "b": b}
        values, keep = pair_scores(zs[a], zs[b], entities[a]["seeds"], entities[b]["seeds"], nodes, config)
        if valid(path, pm):
            with np.load(path) as data:
                null = {s: data[s] for s in values}
        else:
            obs, null = label_null(zs[a], zs[b], nodes, entities[a]["seeds"], entities[b]["seeds"],
                reps=config.permutation_reps, seed=config.propagation.seed, overlap=config.overlap, thresholds=config.thresholds)
            assert all(np.isclose(obs[s], values[s]) for s in values)
            with atomic_path(path) as tmp:
                np.savez_compressed(tmp, **null)
            seal(path, pm)
        for statistic, value in values.items():
            records.append({"a": a, "b": b, "family": family, "statistic": statistic,
                "null": "netcoloc", "primary": statistic == "size", **netcoloc_tail(value, null[statistic])})
        hit = np.zeros(len(nodes), dtype=bool)
        hit[keep] = hit_mask(zs[a][keep], zs[b][keep], config.thresholds)
        for i in np.flatnonzero(hit):
            g = int(nodes[i])
            gene_records.append({"a": a, "b": b, "entrez": g, "symbol": symbols.get(g, str(g)),
                "nps_a": zs[a][i], "nps_b": zs[b][i], "seed_a": g in entities[a]["seeds"],
                "seed_b": g in entities[b]["seeds"], "status": "thresholded_candidate_not_gene_FDR"})
    columns = ["a", "b", "entrez", "symbol", "nps_a", "nps_b", "seed_a", "seed_b", "status"]
    write_table(pd.DataFrame(gene_records, columns=columns), resolve_path(out/"candidate_genes.tsv"))
    _save_tests(records, resolve_path(out/"pair_tests.tsv"))
    interpretation(out, config)


def _save_tests(records, path):
    """Write tests with BH correction within each null, family, and statistic."""
    df = pd.DataFrame(records)
    df["q"] = np.nan
    for _, idx in df.groupby(["null", "family", "statistic"]).groups.items():
        df.loc[idx, "q"] = bh(df.loc[idx, "p"].to_numpy())
    write_table(df, path)


def interpretation(out, config):
    """Write descriptive gene recurrence and a summary of significant size tests."""
    tests = pd.read_csv(resolve_path(out/"pair_tests.tsv"), sep="\t")
    genes = pd.read_csv(resolve_path(out/"candidate_genes.tsv"), sep="\t")
    null_name = "netcoloc"
    selected = tests[(tests["null"] == null_name) & (tests.statistic == "size") & (tests.q < config.fdr)]
    chosen = set(zip(selected.a, selected.b))
    keep = genes.loc[np.array([p in chosen for p in zip(genes.a, genes.b)], dtype=bool)]
    columns = ["entrez", "symbol", "n_pairs", "n_regions", "interpretation"]
    rows = []
    kinds = json.loads((out/"manifest.json").read_text())["identity"]["entities"]
    for (g, symbol), frame in keep.groupby(["entrez", "symbol"]):
        reached = set(frame.a)|set(frame.b)
        rows.append({"entrez": g, "symbol": symbol, "n_pairs": len(frame),
                     "n_regions": sum(kinds[n]["kind"] != "target" for n in reached),
                     "interpretation": "descriptive_recurrence_not_independent_evidence"})
    write_table(pd.DataFrame(rows, columns=columns), resolve_path(out/"recurrence.tsv"))
    (out/"summary.md").write_text(
        "# Run summary\n\nPrimary endpoint: positive-proximity set size. "
        f"Selected null: {null_name}. {len(selected)} pairs pass q < {config.fdr}.\n\n"
        "BH families: locus–target and region–region, separately. Secondary statistics do not select discoveries. "
        "Genes are thresholded candidates; no gene-level FDR claim is made. Recurrence and ORA are descriptive.\n\n"
        "NetColoc normal-tail p-values use gene-label permutations with the configured shared-seed control. "
        "Empirical p-values and binomial tail intervals are diagnostics. "
        "This tests network colocalization, not specificity against other genomic regions or dosage direction.\n")
