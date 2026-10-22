"""Load and validate frozen network, seed, and coordinate inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .provenance import file_hash


def normalized_coordinates(coords):
    """Return sorted, unique gene coordinates in 1-based inclusive format."""
    c = coords.copy()
    required = {"entrez", "chrom", "start", "end"}
    if not required <= set(c):
        raise ValueError(f"Coordinates need {required}")
    c["chrom"] = c.chrom.astype(str).str.removeprefix("chr")
    for col in ["entrez", "start", "end"]:
        numeric = pd.to_numeric(c[col], errors="raise")
        if numeric.isna().any() or (numeric != np.floor(numeric)).any():
            raise ValueError(f"Noninteger {col}")
        c[col] = numeric.astype(np.int64)
    if (c.start < 1).any() or (c.end < c.start).any() or c.entrez.duplicated().any():
        raise ValueError("Need unique genes and valid 1-based inclusive coordinates")
    return c.sort_values(["chrom", "start", "end", "entrez"]).reset_index(drop=True)


def load_inputs(config_path):
    """Load and validate a JSON input specification.

    Return (spec, root, inputs, provenance), where inputs contains adjacency,
    node IDs, symbols, entities, coordinates, and regions. Keep the largest
    connected component after any gene-universe restriction.
    """
    config_path = Path(config_path).resolve()
    spec = json.loads(config_path.read_text())
    root = (config_path.parent/spec.get("root", "..")).resolve()
    paths = {k: root/v for k, v in spec["inputs"].items()}
    expected = spec.get("input_sha256", {})
    hashes = {k: file_hash(p) for k, p in paths.items()}
    for key, wanted in expected.items():
        if hashes.get(key) != wanted:
            raise ValueError(f"Input changed: {key}; explicitly prepare a new snapshot")
    if "adjacency" in paths:
        A = sp.load_npz(paths["adjacency"])
        table = pd.read_csv(paths["nodes"], sep="\t")
        nodes = table.entrez.to_numpy(dtype=np.int64)
        symbols = dict(zip(nodes, table.symbol.fillna(table.entrez.astype(str))))
    else:
        raw_nodes = json.loads(paths["network_nodes"].read_text())
        cx = {int(n["@id"]): int(str(n["r"]).split(":")[-1]) for n in raw_nodes}
        if len(set(cx.values())) != len(cx):
            raise ValueError("Network maps multiple nodes to one Entrez ID")
        nodes = np.array(sorted(cx.values()), dtype=np.int64)
        symbols = {cx[int(n["@id"])]: n["n"] for n in raw_nodes}
        pos = {g: i for i, g in enumerate(nodes)}
        edges = json.loads(paths["network_edges"].read_text())
        i = np.fromiter((pos[cx[int(e["s"])]] for e in edges), dtype=np.int64)
        j = np.fromiter((pos[cx[int(e["t"])]] for e in edges), dtype=np.int64)
        keep = i != j
        A = sp.coo_matrix((np.ones(2*keep.sum(), dtype=float),
            (np.r_[i[keep], j[keep]], np.r_[j[keep], i[keep]])), shape=(len(nodes), len(nodes))).tocsr()
        A.data[:] = 1
    seeds = pd.read_csv(paths["seeds"], sep="\t")
    required = {"entity", "kind", "entrez"}
    if not required <= set(seeds) or not set(seeds.kind) <= {"cnv", "comparator", "target"}:
        raise ValueError("Seed table requires entity, kind, entrez and supported entity kinds")
    if seeds[list(required)].isna().any().any():
        raise ValueError("Missing seed identifiers or kinds")
    entities = {}
    for name, frame in seeds.groupby("entity", sort=True):
        if frame.kind.nunique() != 1:
            raise ValueError(f"Conflicting entity kinds for {name}")
        numeric = pd.to_numeric(frame.entrez)
        if (numeric <= 0).any() or (numeric != np.floor(numeric)).any():
            raise ValueError("Entrez IDs must be positive integers")
        entities[name] = {"kind": frame.kind.iloc[0], "seeds": sorted(set(numeric.astype(int)))}
    if "gene_universe" in paths:
        universe = set(pd.read_csv(paths["gene_universe"], sep="\t").entrez.astype(int))
        keep = np.isin(nodes, list(universe))
        A, nodes = A[keep][:, keep], nodes[keep]
    _, labels = sp.csgraph.connected_components(A, directed=False)
    keep = labels == np.bincount(labels).argmax()
    dropped = nodes[~keep].tolist()
    A, nodes = A[keep][:, keep], nodes[keep]
    coords = normalized_coordinates(pd.read_csv(paths["coordinates"], sep="\t"))
    regions = pd.read_csv(paths["regions"], sep="\t")
    if not {"entity", "chrom", "start", "end", "assembly"} <= set(regions) or not (regions.assembly == "GRCh37").all():
        raise ValueError("Regions need explicit GRCh37 1-based inclusive coordinates")
    if (regions.start < 1).any() or (regions.end < regions.start).any() or regions.entity.duplicated().any():
        raise ValueError("Invalid region coordinates/duplicate region names")
    region_names = set(n for n, e in entities.items() if e["kind"] != "target")
    if not region_names <= set(regions.entity):
        raise ValueError("Every region entity requires an interval")
    for name in region_names:
        r = regions.set_index("entity").loc[name]
        known = coords[coords.entrez.isin(entities[name]["seeds"])]
        if not ((known.chrom.astype(str).str.removeprefix("chr") == str(r.chrom).removeprefix("chr")) &
                (known.start <= r.end) & (known.end >= r.start)).all():
            raise ValueError(f"Seed/coordinate conflict in {name}")
    provenance = {"paths": {k: str(p.relative_to(root)) for k, p in paths.items()}, "sha256": hashes,
                  "source_config": file_hash(config_path), "lcc_dropped": dropped,
                  "seed_definition": spec.get("seed_definition", "See source manifest")}
    return spec, root, (A, nodes, symbols, entities, coords, regions), provenance
