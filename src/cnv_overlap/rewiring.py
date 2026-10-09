"""Deterministic degree-preserving rewiring chains and graph-specific rescoring."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .pipeline import pair_plan, pair_scores, write_table
from .propagation import Propagator
from .provenance import atomic_path, graph_hash, run_dir, seal, software, valid, write_json
from .statistics import rng_for


@dataclass(frozen=True)
class RewireConfig:
    chains: int = 3
    steps: int = 5
    attempts_per_edge: float = 2.
    seed: int = 42

    def __post_init__(self):
        if self.chains < 2 or self.steps < 1 or self.attempts_per_edge <= 0 or self.seed < 0:
            raise ValueError("Need at least two chains, positive steps/attempts, nonnegative seed")


def swap(A, attempts, rng):
    upper = sp.triu(A, k=1).tocoo()
    edges = np.column_stack([upper.row, upper.col]).astype(np.int64)
    n, m = A.shape[0], len(edges)
    if m < 2 or attempts < 0:
        raise ValueError("Need at least two edges and nonnegative attempts")
    keys = set((edges[:, 0]*n+edges[:, 1]).tolist())
    accepted = 0
    for _ in range(int(attempts)):
        i, j = rng.integers(m, size=2)
        if i == j:
            continue
        a, b = map(int, edges[i])
        c, d = map(int, edges[j])
        if rng.integers(2):
            c, d = d, c
        if len({a, b, c, d}) != 4:
            continue
        u, v = sorted((a, d)), sorted((c, b))
        k1, k2 = u[0]*n+u[1], v[0]*n+v[1]
        if k1 in keys or k2 in keys:
            continue
        keys.remove(min(a, b)*n+max(a, b))
        keys.remove(min(c, d)*n+max(c, d))
        keys.update([k1, k2])
        edges[i], edges[j] = u, v
        accepted += 1
    i, j = edges.T
    S = sp.coo_matrix((np.ones(2*m), (np.r_[i, j], np.r_[j, i])), shape=A.shape).tocsr()
    if (S != S.T).nnz or S.diagonal().any() or S.nnz != A.nnz or not np.array_equal(S.sum(0), A.sum(0)):
        raise RuntimeError("Rewiring invariant failed")
    return S, accepted


def sampled_topology(A, rng):
    """Fixed-budget diagnostics; estimates, not exact graph statistics or mixing tests."""
    A = sp.csr_matrix(A)
    degrees = np.diff(A.indptr)
    clustering = []
    for node in rng.choice(A.shape[0], size=min(96, A.shape[0]), replace=False):
        neighbors = A.indices[A.indptr[node]:A.indptr[node+1]]
        k = len(neighbors)
        if k < 2:
            clustering.append(0.)
            continue
        left = rng.integers(k, size=256)
        right = rng.integers(k-1, size=256)
        right += right >= left
        clustering.append(float(np.asarray(A[neighbors[left], neighbors[right]]).mean()))
    slots = rng.integers(A.nnz, size=min(10000, A.nnz))
    rows = np.searchsorted(A.indptr, slots, side="right")-1
    cols = A.indices[slots]
    x, y = degrees[rows], degrees[cols]
    assortativity = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    return {"sampled_mean_clustering": float(np.mean(clustering)),
            "sampled_degree_assortativity": assortativity,
            "sampled_clustering_nodes": len(clustering)}


def chains(A, nodes, directory, config=RewireConfig()):
    directory = Path(directory)
    identity = {"graph": graph_hash(A, nodes), "config": asdict(config), "software": software()}
    root = run_dir(directory, identity, f"chains{config.chains}", f"steps{config.steps}")
    root.mkdir(parents=True, exist_ok=True)
    endpoints, diagnostics = [], []
    for chain in range(config.chains):
        S = A
        for step in range(config.steps):
            path = root/f"chain-{chain}-step-{step}.npz"
            metadata = {**identity, "chain": chain, "step": step, "parent": graph_hash(S, nodes)}
            if valid(path, metadata) and path.with_suffix(".diagnostics.json").exists():
                import json
                info = json.loads(path.with_suffix(".diagnostics.json").read_text())
                S = sp.load_npz(path)
            else:
                attempts = int(config.attempts_per_edge*(A.nnz//2))
                candidate, accepted = swap(S, attempts, rng_for(config.seed, chain, step))
                connected = sp.csgraph.connected_components(candidate, directed=False)[0] == 1
                # Keep the same universe; reject a disconnected proposal as a whole.
                if connected:
                    S = candidate
                info = {"chain": chain, "step": step, "attempts": attempts, "accepted_swaps": accepted,
                    "accepted_graph": bool(connected), "edge_overlap": float(A.multiply(S).nnz/A.nnz),
                    "degree_preserved": bool(np.array_equal(A.sum(0), S.sum(0))),
                    **sampled_topology(S, rng_for(config.seed, "topology", chain, step))}
                with atomic_path(path) as tmp:
                    sp.save_npz(tmp, S)
                write_json(path.with_suffix(".diagnostics.json"), info)
                seal(path, metadata)
            diagnostics.append(info)
        endpoints.append(S)
    write_table(pd.DataFrame(diagnostics), root/"diagnostics.tsv")
    return root, endpoints


def score_ensemble(inputs, analysis_config, *, output_root, cache_dir, config=RewireConfig(), progress=print):
    A, nodes, symbols, entities, _, _ = inputs
    pairs = pair_plan(entities, analysis_config.comparisons)
    active = {n for a, b, _ in pairs for n in (a, b)}
    entities = {n: e for n, e in entities.items() if n in active}
    if not pairs:
        raise ValueError("No comparisons available for the selected analysis")
    base = Propagator(A, nodes, cache_dir, analysis_config.propagation)
    A, nodes = base.A, base.nodes
    root, graphs = chains(A, nodes, Path(cache_dir)/"rewiring", config)
    identity = {"graph": base.graph_id, "analysis": asdict(analysis_config), "rewiring": asdict(config),
                "entities": entities, "software": software()}
    out = run_dir(output_root, identity, "rewiring", analysis_config.comparisons)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, graph in enumerate([A, *graphs]):
        progress(f"Rewiring propagation: {'real' if index == 0 else f'chain {index-1} endpoint'}")
        prop = base if index == 0 else Propagator(graph, nodes, cache_dir, analysis_config.propagation)
        seeds = {n: sorted(set(e["seeds"]) & prop.pos.keys()) for n, e in entities.items()}
        zs = {n: np.asarray(prop.nps(s)) for n, s in seeds.items()}
        for a, b, family in pairs:
            score, _ = pair_scores(zs[a], zs[b], seeds[a], seeds[b], nodes, analysis_config)
            rows.append({"graph": "real" if index == 0 else f"chain_{index-1}", "a": a, "b": b,
                         "family": family, **score})
        if index == 0:
            base._H = None
        del prop
    write_table(pd.DataFrame(rows), out/"rewiring_scores.tsv")
    write_json(out/"manifest.json", {"identity": identity, "chain_diagnostics": str(root/"diagnostics.tsv"),
        "interpretation": "robustness diagnostics, one endpoint per chain; no independent-draw p-values or claim of verified mixing"})
    return out
