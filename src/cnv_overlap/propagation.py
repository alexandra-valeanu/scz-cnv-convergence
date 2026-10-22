"""Validated heat-conserving propagation and seed-degree-profile references."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from netcoloc.netcoloc_utils import get_degree_binning
from netcoloc.netprop import network_propagation
from threadpoolctl import threadpool_limits

from .provenance import cached_array, graph_hash, software
from .statistics import rng_for


@dataclass(frozen=True)
class PropagationConfig:
    """Settings for heat propagation and degree-matched reference sampling."""
    alpha: float = .5
    reference_reps: int = 1000
    minimum_bin_size: int = 20
    seed: int = 42
    dtype: str = "float32"
    threads: int = 2
    min_reference_count: int = 30

    def __post_init__(self):
        """Validate propagation, sampling, precision, and thread settings."""
        if not 0 < self.alpha < 1:
            raise ValueError("Log-heat NPS requires 0 < alpha < 1")
        if self.reference_reps < 40 or self.minimum_bin_size < 2 or self.threads < 1:
            raise ValueError("Need >=40 reference draws, bin size >=2 and positive threads")
        if not 2 <= self.min_reference_count <= self.reference_reps or self.seed < 0:
            raise ValueError("Invalid reference count or RNG seed")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")


def validate_graph(A, nodes):
    """Validate a connected binary graph and return it sorted by unique node ID."""
    A = sp.csr_matrix(A, dtype=float, copy=True)
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    nodes = np.asarray(nodes, dtype=np.int64)
    if A.shape != (len(nodes), len(nodes)) or len(np.unique(nodes)) != len(nodes) or len(nodes) < 3:
        raise ValueError("Graph shape and unique node IDs must agree")
    if not np.isfinite(A.data).all() or not (A.data == 1).all():
        raise ValueError("This model requires a binary adjacency")
    if (A != A.T).nnz or A.diagonal().any() or np.any(np.asarray(A.sum(0)).ravel() == 0):
        raise ValueError("Need an undirected graph without loops or isolated nodes")
    if sp.csgraph.connected_components(A, directed=False)[0] != 1:
        raise ValueError("Choose a common connected universe before propagation")
    order = np.argsort(nodes)
    return A[order][:, order].tocsr(), nodes[order]


def degree_bins(degrees, minimum_size):
    """Use pinned NetColoc bins with canonical positional node IDs."""
    degrees = np.asarray(degrees)
    if not np.isfinite(degrees).all() or np.any(degrees != np.floor(degrees)):
        raise ValueError("NetColoc degree bins require integer graph degrees")
    bins, _ = get_degree_binning({i: int(d) for i, d in enumerate(degrees)},
                               min(minimum_size, len(degrees)))
    return [np.asarray(group, dtype=int) for group in bins]


def heat_kernel(A, alpha=.5, dtype="float32", threads=2):
    """Return the dense kernel (1 - alpha) * inverse(I - alpha * A / degree).

    Columns conserve heat. Reject nonpositive or nonfinite entries and
    failed conservation checks. Memory use grows quadratically in node count.
    """
    if not 0 <= alpha < 1:
        raise ValueError("Heat propagation requires 0 <= alpha < 1")
    d = np.asarray(A.sum(axis=0)).ravel()
    if np.any(d <= 0):
        raise ValueError("Cannot propagate through degree-zero nodes")
    M = A.multiply(1/d).toarray().astype(dtype)
    M *= -alpha
    M[np.diag_indices_from(M)] += 1
    with threadpool_limits(limits=threads):
        H = scipy.linalg.inv(M, overwrite_a=True, check_finite=True)
    H *= 1-alpha
    if not np.isfinite(H).all() or (H <= 0).any():
        raise ValueError("Nonpositive/nonfinite heat; use float64 or check connectivity")
    if not np.allclose(H.sum(axis=0, dtype=float), 1, atol=2e-5):
        raise ValueError("Heat conservation failed")
    return H


class Propagator:
    """Cache heat kernels and network proximity scores on a validated graph."""
    def __init__(self, A, nodes, cache_dir, config=PropagationConfig()):
        """Validate the graph and prepare node positions, degree bins, and cache keys."""
        self.A, self.nodes = validate_graph(A, nodes)
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.pos = {int(g): i for i, g in enumerate(self.nodes)}
        self.degrees = np.asarray(self.A.sum(0)).ravel()
        self.graph_id = graph_hash(self.A, self.nodes)
        self.metadata = {"graph": self.graph_id, "config": asdict(config), "software": software()}
        self.bins = degree_bins(self.degrees, config.minimum_bin_size)
        self.bin_of = np.empty(len(self.nodes), dtype=int)
        for i, group in enumerate(self.bins):
            self.bin_of[group] = i
        self._H = None
        self._refs = {}
        self._zs = {}

    @property
    def H(self):
        """Load or compute the heat kernel on first access."""
        if self._H is None:
            c = self.config
            meta = {"graph": self.graph_id, "alpha": c.alpha, "dtype": c.dtype,
                    "software": self.metadata["software"], "algorithm": "column_normalized_inverse_v1"}
            self._H = cached_array(self.cache_dir, "kernel", meta,
                lambda: heat_kernel(self.A, c.alpha, c.dtype, c.threads))
        return self._H

    def seed_indices(self, seeds):
        """Return positions of unique seeds; reject empty or out-of-network sets."""
        unique = sorted(set(int(g) for g in seeds))
        missing = set(unique)-self.pos.keys()
        if missing:
            raise ValueError(f"Seeds outside graph; filter and report before propagation: {sorted(missing)[:10]}")
        if not unique:
            raise ValueError("No retained seed genes")
        return np.array([self.pos[g] for g in unique], dtype=int)

    def heat(self, seeds):
        """Return propagated heat in canonical node order for a seed set."""
        cols = self.seed_indices(seeds)
        return network_propagation(self.H, self.nodes.tolist(), self.nodes[cols]).to_numpy()

    def signature(self, cols):
        """Count seed positions in each degree bin for reference matching."""
        return tuple(np.bincount(self.bin_of[cols], minlength=len(self.bins)).tolist())

    def reference(self, cols):
        """Return log-heat mean, population SD, and count for each node.

        Random seed sets match the supplied seed positions by degree bin. Each
        node contributes only when it is not a seed in that reference draw.
        """
        signature = self.signature(cols)
        if signature in self._refs:
            return self._refs[signature]
        c = self.config
        meta = {**self.metadata, "degree_histogram": list(signature),
                "algorithm": "log_heat_random_seed_mask_v1"}
        def build():
            """Estimate log-heat moments from degree-matched random seed sets."""
            rng = rng_for(c.seed, "reference", signature)
            count = np.zeros(len(self.nodes), dtype=int)
            mean = np.zeros(len(self.nodes))
            m2 = np.zeros(len(self.nodes))
            for _ in range(c.reference_reps):
                draw = np.concatenate([rng.choice(group, size=n, replace=False)
                    for group, n in zip(self.bins, signature) if n])
                values = np.log(self.heat(self.nodes[draw]))
                valid = np.ones(len(self.nodes), dtype=bool)
                valid[draw] = False
                count[valid] += 1
                delta = values[valid]-mean[valid]
                mean[valid] += delta/count[valid]
                m2[valid] += delta*(values[valid]-mean[valid])
            if np.any(count < c.min_reference_count):
                raise ValueError("Insufficient unseeded reference draws at some nodes; increase reps/bin size")
            sd = np.sqrt(m2/count)
            if np.any(sd <= 1e-12):
                raise ValueError("Degenerate NPS reference; choose a nondegenerate graph/seed definition")
            return np.stack([mean, sd, count])
        ref = cached_array(self.cache_dir, "reference", meta, build)
        if len(self._refs) >= 128:
            self._refs.clear()
        self._refs[signature] = ref
        return ref

    def nps(self, seeds):
        """Return cached log-heat z-scores against the degree-matched reference."""
        cols = self.seed_indices(seeds)
        key = tuple(cols.tolist())
        if key in self._zs:
            return self._zs[key]
        meta = {**self.metadata, "seeds": self.nodes[cols].tolist(), "algorithm": "nps_v1"}
        def build():
            """Standardize log heat and reject nonfinite proximity scores."""
            mu, sd, _ = self.reference(cols)
            z = (np.log(self.heat(self.nodes[cols]))-mu)/sd
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite NPS")
            return z
        z = cached_array(self.cache_dir, "nps", meta, build)
        # Limit RAM/file descriptors when processing many gene sets.
        if len(self._zs) >= 128:
            self._zs.clear()
        self._zs[key] = z
        return z
