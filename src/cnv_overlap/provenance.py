"""Content-addressed, atomic artifacts. Unmanifested legacy files are never hits."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy.sparse as sp


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    if a.dtype.hasobject:
        return digest(a.tolist())
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def graph_hash(A, nodes):
    a = sp.csr_matrix(A, dtype=np.float64, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    return digest({"nodes": np.asarray(nodes).tolist(), "shape": list(a.shape),
                   "indptr": array_hash(a.indptr.astype(np.int64)),
                   "indices": array_hash(a.indices.astype(np.int64)), "data": array_hash(a.data)})


def software():
    versions = {}
    for name in ("numpy", "scipy", "pandas", "networkx", "netcoloc", "cnv-network-overlap"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "uninstalled"
    distribution = importlib.metadata.distribution("netcoloc")
    direct_url = distribution.read_text("direct_url.json")
    versions["netcoloc_source"] = json.loads(direct_url) if direct_url else None
    versions["source"] = digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    return versions


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def seal(path, metadata):
    write_json(str(path) + ".json", {"metadata": metadata, "sha256": file_hash(path)})


def valid(path, metadata):
    path = Path(path)
    try:
        stamp = json.loads(Path(str(path) + ".json").read_text())
        return stamp["metadata"] == metadata and stamp["sha256"] == file_hash(path)
    except (FileNotFoundError, ValueError, KeyError):
        return False


def cached_array(directory, kind, metadata, build):
    path = Path(directory) / f"{kind}-{digest(metadata)}.npy"
    if not valid(path, metadata):
        a = np.asarray(build())
        if not np.isfinite(a).all():
            raise ValueError(f"{kind} contains nonfinite values")
        with atomic_path(path) as tmp:
            np.save(tmp, a, allow_pickle=False)
        seal(path, metadata)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def slug(text):
    """Reduce a label to lowercase alphanumeric words joined by single hyphens."""
    cleaned = "".join(c.lower() if c.isalnum() or c == "." else "-" for c in str(text))
    return "-".join(w for w in cleaned.split("-") if w)


def run_dir(root, metadata, *label):
    """Directory named '<label>-<digest>': readable, still unique per identity."""
    parts = [slug(p) for p in label if p is not None and slug(p)]
    return Path(root)/("-".join(parts + [digest(metadata)[:12]]))
