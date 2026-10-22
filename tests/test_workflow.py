from cnv_overlap.results import resolve_path
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cnv_overlap.pipeline import AnalysisConfig, run
from cnv_overlap.propagation import PropagationConfig


def fixture():
    n = 60
    rng = np.random.default_rng(6)
    a = np.triu((rng.random((n, n)) < .1).astype(float), 1)
    for i in range(n-1):
        a[i, i+1] = 1
    A = sp.csr_matrix(a+a.T)
    nodes = np.arange(1, n+1)
    coords = pd.DataFrame({"entrez": nodes, "chrom": "1", "start": nodes*100, "end": nodes*100+10})
    regions = pd.DataFrame([{"entity": "cnv_a", "chrom": "1", "start": 100, "end": 210, "assembly": "GRCh37"},
                            {"entity": "cnv_b", "chrom": "1", "start": 500, "end": 610, "assembly": "GRCh37"}])
    entities = {"cnv_a": {"kind": "cnv", "seeds": [1, 2]}, "cnv_b": {"kind": "cnv", "seeds": [5, 6]},
                "target_a": {"kind": "target", "seeds": [3, 4]}, "target_b": {"kind": "target", "seeds": [30, 31]}}
    return A, nodes, {int(g): f"G{g}" for g in nodes}, entities, coords, regions


def settings():
    return AnalysisConfig(propagation=PropagationConfig(reference_reps=60, minimum_bin_size=10, min_reference_count=20),
        permutation_reps=49)


def test_clean_end_to_end_and_integrity(tmp_path):
    args = fixture()
    kwargs = dict(config=settings(), output_root=tmp_path/"results", cache_dir=tmp_path/"cache", progress=lambda _: None)
    out = run(*args, **kwargs)
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["primary_null"] == "netcoloc"
    assert manifest["p_method"] == "normal_permutation"
    tests = pd.read_csv(resolve_path(out/"pair_tests.tsv"), sep="\t")
    assert len(tests) == 5*3
    assert tests.p.dropna().between(0, 1).all()
    assert tests.p_empirical.between(0, 1).all()
    assert run(*args, **kwargs) == out
    (resolve_path(out/"pair_tests.tsv")).write_text("corrupt")
    with pytest.raises(RuntimeError, match="artifacts changed"):
        run(*args, **kwargs)


def test_legacy_config_rejected_before_execution():
    for key, value in [("windows", {}), ("run_windows", False), ("contrasts", []), ("label_reps", 99)]:
        with pytest.raises(ValueError, match="Obsolete analysis settings"):
            AnalysisConfig.from_dict({key: value})
