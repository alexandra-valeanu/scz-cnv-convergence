import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cnv_overlap.pipeline import AnalysisConfig, interpretation
from cnv_overlap.interpretation import candidate_network


def test_zero_candidates_and_zero_discoveries(tmp_path):
    pd.DataFrame([{"a": "a", "b": "b", "null": "netcoloc", "statistic": "size", "q": 1.}]).to_csv(tmp_path/"pair_tests.tsv", sep="\t", index=False)
    pd.DataFrame(columns=["a", "b", "entrez", "symbol", "nps_a", "nps_b", "seed_a", "seed_b", "status"]).to_csv(tmp_path/"candidate_genes.tsv",sep="\t",index=False)
    (tmp_path/"manifest.json").write_text(json.dumps({"identity":{"entities": {"a":{"kind":"cnv"},"b":{"kind":"target"}}}}))
    interpretation(tmp_path, AnalysisConfig())
    assert pd.read_csv(tmp_path/"recurrence.tsv",sep="\t").empty
    G, tab = candidate_network(sp.csr_matrix((3,3)), np.arange(3), pd.DataFrame(columns=["entrez"]))
    assert G.number_of_nodes() == 0 and tab.empty
