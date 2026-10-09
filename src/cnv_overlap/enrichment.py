"""Offline, versioned ORA: descriptive annotation, not network-null evidence."""
from pathlib import Path
import json

import pandas as pd
from scipy.stats import hypergeom

from .provenance import file_hash, write_json
from .statistics import bh


def load_gmt(path):
    terms = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed GMT row in {path}")
            key, description, *genes = fields
            if key in terms:
                raise ValueError(f"Duplicate term {key}")
            terms[key] = (description, {g.strip() for g in genes if g.strip()})
    return terms


def ora(query, background, terms, min_size=5):
    bg = set(background)
    query = set(query)&bg
    rows = []
    for term, (description, genes) in sorted(terms.items()):
        eligible = genes&bg
        if len(eligible) < min_size:
            continue
        k = len(query&eligible)
        p = float(hypergeom.sf(k-1, len(bg), len(eligible), len(query))) if query else 1.
        rows.append({"term": term, "description": description, "query_size": len(query), "term_size": len(eligible),
                     "background_size": len(bg), "overlap": k, "p": p, "genes": ",".join(sorted(query&eligible))})
    out = pd.DataFrame(rows, columns=["term", "description", "query_size", "term_size", "background_size", "overlap", "p", "genes"])
    out["q"] = bh(out.p)
    out['gene_ratio'] = out.overlap / out.query_size.replace(0, float('nan'))
    out['fold_enrichment'] = out.gene_ratio / (out.term_size / out.background_size)
    return out


def annotate_run(run_dir, gmt_paths, background_symbols, *, output_dir, min_size=5):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genes = pd.read_csv(run_dir/"candidate_genes.tsv", sep="\t")
    identity = json.loads((run_dir/'manifest.json').read_text())['identity']
    terms = {}
    for path in gmt_paths:
        for key, value in load_gmt(path).items():
            if key in terms and terms[key] != value:
                raise ValueError(f"Conflicting GMT definitions of {key}")
            terms[key] = value
    frames = []
    hypotheses = pd.read_csv(run_dir/"hypotheses.tsv", sep="\t")
    qc = []
    for r in hypotheses.itertuples():
        query = genes.loc[(genes.a == r.a)&(genes.b == r.b), "symbol"].dropna()
        background = set(background_symbols)
        if identity['config']['overlap'] == 'remove':
            shared = set(identity['entities'][r.a]['seeds']) & set(identity['entities'][r.b]['seeds'])
            excluded = {identity['symbols'].get(str(g), str(g)) for g in shared}
            background -= excluded
        covered = set().union(*(g for _, g in terms.values())) if terms else set()
        qc.append({'a': r.a, 'b': r.b, 'query_symbols': len(set(query)),
                   'background_symbols': len(background), 'query_annotated': len(set(query)&covered),
                   'background_annotated': len(background&covered)})
        frame = ora(query, background, terms, min_size)
        frame.insert(0, "b", r.b)
        frame.insert(0, "a", r.a)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out["q_all_pair_terms"] = bh(out.p) if 'p' in out else pd.Series(dtype=float)
    out.to_csv(output_dir/"descriptive_ora.tsv", sep="\t", index=False)
    pd.DataFrame(qc).to_csv(output_dir/'annotation_coverage.tsv', sep='\t', index=False)
    write_json(output_dir/"manifest.json", {"gmt": {str(p): file_hash(p) for p in gmt_paths},
        "background": sorted(set(background_symbols)), "min_size": min_size,
        "candidates_sha256": file_hash(run_dir/"candidate_genes.tsv"),
        "background_policy": "network symbols minus shared seeds when overlap=remove",
        "interpretation": "descriptive ORA; q within pair and across pair-terms; not calibrated for network selection"})
    return out
