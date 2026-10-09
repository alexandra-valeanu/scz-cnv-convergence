"""Verify pair selection, eligible backgrounds, the combined test family and plots."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom

from cnv_overlap.provenance import file_hash
from cnv_overlap.statistics import bh

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))  # as the notebooks do
spec = importlib.util.spec_from_file_location('brain_pair_enrichment', Path(__file__).parents[1]/'scripts/brain_pair_enrichment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_selection_background_and_multiple_testing(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    pd.DataFrame([
        ['cnv1', 'schema', 'netcoloc', 'size', 'ok', .01],
        ['cnv1', 'gwaspri', 'netcoloc', 'size', 'ok', .2],
        ['cnv1', 'gwaspri', 'netcoloc', 'positive_mean', 'ok', .001],
    ], columns=['a', 'b', 'null', 'statistic', 'status', 'q']).to_csv(source/'pair_tests.tsv', sep='\t', index=False)
    pd.DataFrame([['cnv1', 'schema', 2, 'B'], ['cnv1', 'gwaspri', 3, 'C']],
        columns=['a', 'b', 'entrez', 'symbol']).to_csv(source/'candidate_genes.tsv', sep='\t', index=False)
    pd.DataFrame({'entrez': [1, 2, 3, 4, 5]}).to_csv(source/'nps.tsv', sep='\t', index=False)
    manifest = {'status':'complete', 'identity': {
        'inputs': {'network':'test_brain'},
        'config': {'overlap':'remove', 'comparisons':'cnv_disorder'},
        'symbols': {str(i): s for i,s in enumerate('ABCDEZ', 1)},
        'entities': {'cnv1': {'kind':'cnv', 'seeds':[1]},
                     'schema': {'kind':'target', 'seeds':[1]},
                     'gwaspri': {'kind':'target', 'seeds':[3]}}},
        'artifacts': {p.name:file_hash(p) for p in source.iterdir()}}
    (source/'manifest.json').write_text(json.dumps(manifest))
    annotations=tmp_path/'annotations'
    annotations.mkdir()
    for name in module.LIBRARIES.values():
        (annotations/name).write_text('hit\tdescription\tA\tB\tZ\nzero\tdescription\tC\n')
    out=module.enrich_pairs(source,annotation_dir=annotations,output_root=tmp_path/'out',min_size=1,backend="offline")
    tested=pd.read_csv(out/'tested_pairs.tsv',sep='\t').set_index('b')
    # Every CNV × target pair is tested; only the size endpoint's q marks significance.
    assert tested.significant.to_dict()=={'schema': True, 'gwaspri': False}
    assert set(tested.locus)=={'cnv1'}
    result=pd.read_csv(out/'enrichment_all_terms.tsv',sep='\t')
    assert len(result)==12  # 2 pairs × 3 libraries × 2 terms; zero hits enter BH.
    schema=result.loc[result.b=='schema']
    assert set(schema.background_size)=={4}  # Shared seed A removed; off-network Z excluded.
    assert set(result.loc[result.b=='gwaspri','background_size'])=={5}
    assert set(result.query_size)=={1}
    assert set(schema.loc[schema.term=='hit','term_size'])=={1}
    assert set(schema.loc[schema.term=='hit','p'])=={hypergeom.sf(0,4,1,1)}
    assert set(schema.loc[schema.term=='zero','p'])=={1.}
    assert list(result.q_all_pair_terms)==list(bh(result.p))
    background=pd.read_csv(out/'background_genes.tsv',sep='\t')
    assert set(background.loc[background.pair=='cnv1 × SCHEMA','symbol'])==set('BCDE')
    for stem in ('enrichment_schema', 'enrichment_gwas'):
        assert (out/f'{stem}.png').exists() and (out/f'{stem}.pdf').exists()
        assert (out/f'{stem}_all_terms.png').exists()  # Uncollapsed companion.
    # orsum accounts for every significant term in each plot.
    mapping=pd.read_csv(out/'enrichment_schema_term_mapping.tsv',sep='\t')
    significant=result.loc[(result.b=='schema')&(result.q_all_pair_terms<.05)]
    assert set(zip(mapping.library,mapping.represented))==set(zip(significant.library,significant.term))
    # No significant pairs still tests every CNV × target pair and plots them.
    out=module.enrich_pairs(source,annotation_dir=annotations,output_root=tmp_path/'out',min_size=1,fdr=.001,backend="offline")
    assert not pd.read_csv(out/'tested_pairs.tsv',sep='\t').significant.any()
    assert (out/'enrichment_schema.png').exists()


def test_cnv_cnv_keeps_only_significant_pairs(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    pd.DataFrame([['3q29', '16p11.2', 'netcoloc', 'size', 'ok', .01],
                  ['3q29', '22q11.2', 'netcoloc', 'size', 'ok', .2]],
        columns=['a', 'b', 'null', 'statistic', 'status', 'q']).to_csv(source/'pair_tests.tsv', sep='\t', index=False)
    pd.DataFrame([['3q29', '16p11.2', 1, 'A'], ['3q29', '22q11.2', 2, 'B']],
        columns=['a', 'b', 'entrez', 'symbol']).to_csv(source/'candidate_genes.tsv', sep='\t', index=False)
    pd.DataFrame({'entrez': [1, 2, 3]}).to_csv(source/'nps.tsv', sep='\t', index=False)
    manifest = {'status':'complete', 'identity': {
        'inputs': {'network':'test_brain'}, 'config': {'overlap':'bin', 'comparisons':'cnv_cnv'},
        'symbols': {'1':'A', '2':'B', '3':'C'},
        'entities': {name: {'kind':'cnv', 'seeds':[]} for name in ('3q29', '16p11.2', '22q11.2')}},
        'artifacts': {p.name:file_hash(p) for p in source.iterdir()}}
    (source/'manifest.json').write_text(json.dumps(manifest))
    annotations=tmp_path/'annotations'
    annotations.mkdir()
    for name in module.LIBRARIES.values():
        (annotations/name).write_text('hit\tdescription\tA\tB\n')
    out=module.enrich_pairs(source,annotation_dir=annotations,output_root=tmp_path/'out',min_size=1,backend="offline")
    tested=pd.read_csv(out/'tested_pairs.tsv',sep='\t')
    assert tested.locus.tolist()==['3q29 × 16p11.2']  # Chromosome order within the pair label.
    assert (out/'enrichment_cnv_cnv.png').exists()


def test_term_labels():
    assert module.term_label('Chemical Synaptic Transmission (GO:0007268)')=='Chemical Synaptic Transmission'
    assert module.term_label('Presynapse (GO:0098793) CC')=='Presynapse [CC]'
    assert module.locus_key('15q11.2-q13.1') < module.locus_key('15q13.3') < module.locus_key('16p11.2')


def test_enrichr_uses_service_pvalues_and_retains_zero_hits(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
    import enrichr_background as remote
    calls = []
    def fake_request(endpoint, payload, cache_dir, **kwargs):
        calls.append((endpoint, payload))
        if endpoint == 'addbackground':
            return {'backgroundid': 'bg'}
        if endpoint == 'addList':
            return {'userListId': 1}
        return {'SynGO_2024': [[1, 'hit', .0123, 3., 4., ['A'], .0246, 0, 0]]}
    monkeypatch.setattr(remote, 'request_cached', fake_request)
    result = remote.enrichr_ora({'A'}, {'A','B','C'},
        {'hit': ('', {'A'}), 'zero': ('', {'B'})}, library='SynGO synaptic function',
        cache_dir=tmp_path, min_size=1).set_index('term')
    assert result.loc['hit','p'] == .0123
    assert result.loc['hit','enrichr_adjusted_p'] == .0246
    assert result.loc['zero','p'] == 1.
    assert calls[0][1]['background'] == 'A\nB\nC'
    assert calls[2][1]['backgroundid'] == 'bg'
