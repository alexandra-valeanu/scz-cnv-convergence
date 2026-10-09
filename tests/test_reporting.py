import json
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pytest
from cnv_overlap.reporting import export_report
from cnv_overlap.pipeline import run
from cnv_overlap.enrichment import annotate_run
from test_workflow import fixture, settings


def test_report_and_annotation_integrity(tmp_path):
    inputs = fixture()
    result = run(*inputs, config=settings(), output_root=tmp_path/'runs',
                 cache_dir=tmp_path/'cache', progress=lambda _: None)
    gmt = tmp_path/'test.gmt'
    gmt.write_text('pathway\tdescription\t'+'\t'.join(list(inputs[2].values())[:12])+'\n')
    before = (result/'manifest.json').read_bytes()
    report = export_report(result, inputs, output_dir=tmp_path/'report', gmt_paths=[gmt], fdr=1)
    assert (report/'region_pair_excess.pdf').exists()
    assert (report/'locus_target_observed.png').exists()
    assert (report/'enrichment_summary.pdf').exists()
    assert (report/'enrichment/annotation_coverage.tsv').exists()
    assert (result/'manifest.json').read_bytes() == before
    assert list((report/'networks').glob('*/network.graphml'))
    with pytest.raises(ValueError, match='outside'):
        export_report(result, inputs, output_dir=result/'report')
    (result/'candidate_genes.tsv').write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        export_report(result, inputs, output_dir=tmp_path/'bad')


def test_annotation_excludes_shared_seeds_from_background(tmp_path):
    pd.DataFrame([{'a':'a','b':'b','symbol':'B'}]).to_csv(tmp_path/'candidate_genes.tsv',sep='\t',index=False)
    pd.DataFrame([{'a':'a','b':'b'}]).to_csv(tmp_path/'hypotheses.tsv',sep='\t',index=False)
    (tmp_path/'manifest.json').write_text(json.dumps({'identity':{'config':{'overlap':'remove'},
      'entities':{'a':{'seeds':[1]},'b':{'seeds':[1]}},'symbols':{'1':'A'}}}))
    gmt=tmp_path/'terms.gmt'
    gmt.write_text('term\tdesc\tA\tB\n')
    table=annotate_run(tmp_path,[gmt],['A','B','C'],output_dir=tmp_path/'annotation',min_size=1)
    assert table.iloc[0].background_size == 2
    assert table.iloc[0].term_size == 1
    assert table.iloc[0].p == .5
