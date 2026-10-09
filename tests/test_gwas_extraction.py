import importlib.util
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from cnv_overlap.provenance import file_hash


def test_custom_trait_filters_mapping_and_snapshot(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/'scripts/gwas_catalog.py'
    spec = importlib.util.spec_from_file_location('gwas_extraction_test', script)
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    monkeypatch.setattr(gc, 'ROOT', tmp_path)
    studies = pd.DataFrame([
        {'STUDY ACCESSION': name, 'MAPPED BACKGROUND TRAIT': background,
         'GENOTYPING TECHNOLOGY': tech, 'DISEASE/TRAIT': 'example', 'MAPPED_TRAIT': 'example',
         'MAPPED_TRAIT_URI': 'http://example/EFO_123', 'ASSOCIATION COUNT': 10}
        for name, background, tech in [('good', None, gc.GENOME_WIDE[0]),
            ('background', 'other', gc.GENOME_WIDE[0]), ('badtech', None, 'Exome sequencing')]])
    studies.to_csv(tmp_path/'studies.tsv', sep='\t', index=False)
    rows = []
    for gene, p, study, intergenic in [('ENSG1', 0., 'good', 0), ('ENSG2', 1e-9, 'good', 0),
        ('ENSG3', 2e-9, 'good', 0), ('ENSG4', 1e-10, 'good', 0),  # conflicting mapping
        ('ENSG5', .01, 'good', 0), ('ENSG6', 1e-10, 'background', 0),
        ('ENSG6', 1e-10, 'badtech', 0), ('ENSG6', 1e-10, 'good', 1),
        ('ENSG5,ENSG6', 1e-10, 'good', 0)]:
        rows.append({'DISEASE/TRAIT': 'example', 'MAPPED_GENE': gene, 'SNP_GENE_IDS': gene,
                     'P-VALUE': p, 'MAPPED_TRAIT': 'example', 'MAPPED_TRAIT_URI': 'http://example/EFO_123',
                     'INTERGENIC': intergenic, 'STUDY ACCESSION': study})
    with zipfile.ZipFile(tmp_path/'associations.zip', 'w') as z:
        z.writestr('folder/', '')
        z.writestr('folder/associations.tsv', pd.DataFrame(rows).to_csv(sep='\t', index=False))
    pd.DataFrame({'ensembl_gene_id': ['ENSG1', 'ENSG2', 'ENSG3', 'ENSG4', 'ENSG4', 'ENSG5', 'ENSG6'],
                  'entrez': [1, 2, 3, 4, 5, 5, 6]}).to_csv(tmp_path/'mapping.tsv', sep='\t', index=False)
    (tmp_path/'nodes.json').write_text(json.dumps([{'r': f'ncbigene:{i}'} for i in range(1, 7)]))
    pd.DataFrame([{'entity': 'cnv_a', 'kind': 'cnv', 'entrez': 5, 'source': 'fixture'},
                  {'entity': 'old', 'kind': 'target', 'entrez': 6, 'source': 'fixture'}]).to_csv(tmp_path/'seeds.tsv', sep='\t', index=False)
    inputs = {'seeds': 'seeds.tsv', 'network_nodes': 'nodes.json'}
    base = {'root': '.', 'inputs': inputs, 'input_sha256': {k: file_hash(tmp_path/v) for k, v in inputs.items()},
            'analysis': {'comparisons': 'cnv_disorder'}, 'seed_definition': 'fixture'}
    (tmp_path/'base.json').write_text(json.dumps(base))
    argv = ['--trait', 'newtrait=EFO_123', '--studies', str(tmp_path/'studies.tsv'),
            '--associations', str(tmp_path/'associations.zip'), '--mapping', str(tmp_path/'mapping.tsv'),
            '--base-config', str(tmp_path/'base.json'), '--output-dir', str(tmp_path/'snapshot'), '--max-seeds', '2']
    gc.main(argv)
    new = pd.read_csv(tmp_path/'snapshot/seeds.tsv', sep='\t')
    assert set(new.loc[new.entity == 'newtrait', 'entrez']) == {1, 2}
    assert set(new.entity) == {'cnv_a', 'old', 'newtrait'}
    assert (tmp_path/'base.json').read_text() == json.dumps(base)
    qc = pd.read_csv(tmp_path/'snapshot/trait_qc.tsv', sep='\t').iloc[0]
    assert (qc.n_genes_reported, qc.n_unmapped, qc.n_studies, qc.n_selected) == (4, 1, 1, 2)
    config = json.loads((tmp_path/'snapshot/analysis.json').read_text())
    for key, p in config['inputs'].items():
        assert file_hash(tmp_path/p) == config['input_sha256'][key]
    with pytest.raises(SystemExit):
        gc.main(argv)  # refuses to overwrite a frozen snapshot
