"""Freeze SynGO and HPA annotation snapshots for offline analysis."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

from cnv_overlap.provenance import atomic_path, file_hash, write_json

ENRICHR = 'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={}'
ENRICHR_LIBRARIES = {
    'SynGO_2024.gmt': ('SynGO_2024', None),
}
HPA_URL = 'https://www.proteinatlas.org/download/proteinatlas.tsv.zip'
HPA_ARCHIVE = 'proteinatlas.tsv.zip'
ELEVATED = {'Tissue enriched', 'Group enriched', 'Tissue enhanced', 'Region enriched', 'Region enhanced',
            'Cell type enriched', 'Cell type enhanced'}
# GMT file -> (HPA specificity column, HPA column listing the elevated contexts).
# The single-cell-type GMT is only the source of HPA_brain_broad_cell_types.gmt.
HPA_DIMENSIONS = {
    'HPA_brain_region_elevated.gmt': ('RNA brain regional specificity', 'RNA brain regional specific nTPM'),
    'HPA_brain_cell_type_elevated.gmt': ('RNA single nuclei brain specificity',
                                         'RNA single nuclei brain specific nCPM'),
    'HPA_single_cell_type_elevated.gmt': ('RNA single cell type specificity',
                                          'RNA single cell type specific nCPM'),
}
HPA_GENES = 'HPA_gene_specificity.tsv'


def download(url):
    with urllib.request.urlopen(url, timeout=600) as response:
        return response.read(), dict(response.headers)


def write_text(path, text):
    with atomic_path(path) as tmp:
        tmp.write_text(text)


def elevated_contexts(value):
    """'brain: 215.2;retina: 30.1' -> ['brain', 'retina']."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.rsplit(':', 1)[0].strip() for item in value.split(';') if item.strip()]


def hpa_gene_sets(table, category_column, context_column):
    sets = {}
    for record in table[['Gene', category_column, context_column]].itertuples(index=False):
        gene, category, contexts = record
        if category not in ELEVATED:
            continue
        for context in elevated_contexts(contexts):
            sets.setdefault(context, set()).add(gene)
    return dict(sorted(sets.items()))


def fetch(annotation_dir):
    annotation_dir = Path(annotation_dir)
    sources_path = annotation_dir/'sources.json'
    sources = json.loads(sources_path.read_text())
    now = lambda: datetime.now(timezone.utc).isoformat()
    for filename, (library, keep) in ENRICHR_LIBRARIES.items():
        path = annotation_dir/filename
        if path.exists():
            print(f'Keeping existing {filename}')
            continue
        raw, _ = download(ENRICHR.format(library))
        lines = [line for line in raw.decode().splitlines() if line.strip()]
        if keep is not None:
            lines = [line for line in lines if keep(line.split('\t', 1)[0])]
        write_text(path, '\n'.join(lines) + '\n')
        sources[filename] = {'url': ENRICHR.format(library), 'provider': 'Enrichr', 'library': library,
            'retrieved_utc': now(), 'sha256': file_hash(path), 'identifiers': 'human gene symbols',
            'term_count': len(lines),
            **({'filter': 'terms starting "Human " and ending " up" (human MTG cell-type markers)'}
               if keep is not None else {})}
        print(f'Wrote {filename}: {len(lines)} terms')
    archive = annotation_dir/HPA_ARCHIVE
    if not archive.exists():
        raw, headers = download(HPA_URL)
        with atomic_path(archive) as tmp:
            tmp.write_bytes(raw)
        sources[HPA_ARCHIVE] = {'url': HPA_URL, 'provider': 'Human Protein Atlas',
            'last_modified': headers.get('Last-Modified'), 'retrieved_utc': now(),
            'sha256': file_hash(archive), 'identifiers': 'HGNC symbols and Ensembl gene IDs',
            'note': 'HPA release number is not embedded in the file; identify it by the checksum/date'}
    with zipfile.ZipFile(archive) as z:
        table = pd.read_csv(io.BytesIO(z.read('proteinatlas.tsv')), sep='\t', dtype=str)
    for filename, (category_column, context_column) in HPA_DIMENSIONS.items():
        path = annotation_dir/filename
        if path.exists():
            print(f'Keeping existing {filename}')
            continue
        sets = hpa_gene_sets(table, category_column, context_column)
        write_text(path, ''.join(f'{context}\tHPA elevated ({category_column})\t' + '\t'.join(sorted(genes)) + '\n'
                                 for context, genes in sets.items()))
        sources[filename] = {'derived_from': HPA_ARCHIVE, 'archive_sha256': sources[HPA_ARCHIVE]['sha256'],
            'provider': 'Human Protein Atlas', 'sha256': file_hash(path), 'term_count': len(sets),
            'definition': f'genes whose "{category_column}" is enriched, group enriched or enhanced, '
                          f'assigned to each context listed in "{context_column}"'}
        print(f'Wrote {filename}: {len(sets)} contexts')
    genes_path = annotation_dir/HPA_GENES
    if not genes_path.exists():
        columns = ['Gene', 'Ensembl', 'RNA tissue specificity', 'RNA tissue specific nTPM'] + [
            c for pair in HPA_DIMENSIONS.values() for c in pair]
        with atomic_path(genes_path) as tmp:
            table[columns].to_csv(tmp, sep='\t', index=False)
        sources[HPA_GENES] = {'derived_from': HPA_ARCHIVE, 'archive_sha256': sources[HPA_ARCHIVE]['sha256'],
            'provider': 'Human Protein Atlas', 'sha256': file_hash(genes_path),
            'definition': 'per-gene HPA RNA specificity categories and elevated contexts'}
    write_json(sources_path, sources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotation-dir', default=Path(__file__).resolve().parents[1]/'input/annotations')
    fetch(parser.parse_args(argv).annotation_dir)


if __name__ == '__main__':
    main()
