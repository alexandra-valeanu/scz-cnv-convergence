"""Freeze MitoCarta pathways and broad HPA brain cell classes for publication."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from urllib.request import urlopen

from cnv_overlap.enrichment import load_gmt
from cnv_overlap.provenance import atomic_path, file_hash, write_json

MITO_URL = 'https://personal.broadinstitute.org/scalvo/MitoCarta_Download/MitoPathways3.0.gmx'
CELL_TYPES = ('Brain excitatory neurons', 'Brain inhibitory neurons', 'Astrocytes',
              'Oligodendrocytes', 'Oligodendrocyte progenitor cells', 'Microglia',
              'Ependymal cells', 'Other brain neurons')


def parse_gmx(text):
    rows = list(csv.reader(io.StringIO(text), delimiter='\t'))
    if len(rows) < 3 or len(set(rows[0])) != len(rows[0]):
        raise ValueError('Invalid MitoCarta GMX header')
    return {name: {row[i].strip() for row in rows[2:] if i < len(row) and row[i].strip()}
            for i, name in enumerate(rows[0]) if name}


def prepare(directory):
    directory = Path(directory)
    sources = json.loads((directory/'sources.json').read_text())
    source = directory/'HPA_single_cell_type_elevated.gmt'
    terms = load_gmt(source)
    if set(CELL_TYPES) - terms.keys():
        raise ValueError('HPA snapshot lacks required broad brain cell types')
    cell_path = directory/'HPA_brain_broad_cell_types.gmt'
    text = ''.join('\t'.join([t, terms[t][0], *sorted(terms[t][1])])+'\n' for t in CELL_TYPES)
    if cell_path.exists() and cell_path.read_text() != text:
        raise ValueError('Broad cell-type snapshot differs; use a new annotation directory')
    cell_path.write_text(text)
    sources[cell_path.name] = {'derived_from': source.name, 'source_sha256': file_hash(source),
        'sha256': file_hash(cell_path), 'definition': 'Exact HPA broad brain cell labels; no pooling of subtype markers',
        'cell_types': CELL_TYPES}
    raw = directory/'MitoPathways3.0.gmx'
    if not raw.exists():
        with urlopen(MITO_URL, timeout=120) as response, atomic_path(raw) as tmp:
            tmp.write_bytes(response.read())
    if raw.name in sources and sources[raw.name]['sha256'] != file_hash(raw):
        raise ValueError('MitoCarta source checksum changed')
    pathways = parse_gmx(raw.read_text())
    if len(pathways) != 149:
        raise ValueError(f'Expected 149 MitoCarta3.0 pathways, found {len(pathways)}')
    mito = directory/'MitoCarta3.0.gmt'
    content = ''.join('\t'.join([t, 'MitoCarta3.0', *sorted(g)])+'\n' for t, g in pathways.items())
    if mito.exists() and mito.read_text() != content:
        raise ValueError('MitoCarta GMT differs; use a new annotation directory')
    mito.write_text(content)
    sources.setdefault(raw.name, {'url': MITO_URL, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                                  'sha256': file_hash(raw), 'provider': 'Broad Institute'})
    sources[mito.name] = {'derived_from': raw.name, 'source_sha256': file_hash(raw),
        'sha256': file_hash(mito), 'term_count': len(pathways), 'identifiers': 'human gene symbols'}
    write_json(directory/'sources.json', sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-dir', type=Path, default=Path(__file__).resolve().parents[1]/'input/annotations')
    prepare(parser.parse_args().annotation_dir)
