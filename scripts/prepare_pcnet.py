"""Restore the pinned PCNet 2.2 node and edge snapshots from NDEx."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from cnv_overlap.provenance import atomic_path, file_hash, write_json

NETWORK_ID = '8b4b54fa-e87d-11ee-9621-005056ae23aa'
BASE_URL = f'https://www.ndexbio.org/v2/network/{NETWORK_ID}'
SNAPSHOTS = {
    'nodes': '7e31675836afab006389614e4b745840943f024c042c965ebf53f6cc3cab4dbc',
    'edges': '4bd6d5f9bee6f019dc4a5344dbc27e59769745236cf4b9fd0decb053a25fd19d',
}


def prepare_pcnet(root):
    """Download missing aspects atomically and verify the pinned byte checksums.

    Existing files are checked without network access or replacement. Reject a
    changed remote snapshot before publishing it; retain the frozen hashes.
    """
    directory = Path(root)/'input/interactome'
    # Validate all existing files before downloading anything.
    for aspect, expected in SNAPSHOTS.items():
        path = directory/f'pcnet22_{aspect}.json'
        if path.exists() and file_hash(path) != expected:
            raise ValueError(f'PCNet snapshot changed: {path}; restore the pinned archive')
    for aspect, expected in SNAPSHOTS.items():
        path = directory/f'pcnet22_{aspect}.json'
        if path.exists():
            continue
        url = f'{BASE_URL}/aspect/{aspect}'
        with atomic_path(path) as temporary:
            with urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            if file_hash(temporary) != expected:
                raise ValueError(f'NDEx {aspect} checksum differs from pinned PCNet 2.2; '
                                 'restore the archived snapshot or review a new version')
        write_json(directory/f'pcnet22_{aspect}.source.json', {
            'network': 'PCNet 2.2', 'ndex_uuid': NETWORK_ID,
            'url': url, 'sha256': expected,
            'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'format': 'CX v1 aspect array, unchanged response bytes',
        })
    return directory


def prepare_config_pcnet(config_path):
    """Restore standard PCNet paths used by a config; leave custom inputs alone."""
    config_path = Path(config_path).resolve()
    spec = json.loads(config_path.read_text())
    inputs = spec.get('inputs', {})
    if 'adjacency' in inputs or not all(
        inputs.get(f'network_{aspect}') == f'input/interactome/pcnet22_{aspect}.json'
        for aspect in SNAPSHOTS
    ):
        return
    prepare_pcnet((config_path.parent/spec.get('root', '..')).resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    print(prepare_pcnet(parser.parse_args().root))
