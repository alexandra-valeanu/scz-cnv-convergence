"""PCNet restoration must preserve the pinned snapshot and work offline on reuse."""
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_pcnet as pcnet


@pytest.fixture
def snapshots(monkeypatch):
    data = {'nodes': b'[{"@id":1}]', 'edges': b'[]'}
    monkeypatch.setattr(pcnet, 'SNAPSHOTS', {
        name: hashlib.sha256(value).hexdigest() for name, value in data.items()})
    return data


def test_download_verify_and_offline_reuse(tmp_path, monkeypatch, snapshots):
    calls = []
    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(snapshots[url.rsplit('/', 1)[1]])
    monkeypatch.setattr(pcnet, 'urlopen', fetch)
    directory = pcnet.prepare_pcnet(tmp_path)
    assert len(calls) == 2
    for name, value in snapshots.items():
        assert (directory/f'pcnet22_{name}.json').read_bytes() == value
        source = json.loads((directory/f'pcnet22_{name}.source.json').read_text())
        assert source['sha256'] == pcnet.SNAPSHOTS[name]
        assert source['url'].endswith('/aspect/'+name)
    monkeypatch.setattr(pcnet, 'urlopen', lambda *a, **k: pytest.fail('Offline reuse contacted NDEx'))
    pcnet.prepare_pcnet(tmp_path)
    (directory/'pcnet22_edges.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='snapshot changed'):
        pcnet.prepare_pcnet(tmp_path)
    assert (directory/'pcnet22_edges.json').read_bytes() == b'changed'


@pytest.mark.parametrize('failure', ['mismatch', 'interrupted'])
def test_failed_download_is_not_published(tmp_path, monkeypatch, snapshots, failure):
    class Interrupted(io.BytesIO):
        def read(self, *args):
            raise OSError('interrupted transfer')
    monkeypatch.setattr(pcnet, 'urlopen', lambda *a, **k:
                        Interrupted() if failure == 'interrupted' else io.BytesIO(b'changed'))
    with pytest.raises((ValueError, OSError)):
        pcnet.prepare_pcnet(tmp_path)
    assert not list((tmp_path/'input/interactome').iterdir())


def test_config_restores_only_standard_inputs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pcnet, 'prepare_pcnet', calls.append)
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'root': '.', 'inputs': {
        f'network_{a}': f'input/interactome/pcnet22_{a}.json' for a in pcnet.SNAPSHOTS}}))
    pcnet.prepare_config_pcnet(config)
    assert calls == [tmp_path]
    config.write_text(json.dumps({'inputs': {'network_nodes': 'custom.json'}}))
    pcnet.prepare_config_pcnet(config)
    assert calls == [tmp_path]
