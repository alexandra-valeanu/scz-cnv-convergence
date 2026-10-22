"""Prepare frozen PCNet/STRING networks and select notebook inputs."""
import json
import gzip
from collections import defaultdict
from urllib.request import urlopen
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cnv_overlap.inputs import load_inputs
from cnv_overlap.provenance import atomic_path, file_hash, write_json
from cnv_overlap.results import network_root
from results_layout import FOLDERS

BASE_NETWORKS = ('PCNET2.2', 'STRING_high_conf')
NETWORKS = BASE_NETWORKS + tuple(n + '_brain' for n in BASE_NETWORKS)
STRING_VERSION = '12.0'
STRING_TAXON = 9606
STRING_MIN_SCORE = 700
STRING_DOWNLOAD_BASE = 'https://stringdb-downloads.org/download'
STRING_PHYSICAL_FILE = f'{STRING_TAXON}.protein.physical.links.detailed.v{STRING_VERSION}.txt.gz'
STRING_ALIAS_FILE = f'{STRING_TAXON}.protein.aliases.v{STRING_VERSION}.txt.gz'
STRING_PHYSICAL_URL = f'{STRING_DOWNLOAD_BASE}/protein.physical.links.detailed.v{STRING_VERSION}/{STRING_PHYSICAL_FILE}'
STRING_ALIAS_URL = f'{STRING_DOWNLOAD_BASE}/protein.aliases.v{STRING_VERSION}/{STRING_ALIAS_FILE}'
# Gene sets with fewer seeds inside the selected network are excluded from that network's run.
MIN_SEEDS = 3


def download_string_file(url, destination):
    """Publish complete HTTP downloads atomically; interrupted transfers are retryable."""
    destination = Path(destination)
    if not destination.exists():
        with atomic_path(destination) as temporary:
            with urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
    return destination


def read_string_physical(path):
    """Read STRING physical links with combined scores >=700 (scale 0–1000)."""
    columns = ['protein1', 'protein2', 'experimental', 'database', 'textmining', 'combined_score']
    frames = []
    for chunk in pd.read_csv(path, sep=r'\s+', compression='gzip', chunksize=100_000):
        if not set(columns) <= set(chunk):
            raise ValueError('Malformed STRING physical interaction table: missing required columns')
        chunk = chunk[columns].copy()
        for name in columns[2:]:
            values = pd.to_numeric(chunk[name], errors='raise')
            if values.isna().any() or ((values < 0) | (values > 1000) | (values != np.floor(values))).any():
                raise ValueError(f'Invalid STRING score column: {name}')
            chunk[name] = values.astype('int64')
        for name in columns[:2]:
            if not chunk[name].astype('string').str.fullmatch(r'9606\.ENSP[0-9]+').fillna(False).all():
                raise ValueError('STRING physical file must contain human (9606) ENSP identifiers')
            chunk[name] = chunk[name].str.removeprefix('9606.')
        frames.append(chunk.loc[chunk.combined_score >= STRING_MIN_SCORE])
    if not frames or not any(len(frame) for frame in frames):
        raise ValueError('No high-confidence STRING physical interactions found')
    return pd.concat(frames, ignore_index=True).rename(columns={'protein1': 'protein_a', 'protein2': 'protein_b'})


def mapped_string_graph(frame):
    """Binary undirected Entrez graph; remove unmapped endpoints and self edges."""
    pairs = frame[['entrez_a', 'entrez_b']].apply(pd.to_numeric, errors='coerce').dropna()
    if ((pairs <= 0) | (pairs != np.floor(pairs))).any().any():
        raise ValueError('STRING mapping must contain positive integer Entrez IDs')
    pairs = pairs.astype('int64')
    pairs = pairs[pairs.entrez_a != pairs.entrez_b]
    edges = sorted({tuple(sorted(pair)) for pair in pairs.itertuples(index=False, name=None)})
    if not edges:
        raise ValueError('No mapped STRING interactions remain')
    nodes = sorted({g for pair in edges for g in pair})
    pos = {g: i for i, g in enumerate(nodes)}
    i, j = np.array([(pos[a], pos[b]) for a, b in edges]).T
    matrix = sp.coo_matrix((np.ones(2*len(i)), (np.r_[i, j], np.r_[j, i])),
                           shape=(len(nodes), len(nodes))).tocsr()
    return matrix, nodes


def prepare_string(root):
    """Download once; cache scores and unambiguous mappings with checksums."""
    directory = Path(root)/'input/interactome/STRING_high_conf'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, checksum in manifest['artifacts'].items():
            if file_hash(directory/name) != checksum:
                raise ValueError(f'STRING snapshot changed: {name}')
        return directory
    # Existing sealed snapshots remain valid and keep their original provenance.
    # An unsealed CSV alone is not trusted as evidence of physical-network provenance.
    physical_path = download_string_file(STRING_PHYSICAL_URL, directory/STRING_PHYSICAL_FILE)
    frame = read_string_physical(physical_path)
    with atomic_path(directory/'string.csv') as temporary:
        frame.to_csv(temporary, index=False)
    proteins = sorted(set(frame.protein_a) | set(frame.protein_b))
    map_path = directory/'protein_mapping.tsv'
    alias_path = download_string_file(STRING_ALIAS_URL, directory/STRING_ALIAS_FILE)
    # Rebuild derived mappings from the frozen alias source until the snapshot is sealed.
    identifiers, gene_symbols = defaultdict(set), defaultdict(set)
    with gzip.open(alias_path, 'rt') as stream:
        next(stream)
        for line in stream:
            protein, alias, sources = line.rstrip('\n').split('\t')
            protein = protein.removeprefix('9606.')
            sources = set(sources.split())
            if sources & {'Ensembl_HGNC_entrez_id', 'Ensembl_EntrezGene', 'UniProt_DR_GeneID'} and alias.isdigit():
                identifiers[protein].add(alias)
            if 'Ensembl_HGNC_symbol' in sources:
                gene_symbols[protein].add(alias)
    rows = []
    for protein in proteins:
        ensp = protein.removeprefix('9606.')
        ids, symbols = sorted(identifiers[ensp]), sorted(gene_symbols[ensp])
        rows.append({'protein': protein, 'entrez': ids[0] if len(ids) == 1 else '',
                     'symbol': symbols[0] if len(symbols) == 1 else '',
                     'entrez_candidates': ','.join(ids), 'symbol_candidates': ','.join(symbols)})
    protein_map = pd.DataFrame(rows)
    protein_map.to_csv(map_path, sep='\t', index=False)
    lookup = protein_map.set_index('protein')
    for side, label in [('a', 'source'), ('b', 'target')]:
        frame[f'entrez_{side}'] = frame[f'protein_{side}'].map(lookup.entrez)
        frame[label] = frame[f'protein_{side}'].map(lookup.symbol)
    frame = frame.rename(columns={'database': 'database_score'})
    frame['database'] = 'STRING'
    frame.to_csv(directory/'string_mapped.csv', index=False)
    matrix, nodes = mapped_string_graph(frame)
    # Keep only unique symbol assignments; an Entrez label is preferable to an arbitrary mapping.
    symbols = {}
    for gene, group in protein_map[protein_map.entrez.ne('')].groupby('entrez'):
        values = set(group.symbol)-{''}
        if len(values) == 1:
            symbols[int(gene)] = next(iter(values))
    pd.DataFrame({'entrez': nodes, 'symbol': [symbols.get(g, str(g)) for g in nodes]}).to_csv(
        directory/'nodes.tsv', sep='\t', index=False)
    sp.save_npz(directory/'adjacency.npz', matrix)
    write_json(manifest_path, {'network': 'STRING', 'interaction_type': 'physical', 'taxon': 9606,
        'score_threshold': 'high_confidence', 'minimum_combined_score': STRING_MIN_SCORE,
        'string_version': STRING_VERSION, 'retrieval_method': 'direct_https_bulk',
        'physical_source_url': STRING_PHYSICAL_URL, 'mapping_source_url': STRING_ALIAS_URL,
        'mapping_policy': 'unique ENSP to Entrez only; ambiguous/unmapped excluded',
        'raw_interactions': len(frame), 'mapped_nodes': len(nodes), 'mapped_edges': matrix.nnz//2,
        'artifacts': {p.name: file_hash(p) for p in directory.iterdir() if p.is_file() and p != manifest_path}})
    return directory


def drop_small_seed_sets(inputs, provenance, min_seeds):
    """Exclude gene sets below the retained-seed minimum and record exclusions."""
    A, nodes, symbols, entities, coordinates, regions = inputs
    universe = set(nodes.tolist())
    excluded = {}
    for name, entity in entities.items():
        retained = len(set(entity['seeds']) & universe)
        if retained < min_seeds:
            excluded[name] = retained
    if excluded:
        print(f'Excluded gene sets with < {min_seeds} seeds in the network:', excluded)
        entities = {name: e for name, e in entities.items() if name not in excluded}
        provenance = {**provenance, 'min_seeds': min_seeds, 'excluded_below_min_seeds': excluded}
    return (A, nodes, symbols, entities, coordinates, regions), provenance


def load_network_inputs(config_path, network=None, min_seeds=MIN_SEEDS, target_scope='default'):
    """Return (spec, root, inputs, provenance) for the selected network.

    Prepare missing network inputs and derived configs, then exclude small seed
    sets. Brain networks keep SCHEMA/GWAS targets unless target_scope is all."""
    if target_scope not in ('default', 'all'):
        raise ValueError('target_scope must be default or all')
    if network is None or network in ('PCNET2.2', 'PCNET2.2_brain'):
        from prepare_pcnet import prepare_config_pcnet
        prepare_config_pcnet(config_path)
    if network is None:
        spec, root, inputs, provenance = load_inputs(config_path)
        inputs, provenance = drop_small_seed_sets(inputs, provenance, min_seeds)
        spec.setdefault('results_root', str(Path(spec['output_root']).parent))
        return spec, root, inputs, provenance
    if network not in NETWORKS:
        raise ValueError(f'Unknown network {network!r}; choose {NETWORKS}')
    config_path = Path(config_path).resolve()
    spec = json.loads(config_path.read_text())
    root = (config_path.parent/spec.get('root', '..')).resolve()
    base_network = network.removesuffix('_brain')
    if base_network == 'STRING_high_conf':
        # Validate the original snapshot before deriving the alternative network configuration.
        for key, checksum in spec.get('input_sha256', {}).items():
            if file_hash(root/spec['inputs'][key]) != checksum:
                raise ValueError(f'Input changed: {key}')
        directory = prepare_string(root)
        for key in ('network_edges', 'network_nodes', 'adjacency', 'nodes'):
            spec['inputs'].pop(key, None)
            spec.get('input_sha256', {}).pop(key, None)
        for key, filename in [('adjacency', 'adjacency.npz'), ('nodes', 'nodes.tsv'), ('network_manifest', 'manifest.json')]:
            path = directory/filename
            spec['inputs'][key] = str(path.relative_to(root))
            spec.setdefault('input_sha256', {})[key] = file_hash(path)
        spec['root'] = str(root)
        spec['cache_dir'] = str(Path('data/cache')/network)
        derived =root/'data/network_configs'/network/config_path.name
        write_json(derived, spec)
        config_path = derived
    if network.endswith('_brain'):
        from brain_inputs import prepare_brain_universe
        # Verify source inputs before building a derived config with its own identity.
        for key, checksum in spec.get('input_sha256', {}).items():
            if file_hash(root/spec['inputs'][key]) != checksum:
                raise ValueError(f'Input changed: {key}')
        universe, manifest = prepare_brain_universe(root, spec, network)
        for key, path in [('gene_universe', universe), ('brain_filter', manifest)]:
            spec['inputs'][key] = str(path.relative_to(root))
            spec.setdefault('input_sha256', {})[key] = file_hash(path)
        spec['root'] = str(root)
        derived = root/'data/network_configs'/network/config_path.name
        write_json(derived, spec)
        config_path = derived
    spec, root, inputs, provenance = load_inputs(config_path)
    if network.endswith('_brain') and target_scope == 'default':
        A, nodes, symbols, entities, coordinates, regions = inputs
        entities = {n: e for n, e in entities.items()
                    if e['kind'] == 'cnv' or (e['kind'] == 'target' and n in ('schema', 'gwaspri'))}
        inputs = A, nodes, symbols, entities, coordinates, regions
        provenance['brain_target_scope'] = ['schema', 'gwaspri']
    # results/<network>/<group>/<analysis>; see scripts/results_layout.py.
    spec['results_root'] = str(network_root(network))
    name = Path(spec['output_root']).name
    if name not in FOLDERS:
        name = {'cnv_cnv': 'cnv_convergence', 'cnv_disorder': 'cnv_disorder_overlap'}.get(spec.get('analysis', {}).get('comparisons'), name)
    spec['output_root'] = str(Path(spec['results_root'])/FOLDERS.get(name, name))
    # Assign the network cache after loading the source configuration.
    spec['cache_dir'] = str(Path('data/cache')/network)
    provenance['network'] = network
    inputs, provenance = drop_small_seed_sets(inputs, provenance, min_seeds)
    return spec, root, inputs, provenance
