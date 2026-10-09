"""Cached Enrichr custom-background requests; preserve raw responses and GMTs."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cnv_overlap.enrichment import ora
from cnv_overlap.provenance import digest, file_hash, write_json

BASE = 'https://maayanlab.cloud/speedrichr/api/'
ENRICHR = 'https://maayanlab.cloud/Enrichr/'
LIBRARY_IDS = {
    'SynGO synaptic function': 'SynGO_2024',
    'GO Biological Process': 'GO_Biological_Process_2026',
}


def session():
    client = requests.Session()
    client.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET', 'POST'])))
    return client


def prepare_libraries(cache_dir, library_ids=None):
    """Freeze live Enrichr GMTs once; ``library_ids`` maps label → Enrichr library name."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with session() as client:
        for label, library in (library_ids or LIBRARY_IDS).items():
            path = cache_dir/f'{library}.gmt'
            if not path.exists():
                response = client.get(ENRICHR+'geneSetLibrary',
                    params={'mode':'text', 'libraryName':library}, timeout=60)
                response.raise_for_status()
                lines = response.text.splitlines()
                if not lines or any(len(line.split('\t')) < 3 for line in lines):
                    raise ValueError(f'Invalid Enrichr GMT: {library}')
                path.write_text('\n'.join(lines)+'\n')
                write_json(path.with_suffix('.json'), {'url':response.url, 'sha256':file_hash(path),
                    'retrieved_utc':datetime.now(timezone.utc).isoformat()})
            record = json.loads(path.with_suffix('.json').read_text())
            if file_hash(path) != record['sha256']:
                raise ValueError(f'Cached Enrichr GMT changed: {path}')
            paths[label] = path
    return paths


def request_cached(endpoint, payload, cache_dir, *, multipart=False):
    path = Path(cache_dir)/f'{endpoint}-{digest(payload)}.json'
    if path.exists():
        return json.loads(path.read_text())
    with session() as client:
        options = {'files': {key:(None,str(value)) for key,value in payload.items()}} if multipart else {'data':payload}
        response = client.post(BASE+endpoint, timeout=60, **options)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or 'error' in result:
            raise ValueError(f'Enrichr request failed: {result}')
        path.write_text(response.text)
        write_json(path.with_suffix('.metadata.json'), {'url':BASE+endpoint,
            'retrieved_utc':datetime.now(timezone.utc).isoformat(), 'sha256':file_hash(path)})
        return result


def enrichr_ora(query, background, terms, *, library, cache_dir, min_size):
    """Use Enrichr p-values; keep service FDR and recompute combined BH upstream.

    Local intersections give counts/fold enrichment and the eligible test family.
    Missing zero-hit terms have p=1; a missing nonzero-hit term is an error.
    """
    # Enrichr treats submitted symbols case-insensitively and returns uppercase.
    query = {g.upper() for g in query}
    background = {g.upper() for g in background}
    result = ora(query, background, terms, min_size=min_size).drop(columns='q')
    result['p_local_hypergeometric'] = result.p
    result['enrichr_adjusted_p'] = 1.
    result['enrichr_odds_ratio'] = float('nan')
    result['enrichr_combined_score'] = float('nan')
    if not query or result.empty:
        return result
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bg = request_cached('addbackground', {'background':'\n'.join(sorted(background))}, cache_dir, multipart=True)
    q = request_cached('addList', {'list':'\n'.join(sorted(query)),
        'description':'Brain-network significant pair candidate genes'}, cache_dir, multipart=True)
    library_id = LIBRARY_IDS[library]
    response = request_cached('backgroundenrich', {'userListId':q['userListId'],
        'backgroundid':bg['backgroundid'], 'backgroundType':library_id}, cache_dir)
    if library_id not in response:
        raise ValueError(f'Enrichr response lacks {library_id}')
    records = {row[1]:row for row in response[library_id]}
    for i, row in result.iterrows():
        record = records.get(row.term)
        if record is None:
            if row.overlap:
                raise ValueError(f'Enrichr omitted a term with query hits: {library_id}: {row.term}')
            result.loc[i, 'p'] = 1.
            continue
        if set(record[5]) != set(row.genes.split(',')) - {''}:
            raise ValueError(f'Enrichr/GMT overlap mismatch: {library_id}: {row.term}')
        result.loc[i, ['p', 'enrichr_odds_ratio', 'enrichr_combined_score', 'enrichr_adjusted_p']] = [record[2], record[3], record[4], record[6]]
    result['enrichr_library'] = library_id
    if not pd.to_numeric(result.p).between(0, 1).all():
        raise ValueError('Invalid Enrichr p-values')
    return result
