"""Multi-CNV null draws restricted to the fixed SCHEMA-GWAS network region (G_SZ).

Kept separate from the analysis code: its file hash is part of the null identity,
so changing reporting or pathway code never recomputes the draws.

The degree null reproduces run_convergence exactly (same RNG identity, sampling
order and reference profiles), and additionally stores, per draw and CNV, which
G_SZ genes are reached (NPS > threshold) and which are the CNV's own seeds.
"""
import json
from pathlib import Path

import numpy as np

from cnv_overlap.provenance import atomic_path, digest, file_hash, run_dir, write_json
from cnv_overlap.statistics import rng_for

NULL_MODELS = ('degree', 'region', 'degree_no_scz')
BLOCK = 250


def _packed(masks):
    return np.packbits(np.asarray(masks, dtype=bool), axis=-1)


def _unpacked(packed, n):
    return np.unpackbits(packed, axis=-1, count=n).astype(bool)


def observed_masks(prop, seeds, gsz_idx, threshold):
    """Per-CNV reach and own-seed masks over G_SZ for the observed seed sets."""
    names = sorted(seeds)
    reach = np.stack([np.asarray(prop.nps(seeds[n]))[gsz_idx] > threshold for n in names])
    own = np.stack([np.isin(gsz_idx, prop.seed_indices(seeds[n])) for n in names])
    return names, reach, own


def core_null(prop, seeds, gsz_idx, *, config, reps, null_model, output_root,
              windows=None, excluded=(), progress=print):
    """Joint CNV seed randomizations; one directory per identity, resumable by block.

    seeds: retained observed seeds per CNV (for degree_no_scz, already without `excluded`).
    excluded: genes never sampled as random seeds (degree_no_scz only).
    """
    if null_model not in NULL_MODELS or reps < 2:
        raise ValueError(f'Choose one of {NULL_MODELS} and at least two draws')
    if (null_model == 'region') != (windows is not None):
        raise ValueError('Region draws need windows; other nulls must not receive them')
    if bool(excluded) != (null_model == 'degree_no_scz'):
        raise ValueError('Only degree_no_scz excludes genes from random seed sampling')
    threshold = config.thresholds.individual
    gsz_idx = np.asarray(gsz_idx, dtype=int)
    names, reach_obs, own_obs = observed_masks(prop, seeds, gsz_idx, threshold)
    excluded_pos = np.array(sorted(prop.pos[int(g)] for g in excluded if int(g) in prop.pos), dtype=int)
    pools = [group[~np.isin(group, excluded_pos)] for group in prop.bins]
    signatures = {n: prop.signature(prop.seed_indices(seeds[n])) for n in names}
    for n in names:
        if any(k > len(pool) for pool, k in zip(pools, signatures[n])):
            raise ValueError(f'{n}: a degree bin has fewer eligible genes than seeds')
    identity = {'algorithm': 'scz_core_null_v1', 'code': file_hash(__file__),
                'propagation': prop.metadata, 'threshold': threshold, 'rng_seed': config.propagation.seed,
                'null_model': null_model, 'reps': reps, 'seeds': {n: sorted(map(int, seeds[n])) for n in names},
                'gsz': prop.nodes[gsz_idx].tolist(), 'excluded': sorted(map(int, excluded)),
                'windows': digest({n: list(map(int, w)) for n, w in windows.items()}) if windows else None}
    out = run_dir(output_root, identity, 'null', null_model, f'reps{reps}')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get('status') == 'complete':
            if any(not (out/f).exists() or file_hash(out/f) != h for f, h in old['artifacts'].items()):
                raise ValueError(f'Null artifacts changed in {out}')
            return out
    write_json(manifest, {'status': 'running', 'identity': identity})
    blocks = []
    for start in range(0, reps, BLOCK):
        stop = min(start + BLOCK, reps)
        path = out/f'draws-{start:06d}.npz'
        stamp = Path(str(path) + '.json')
        if path.exists() and stamp.exists() and json.loads(stamp.read_text())['sha256'] == file_hash(path):
            blocks.append(path)
            continue
        reach = np.zeros((stop - start, len(names), len(gsz_idx)), dtype=bool)
        own = np.zeros_like(reach)
        for r, rep in enumerate(range(start, stop)):
            # Same identity and per-CNV order as run_convergence: degree draws are identical.
            rng = rng_for(config.propagation.seed, 'joint_convergence', null_model, rep)
            for j, n in enumerate(names):
                if null_model == 'region':
                    genes = windows[n]
                    k = rng.integers(len(genes) - len(seeds[n]) + 1)
                    draw = genes[k:k + len(seeds[n])]
                else:
                    draw = prop.nodes[np.concatenate([rng.choice(pool, size=k, replace=False)
                        for pool, k in zip(pools, signatures[n]) if k])]
                cols = prop.seed_indices(draw)
                mu, sd, _ = prop.reference(cols)
                z = (np.log(prop.heat(draw)) - mu) / sd
                reach[r, j] = z[gsz_idx] > threshold
                own[r, j] = np.isin(gsz_idx, cols)
        with atomic_path(path) as tmp:
            np.savez_compressed(tmp, reach=_packed(reach), own=_packed(own))
        write_json(stamp, {'sha256': file_hash(path)})
        blocks.append(path)
        progress(f'{null_model} core null: {stop}/{reps} draws')
    reach, own = [], []
    for path in blocks:
        with np.load(path) as data:
            reach.append(data['reach'])
            own.append(data['own'])
    with atomic_path(out/'null_reach.npy') as tmp:
        np.save(tmp, np.concatenate(reach))
    with atomic_path(out/'null_own_seed.npy') as tmp:
        np.save(tmp, np.concatenate(own))
    with atomic_path(out/'observed.npz') as tmp:
        np.savez_compressed(tmp, reach=reach_obs, own=own_obs)
    write_json(out/'layout.json', {'cnvs': names, 'gsz_entrez': identity['gsz'], 'reps': reps,
        'arrays': 'null_*.npy: draw x CNV x packbits(G_SZ gene), unpack with count=len(gsz_entrez)'})
    for path in blocks:  # the concatenated arrays supersede the checkpoints
        Path(str(path) + '.json').unlink()
        path.unlink()
    write_json(manifest, {'status': 'complete', 'identity': identity,
        'artifacts': {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and p != manifest}})
    return out


def load_null(out):
    """Observed and null masks as booleans: null arrays are draw x CNV x G_SZ gene."""
    out = Path(out)
    layout = json.loads((out/'layout.json').read_text())
    n = len(layout['gsz_entrez'])
    with np.load(out/'observed.npz') as data:
        reach_obs, own_obs = data['reach'], data['own']
    return {'cnvs': layout['cnvs'], 'gsz': np.asarray(layout['gsz_entrez']),
            'reach_obs': reach_obs, 'own_obs': own_obs,
            'reach': _unpacked(np.load(out/'null_reach.npy'), n),
            'own': _unpacked(np.load(out/'null_own_seed.npy'), n)}
