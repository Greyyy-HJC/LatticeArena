"""Measurement identity, generation ids and recovery transactions (P1).

A measurement is a three-file transaction: {split}.npz, {split}_provenance.json
and {split}.commit.json. Resume is allowed only when all three agree on the
generation and the recorded content hashes, otherwise the entry is refused.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from core.io import digest, fingerprint

DEPENDENCIES = ('kernels.py', 'statistics.py', 'measure.py', 'common.py', 'identity.py')


def measurement_spec(p, split):
    from .common import indices
    here = Path(__file__).parent
    return {'split': split, 'ids': [int(x) for x in indices(split)],
            'protocol_hash': fingerprint(p),
            'kernel_sha256': digest(here / 'kernels.py'),
            'dependencies': {name: digest(here / name) for name in DEPENDENCIES}}


def generation(spec):
    return fingerprint(spec)


def loops_sha256(loops):
    return hashlib.sha256(np.ascontiguousarray(loops).tobytes()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_resume(output, split, spec, verify_gauge=True):
    """Return the number of completed rows that may be resumed, or refuse."""
    root = Path(output)
    npz_path = root / f'{split}.npz'
    provenance_path = root / f'{split}_provenance.json'
    marker_path = root / f'{split}.commit.json'
    existing = [path.exists() for path in (npz_path, provenance_path, marker_path)]
    if not any(existing):
        return 0
    if not all(existing):
        raise ValueError('partial measurement transaction (npz/provenance/commit must all exist); '
                         'refusing to resume or overwrite')
    marker = json.loads(marker_path.read_text())
    wanted = generation(spec)
    if marker.get('generation') != wanted:
        raise ValueError(f'{marker_path}: generation does not match the requested spec')
    if marker.get('npz_sha256') != _file_sha(npz_path):
        raise ValueError(f'{npz_path}: file hash differs from the commit marker')
    if marker.get('provenance_sha256') != _file_sha(provenance_path):
        raise ValueError(f'{provenance_path}: file hash differs from the commit marker')
    provenance = json.loads(provenance_path.read_text())
    if provenance.get('generation') != wanted:
        raise ValueError(f'{provenance_path}: generation does not match the requested spec')
    if provenance.get('kernel_sha256') != spec['kernel_sha256']:
        raise ValueError(f'{provenance_path}: kernel hash differs from the spec')
    with np.load(npz_path, allow_pickle=False) as f:
        if 'generation' not in f:
            raise ValueError(f'{npz_path}: legacy measurement without a generation id')
        if str(f['generation']) != wanted:
            raise ValueError(f'{npz_path}: generation does not match the requested spec')
        ids = f['ids'].tolist()
        loops = f['loops']
    if loops.ndim != 4 or not np.all(np.isfinite(loops)):
        raise ValueError(f'{npz_path}: loops are not a finite 4-D array')
    if ids != spec['ids'][:len(ids)]:
        raise ValueError(f'{npz_path}: stored ids are not a prefix of the requested split')
    records = provenance.get('records', [])
    if len(records) != len(ids):
        raise ValueError(f'{provenance_path}: {len(records)} records versus {len(ids)} ids')
    if [int(r['id']) for r in records] != ids:
        raise ValueError(f'{provenance_path}: record ids do not match the stored ids in order')
    if loops_sha256(loops) != marker.get('loops_sha256'):
        raise ValueError(f'{npz_path}: loops content differs from the commit marker')
    for i, record in enumerate(records):
        if record.get('loops_sha256') is None:
            raise ValueError(f'{provenance_path}: record {i} has no loops hash (legacy entry)')
        if record['loops_sha256'] != loops_sha256(loops[i:i + 1]):
            raise ValueError(f'{provenance_path}: record {i} loops hash does not match the npz')
        if verify_gauge:
            gauge = Path(record.get('gauge_path', ''))
            if not gauge.exists() or digest(gauge) != record.get('gauge_sha256'):
                raise ValueError(f'{provenance_path}: record {i} gauge file/hash mismatch')
    return len(ids)


def write_marker(root, split, npz_path, provenance_path, spec, loops):
    marker = {'generation': generation(spec), 'split': split,
              'npz_sha256': _file_sha(npz_path), 'provenance_sha256': _file_sha(provenance_path),
              'loops_sha256': loops_sha256(loops)}
    (Path(root) / f'{split}.commit.json').write_text(json.dumps(marker, indent=2) + '\n')
    return marker


def trust_status(run, p, submission, provenance):
    """Derive a trust label from identities, not from the protocol name."""
    reasons = []
    try:
        from .common import load_data
        load_data(run, 'validation')
    except (ValueError, FileNotFoundError) as exc:
        reasons.append(f'data validation failed: {exc}')
    here = Path(__file__).parent
    kernel_sha = digest(here / 'kernels.py')
    if submission.get('kernel_sha256') != kernel_sha:
        reasons.append('submission kernel hash does not match the current kernels.py')
    if provenance.get('kernel_sha256') != submission.get('kernel_sha256'):
        reasons.append('measurement provenance kernel hash differs from the submission')
    if provenance.get('protocol', {}).get('version') != submission.get('protocol_version'):
        reasons.append('protocol version differs between provenance and submission')
    if provenance.get('generation') is None:
        reasons.append('measurement has no generation id')
    if not submission.get('identity'):
        reasons.append('no identity block in the submission')
    for name in DEPENDENCIES:
        recorded = provenance.get('dependencies', {}).get(name)
        if recorded is not None and recorded != digest(here / name):
            reasons.append(f'dependency {name} changed since measurement')
    states = {'operator': 'unknown', 'reference': 'unknown', 'numerical': 'unknown',
              'statistical': 'unknown'}
    return {'status': 'physical' if not reasons else 'archived_diagnostic', 'reasons': reasons,
            'states': states}
