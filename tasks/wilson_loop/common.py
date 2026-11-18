import json
import os
from pathlib import Path

import numpy as np

from core.io import digest, fingerprint, write_json  # noqa: F401  (re-export)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path('results/wilson_loop/s16t16_v1')
PROTOCOL_PATH = Path(os.environ.get('WILSON_PROTOCOL', str(HERE / 'protocol.json')))


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def indices(split):
    first, last = protocol()[split]
    return np.arange(first, last + 1)


def _nersc_payload(path, expected_geometry=None):
    """Parse a NERSC gauge file; return (raw bytes, checksum status, dims).

    A trailing 16-byte MD5 (NERSC checksum trailer) is verified when present;
    an absent trailer is reported as 'absent' rather than treated as verified.
    """
    import hashlib
    with open(path, 'rb') as f:
        f.readline()
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'{path}: missing END_HEADER')
            if line.strip() == b'END_HEADER':
                break
            key, value = line.decode().split('=', 1)
            header[key.strip()] = value.strip()
        payload = f.read()
    try:
        nx, ny, nz, nt = (int(header[f'DIMENSION_{i}']) for i in (1, 2, 3, 4))
    except (KeyError, ValueError) as exc:
        raise ValueError(f'{path}: incomplete NERSC dimension header') from exc
    datatype = header.get('DATATYPE')
    if datatype is not None and not datatype.startswith('4D_SU3_GAUGE_3x3'):
        raise ValueError(f'{path}: unsupported DATATYPE {datatype!r}')
    floating = header.get('FLOATING_POINT')
    if floating is not None:
        value = floating.strip().upper()
        if value in ('IEEE64BIG', 'IEEE32LITTLE', 'IEEE32BIG'):
            raise ValueError(f'{path}: unsupported FLOATING_POINT {floating!r}')
        if value not in ('1', 'TRUE', 'YES', 'IEEE64LITTLE'):
            raise ValueError(f'{path}: unsupported FLOATING_POINT {floating!r}')
    if expected_geometry is not None and [nx, ny, nz, nt] != list(expected_geometry):
        raise ValueError(f'{path}: header geometry {[nx, ny, nz, nt]} != run geometry {list(expected_geometry)}')
    expected = nt * nz * ny * nx * 4 * 3 * 3 * 16
    if len(payload) == expected:
        status, raw = 'absent', payload
    elif len(payload) == expected + 16:
        raw, trailer = payload[:expected], payload[expected:]
        if hashlib.md5(raw).digest() != trailer:
            raise ValueError(f'{path}: NERSC trailing MD5 mismatch')
        status = 'verified'
    else:
        raise ValueError(f'{path}: payload has {len(payload)} bytes, expected {expected}'
                         f' (or {expected + 16} with a checksum trailer)')
    return raw, status, (nx, ny, nz, nt)


def read_nersc(path, expected_geometry=None):
    """Read a NERSC 4D_SU3_GAUGE_3x3 file into (mu, t, z, y, x, 3, 3).

    Validates the header keys we rely on, the exact payload length and the
    optional MD5 trailer; unknown optional keys are tolerated, but a wrong data
    type or truncated payload is rejected instead of being reshaped silently.
    """
    raw, _, (nx, ny, nz, nt) = _nersc_payload(path, expected_geometry)
    return np.frombuffer(raw, dtype='<c16').reshape(nt, nz, ny, nx, 4, 3, 3).transpose(4, 0, 1, 2, 3, 5, 6)


def read_nersc_with_status(path, expected_geometry=None):
    """Read the array and report the checksum trailer status in one parse."""
    raw, status, (nx, ny, nz, nt) = _nersc_payload(path, expected_geometry)
    array = np.frombuffer(raw, dtype='<c16').reshape(nt, nz, ny, nx, 4, 3, 3).transpose(4, 0, 1, 2, 3, 5, 6)
    return array, status


def nersc_checksum_status(path, expected_geometry=None):
    """'verified' (trailer matched) or 'absent'; a mismatch raises."""
    _, status, _ = _nersc_payload(path, expected_geometry)
    return status


def indices_for(p, split):
    first, last = p[split]
    return np.arange(first, last + 1)


def load_data(output, split, p=None):
    p = p or protocol()
    with np.load(Path(output) / f'{split}.npz', allow_pickle=False) as f:
        if not np.array_equal(f['ids'], indices_for(p, split)):
            raise ValueError(f'{split}: missing or reordered configurations')
        if str(f['protocol_hash']) != fingerprint(p):
            raise ValueError('Protocol changed since measurement')
        loops = f['loops'].copy()
    expected = (len(indices_for(p, split)), len(p['kernels']), len(p['r_values']), len(p['t_values']))
    if loops.shape != expected:
        raise ValueError(f'Unexpected loop shape {loops.shape}, expected {expected}')
    if loops.dtype.kind not in 'fiu' or not np.all(np.isfinite(loops)):
        raise ValueError('Loops must be finite real numbers')
    return loops
