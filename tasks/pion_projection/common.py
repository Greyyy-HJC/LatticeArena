import json
from io import BytesIO
from pathlib import Path

import numpy as np

from core.io import digest, fingerprint, write_json  # noqa: F401  (re-export)
from core.npz_safety import validate_npz_payload
from core.source_integrity import SourceByteSnapshot

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / 'protocol.json'
DEFAULT_OUTPUT = Path('results/pion_projection/s16t16_v1')
_MAX_COMPRESSED_NPZ_BYTES = 64 * 1024 * 1024


def _read_bounded_npz(path):
    """Capture one path-based NPZ without exceeding the snapshot byte budget."""
    chunks = []
    total = 0
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(min(4 * 1024 * 1024,
                                    _MAX_COMPRESSED_NPZ_BYTES - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_COMPRESSED_NPZ_BYTES:
                raise ValueError('NPZ compressed size limit exceeded')
            chunks.append(chunk)
    return b''.join(chunks)


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def indices(split, p=None):
    p = p or protocol()
    first, last = p[split]
    return np.arange(first, last + 1)


def load_data(output, split, p=None, *, source_snapshot=None, source_name=None):
    p = p or protocol()
    path = Path(output) / f'{split}.npz'
    if source_snapshot is not None:
        if not isinstance(source_snapshot, SourceByteSnapshot):
            raise ValueError('source_snapshot must be a verified SourceByteSnapshot')
        if split != 'development':
            raise ValueError(
                'snapshot data loading is restricted to the open development split '
                'in this workflow'
            )
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError('source_name is required when loading from a source snapshot')
        payload = source_snapshot.read_bytes(source_name)
        validate_npz_payload(payload)
        stream = BytesIO(payload)
    else:
        if source_name is not None:
            raise ValueError('source_name is only valid together with source_snapshot')
        payload = _read_bounded_npz(path)
        validate_npz_payload(payload)
        stream = BytesIO(payload)
    with np.load(stream, allow_pickle=False) as f:
        if not np.array_equal(f['ids'], indices(split, p)):
            raise ValueError(f'{split}: missing or reordered configurations')
        if str(f['protocol_hash']) != fingerprint(p):
            raise ValueError('Protocol changed since measurement')
        data = f['correlators'].copy()
    if data.shape != (len(indices(split, p)), len(p['powers']), p['geometry'][3]):
        raise ValueError('Unexpected correlator shape')
    if not np.all(np.isfinite(data)):
        raise ValueError('Nonfinite measurement')
    return data
