import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / 'protocol.json'
DEFAULT_OUTPUT = Path('results/pion_projection/s16t16_v1')


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def indices(split):
    first, last = protocol()[split]
    return np.arange(first, last + 1)


def load_data(output, split):
    p = protocol()
    path = Path(output) / f'{split}.npz'
    with np.load(path, allow_pickle=False) as f:
        if not np.array_equal(f['ids'], indices(split)):
            raise ValueError(f'{split}: missing or reordered configurations')
        if str(f['protocol_hash']) != fingerprint(p):
            raise ValueError('Protocol changed since measurement')
        data = f['correlators'].copy()
    if data.shape != (len(indices(split)), len(p['powers']), p['geometry'][3]):
        raise ValueError('Unexpected correlator shape')
    if not np.all(np.isfinite(data)):
        raise ValueError('Nonfinite measurement')
    return data
