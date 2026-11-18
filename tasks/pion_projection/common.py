import json
import os
from pathlib import Path

import numpy as np

from core.io import digest, fingerprint, write_json  # noqa: F401  (re-export)

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = Path(os.environ.get('LATTICEARENA_PROTOCOL', str(HERE / 'protocol.json')))
DEFAULT_OUTPUT = Path('results/pion_projection/s16t16_v1')


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def indices(split, p=None):
    p = p or protocol()
    first, last = p[split]
    return np.arange(first, last + 1)


def load_data(output, split, p=None):
    p = p or protocol()
    path = Path(output) / f'{split}.npz'
    with np.load(path, allow_pickle=False) as f:
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
