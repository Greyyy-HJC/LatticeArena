"""Shared evidence schema (todo F0).

Every check records what was run, on which inputs and code, what was observed,
the threshold, the status and the scope. The schema deliberately allows PASS,
FAIL, NOT_RUN and INCONCLUSIVE: a missing check is not silently a pass.
"""
import json
from pathlib import Path

REQUIRED = ('test_id', 'task', 'claim', 'command', 'seed', 'geometry', 'dtype',
            'code', 'inputs', 'observed', 'threshold', 'status', 'scope')
STATUS = ('PASS', 'FAIL', 'NOT_RUN', 'INCONCLUSIVE')
SCOPE = ('synthetic_CPU', 'archived_data', 'live_GPU')


def record(test_id, task, claim, command, observed, threshold, status, scope,
           seed=None, geometry=None, dtype='complex128', code=None, inputs=None):
    entry = {'test_id': test_id, 'task': task, 'claim': claim, 'command': command,
             'seed': seed, 'geometry': geometry, 'dtype': dtype,
             'code': code or {}, 'inputs': inputs or {},
             'observed': observed, 'threshold': threshold,
             'status': status, 'scope': scope}
    validate(entry)
    return entry


def validate(entry):
    missing = [key for key in REQUIRED if key not in entry]
    if missing:
        raise ValueError(f'evidence entry is missing {missing}')
    if entry['status'] not in STATUS:
        raise ValueError(f'unsupported status {entry["status"]!r}')
    if entry['scope'] not in SCOPE:
        raise ValueError(f'unsupported scope {entry["scope"]!r}')
    return entry


def write_evidence(path, entries):
    for entry in entries:
        validate(entry)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema': 'latticearena-evidence-v1', 'entries': entries}, indent=2) + '\n')
    return path


def load_evidence(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 'latticearena-evidence-v1':
        raise ValueError(f'{path}: unsupported evidence schema')
    for entry in data['entries']:
        validate(entry)
    return data['entries']


def summarise(entries):
    rows = []
    for entry in entries:
        rows.append({'task': entry['task'], 'test_id': entry['test_id'], 'claim': entry['claim'],
                     'status': entry['status'], 'scope': entry['scope'],
                     'observed': entry['observed'], 'threshold': entry['threshold']})
    return rows
