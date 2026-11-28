import hashlib
import json
from pathlib import Path
from uuid import uuid4


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
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    tmp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        tmp.write_text(payload)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
