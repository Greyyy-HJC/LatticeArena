"""Frozen identity of the reference model used to score a run.

The kind is resolved once, when a run is created, and stored in
submission.json. Evaluation must use that frozen kind (and the same
statistics/reference sources) or refuse to run; an environment variable may
only choose the model for a new run or an explicit read-only legacy replay.
"""
import os
from pathlib import Path

from .common import digest
from .statistics import ReferenceFit

SCHEMA = 'pion-reference-v1'
HERE = Path(__file__).parent
IMPLEMENTATION = {'single': 'statistics.py', 'twostate': 'reference_twostate.py'}


def reference_kind():
    kind = os.environ.get('LATTICEARENA_REFERENCE', 'single')
    if kind not in IMPLEMENTATION:
        raise ValueError(f'Unknown reference kind {kind!r}')
    return kind


def reference_spec(kind=None):
    kind = kind or reference_kind()
    if kind not in IMPLEMENTATION:
        raise ValueError(f'Unknown reference kind {kind!r}')
    return {'schema': SCHEMA, 'kind': kind,
            'statistics_sha256': digest(HERE / 'statistics.py'),
            'implementation_sha256': digest(HERE / IMPLEMENTATION[kind])}


def make_reference(data, p, window, kind=None):
    kind = kind or reference_kind()
    if kind == 'twostate':
        from .reference_twostate import TwoStateFit
        return TwoStateFit(data, p, window)
    if kind == 'single':
        return ReferenceFit(data, p, window)
    raise ValueError(f'Unknown reference kind {kind!r}')


def verify_reference(spec):
    if not spec or spec.get('schema') != SCHEMA:
        raise ValueError('Reference spec missing or unsupported; legacy submissions are read-only')
    kind = spec.get('kind')
    if kind not in IMPLEMENTATION:
        raise ValueError(f'Unknown frozen reference kind {kind!r}')
    if spec.get('statistics_sha256') != digest(HERE / 'statistics.py'):
        raise ValueError('statistics.py changed since the reference was frozen')
    if spec.get('implementation_sha256') != digest(HERE / IMPLEMENTATION[kind]):
        raise ValueError(f'{IMPLEMENTATION[kind]} changed since the reference was frozen')
    return kind
