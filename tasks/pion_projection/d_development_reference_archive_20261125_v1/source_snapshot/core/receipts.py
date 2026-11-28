"""Execution receipts for RunState (todo F3).

A receipt binds a state to a kind, the code hashes and input hashes actually
used, and the observed result. A free-form string or a modified receipt cannot
make a state physical.
"""
import json

from core.io import digest_text


def make(kind, code, inputs, result):
    payload = {'kind': kind, 'code': dict(code), 'inputs': dict(inputs),
               'result': result}
    return {**payload, 'sha256': digest_text(json.dumps(payload, sort_keys=True))}


def verify(receipt):
    if not isinstance(receipt, dict):
        raise ValueError('receipt is not a mapping')
    for key in ('kind', 'code', 'inputs', 'result', 'sha256'):
        if key not in receipt:
            raise ValueError(f'receipt is missing {key!r}')
    if not receipt['code'] or not receipt['inputs']:
        raise ValueError('receipt has empty code or inputs')
    payload = {'kind': receipt['kind'], 'code': dict(receipt['code']),
               'inputs': dict(receipt['inputs']), 'result': receipt['result']}
    if receipt['sha256'] != digest_text(json.dumps(payload, sort_keys=True)):
        raise ValueError('receipt hash does not match its content')
    return True
