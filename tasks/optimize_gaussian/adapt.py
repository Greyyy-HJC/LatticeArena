"""CPU end-to-end adaptation of the common framework (todo package D).

Exercises OperatorContract, MeasurementBundle and SearchAdapter with the
Gaussian task's own objects: a fixed-W response array, a learned kernel
K = B C(a(theta)) B and a small theta search. Synthetic data only.
"""
import json
from pathlib import Path

import numpy as np

from core.io import digest_text
from core.measurement import bundle, describe, require
from core.operator_checks import hermiticity_residual, linearity_residual
from core.search import Budget, search

OUT = Path('results/optimize_gaussian/cpu_next_package_d')
T, L, CHANNELS = 9, 4, 8


def response_array(rng):
    """Synthetic fixed-W response: (channels, T) with a decaying spectrum."""
    t = np.arange(T)
    base = np.exp(-0.5 * t) + np.exp(-0.5 * (T - t))
    rng = np.random.default_rng(0) if rng is None else rng
    return np.stack([base * (1 + 0.05 * rng.normal(size=t.size)) for _ in range(CHANNELS)])


def kernel(theta, field):
    """Toy learned kernel: local mass term plus a Hermitian hop with theta scaling."""
    mass = 1.0 + 0.1 * float(theta[0])
    hop = 0.5 * (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0))
    return mass * field + 0.2 * float(theta[1]) * hop


def contract_checks(rng):
    field = rng.normal(size=(L, 3)) + 1j * rng.normal(size=(L, 3))
    other = rng.normal(size=(L, 3)) + 1j * rng.normal(size=(L, 3))
    theta = np.array([0.3, -0.2])
    op = lambda f: kernel(theta, f)
    linear = linearity_residual(op, field, other)
    bilinear = lambda f, g: np.vdot(f, op(g))          # <f, K g>
    herm = hermiticity_residual(bilinear, field, other)
    identity = np.linalg.norm(kernel(np.zeros(2), field) - field) / np.linalg.norm(field)
    return {'linearity_residual': linear, 'hermiticity_residual': herm,
            'identity_limit_theta0': float(identity),
            'contract': {'linear_in_field': bool(linear < 1e-12), 'hermitian': bool(herm < 1e-12),
                         'identity_at_theta0': bool(identity < 1e-12),
                         'geometry': [L, 3], 'cost': 'O(volume * channels)'}}


def search_theta(rng):
    target = response_array(rng)
    def evaluate(theta):
        pred = response_array(np.random.default_rng(0)) + 0.1 * float(theta[0]) - 0.05 * float(theta[1])
        return float(np.linalg.norm(pred - target))
    proposals = [('a', [0.0, 0.0]), ('b', [0.5, 0.0]), ('c', [0.0, 0.5]), ('d', [float('nan'), 0.0])]
    result = search(proposals, evaluate, validate=lambda u: (len(u) == 2 and np.all(np.isfinite(u)), 'bad shape/value'),
                    budget=Budget(4))
    return {'accepted': len(result.accepted), 'rejected': len(result.rejected),
            'evaluations': result.evaluations,
            'best': result.accepted[0]['risk'] if result.accepted else None,
            'reasons': [item['reason'] for item in result.rejected]}


def main():
    rng = np.random.default_rng(20261124)
    response = response_array(rng)
    b = bundle('optimize_gaussian', {'response': response},
               {'protocol_hash': digest_text(json.dumps({'T': T, 'channels': CHANNELS})),
                'splits': {'development': {'n': CHANNELS}}, 'code': {'adapt.py': __file__}},
               {'linear_response': True, 'cross_matrix': False, 'new_source': False})
    fields = contract_checks(rng)
    search_info = search_theta(rng)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'optgauss-adapt-v1', 'bundle': describe(b),
              'operator_contract': fields, 'search': search_info,
              'capability_refusal': None}
    try:
        require(b, 'cross_matrix')
    except ValueError as exc:
        result['capability_refusal'] = str(exc)
    (OUT / 'adapt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2)[:1200])
    print('wrote', OUT)


if __name__ == '__main__':
    main()
