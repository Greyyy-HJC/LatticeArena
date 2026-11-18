"""CPU teacher experiment for P3 step 1: spectral class vs a non-commuting class.

Independent links/probes, a teacher T = I + 0.2 H + eps Q with Q = (VH+HV)/2.
The spectral class fits c_k with sum_k c_k H^k; the augmented class adds Q.
Errors are reported on held-out probes from a held-out link configuration.
"""
import json
from pathlib import Path

import numpy as np

from tasks.pion_projection.kernels import build_kernel

OUT = Path('results/optimize_gaussian/p3_teacher_v1')
POWERS = [0, 2, 4, 8, 16, 32, 64]
EPS = [0.0, 0.02, 0.05, 0.1]


def random_links(rng, shape):
    a = rng.normal(size=(3, *shape, 3, 3)) + 1j * rng.normal(size=(3, *shape, 3, 3))
    q, _ = np.linalg.qr(a)
    return q / np.linalg.det(q)[..., None, None] ** (1 / 3)


def operator_matrix(links, shape, rng):
    n = int(np.prod(shape))
    kernel = build_kernel(links, {'powers': [1], 'weights': [1.0], 'alpha': 0.2})
    H = np.zeros((3 * n, 3 * n), dtype=complex)
    for site in range(n):
        for colour in range(3):
            probe = np.zeros((*shape, 3, 1), dtype=complex)
            probe.reshape(n, 3)[site, colour] = 1.0
            H[:, 3 * site + colour] = kernel.apply(probe).reshape(3 * n)
    return H


def features(H, probes, extra=None):
    blocks = [(np.linalg.matrix_power(H, k) @ probes).ravel() for k in POWERS]
    if extra is not None:
        blocks.append((extra @ probes).ravel())
    return np.stack(blocks, axis=1)


def run(shape=(3, 3, 3), seed=11, n_probe=48):
    rng = np.random.default_rng(seed)
    dim = 3 * int(np.prod(shape))
    links = random_links(rng, shape)
    H_train = H_test = operator_matrix(links, shape, rng)
    V_train = np.diag(np.repeat(rng.uniform(0.5, 1.5, size=int(np.prod(shape))), 3))
    Q_train = Q_test = (V_train @ H_train + H_train @ V_train) / 2
    train_probes = rng.normal(size=(dim, n_probe)) + 1j * rng.normal(size=(dim, n_probe))
    test_probes = rng.normal(size=(dim, n_probe)) + 1j * rng.normal(size=(dim, n_probe))
    rows = []
    for eps in EPS:
        teacher_train = (np.eye(dim) + 0.3 * np.linalg.matrix_power(H_train, 2)
                         + 0.1 * np.linalg.matrix_power(H_train, 4) + eps * Q_train)
        teacher_test = teacher_train
        y_train = (teacher_train @ train_probes).ravel()
        y_test = (teacher_test @ test_probes).ravel()
        spectral_train = features(H_train, train_probes)
        spectral_test = features(H_test, test_probes)
        coeff, *_ = np.linalg.lstsq(spectral_train, y_train, rcond=None)
        spectral_err = float(np.linalg.norm(spectral_test @ coeff - y_test) / np.linalg.norm(y_test))
        augmented_train = features(H_train, train_probes, extra=Q_train)
        augmented_test = features(H_test, test_probes, extra=Q_test)
        coeff_a, *_ = np.linalg.lstsq(augmented_train, y_train, rcond=None)
        augmented_err = float(np.linalg.norm(augmented_test @ coeff_a - y_test) / np.linalg.norm(y_test))
        rank = int(np.linalg.matrix_rank(spectral_train))
        rows.append({'eps': eps, 'spectral_test_error': spectral_err,
                     'augmented_test_error': augmented_err,
                     'spectral_effective_rank': rank,
                     'commutator': float(np.linalg.norm(H_train @ Q_train - Q_train @ H_train)),
                     'symmetry_H': float(np.max(np.abs(H_train - H_train.conj().T))),
                     'symmetry_Q': float(np.max(np.abs(Q_train - Q_train.conj().T)))})
    return {'shape': list(shape), 'n_probe': n_probe, 'rows': rows,
            'status': 'PASS' if rows[0]['spectral_test_error'] < 1e-8 and
            rows[-1]['spectral_test_error'] > 1e-3 and
            rows[-1]['augmented_test_error'] < 1e-8 else 'INCONCLUSIVE',
            'scope': 'synthetic_CPU; GPU 2x2 class/optimizer experiment NOT_RUN'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = run()
    (OUT / 'p3_teacher.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in result['rows']:
        print(row)
    print('status', result['status'])


if __name__ == '__main__':
    main()
