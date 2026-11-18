"""Teacher-operator check of the seven-power basis (todo P2, CPU slice).

Builds H from the real diffusion kernel, a non-commuting Hermitian Q = (VH+HV)/2,
and a teacher I + 0.2 H + 0.1 Q. Compares the least-squares fit with the seven
powers against the augmented basis {powers, H, Q}. This is a teacher-level
representability test, not a trained model.
"""
import json
from pathlib import Path

import numpy as np

from tasks.pion_projection.kernels import build_kernel

OUT = Path('results/optimize_gaussian/teacher_v1')
POWERS = [0, 2, 4, 8, 16, 32, 64]


def random_links(rng, shape=(4, 4, 4)):
    a = rng.normal(size=(3, *shape, 3, 3)) + 1j * rng.normal(size=(3, *shape, 3, 3))
    q, _ = np.linalg.qr(a)
    q = q / np.linalg.det(q)[..., None, None] ** (1 / 3)
    return q


def teacher_result(shape=(4, 4, 4), seed=7):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    links = random_links(rng, shape)
    kernel = build_kernel(links, {'powers': [1], 'weights': [1.0], 'alpha': 0.2})
    dim = 3 * n
    H = np.zeros((dim, dim), dtype=complex)
    for site in range(n):
        for colour in range(3):
            probe = np.zeros((*shape, 3, 1), dtype=complex)
            probe.reshape(n, 3)[site, colour] = 1.0
            H[:, 3 * site + colour] = kernel.apply(probe).reshape(n * 3)
    weights = rng.uniform(0.5, 1.5, size=n)
    V = np.diag(np.repeat(weights, 3))
    Q = (V @ H + H @ V) / 2
    teacher = np.eye(dim) + 0.2 * H + 0.1 * Q
    _, eigenvectors = np.linalg.eigh(0.5 * (H + H.conj().T))
    q_eig = eigenvectors.conj().T @ Q @ eigenvectors
    spectral_floor = 0.1 * np.linalg.norm(q_eig - np.diag(np.diag(q_eig))) / np.linalg.norm(teacher)

    def fit_error(columns):
        matrix = np.stack(columns, axis=1)
        coeff, *_ = np.linalg.lstsq(matrix, teacher.ravel(), rcond=None)
        return float(np.linalg.norm(matrix @ coeff - teacher.ravel()) / np.linalg.norm(teacher.ravel()))

    powers = [np.linalg.matrix_power(H, k) for k in POWERS]
    seven = fit_error([m.ravel() for m in powers])
    augmented = fit_error([m.ravel() for m in powers] + [H.ravel(), Q.ravel()])
    result = {'geometry': list(shape), 'n': n, 'powers': POWERS,
              'hermiticity_H': float(np.max(np.abs(H - H.conj().T))),
              'hermiticity_Q': float(np.max(np.abs(Q - Q.conj().T))),
              'commutator': float(np.linalg.norm(H @ Q - Q @ H)),
              'spectral_floor': float(spectral_floor),
              'seven_power_error': seven, 'augmented_error': augmented,
              'status': 'PASS' if seven > 1e-3 and augmented < 1e-10 else 'INCONCLUSIVE',
              'scope': 'synthetic_CPU, teacher operator only; trained class experiment NOT_RUN (GPU)'}
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = teacher_result()
    (OUT / 'teacher.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
