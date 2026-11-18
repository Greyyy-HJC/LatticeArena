"""GEVP from the multi-source measurement (C2, CPU).

Matrix M[a,b](t) = C[source a, sink H^{n_b}](t), a,b over the matched
point/H^4/H^16 operators. Up to a transpose this is the GEVP matrix
(transposition does not change the generalized eigenvalues). The matrix is
ill-conditioned; results depend on the eigenvalue floor, so treat any "plateau"
with care.
"""
import json
import os
from pathlib import Path

import numpy as np

from .gevp import gevp_energies, summary

RUN = Path(os.environ.get('PION_MS_RUN', 'results/pion_projection/s16t16_ms'))
SINKS = (0, 4, 16)


def build_matrix(correlators, powers, sinks=SINKS):
    if correlators.ndim != 4:
        raise ValueError('expected (ncfg, n_sources, n_powers, T)')
    if len(sinks) != correlators.shape[1]:
        raise ValueError('number of sinks must match the number of sources')
    index = [list(powers).index(s) for s in sinks]
    matrix = correlators[:, :, index, :]
    return 0.5 * (matrix + matrix.conj().swapaxes(1, 2))


def regularize(matrix, rel=1e-6):
    """Floor the eigenvalues of each time slice so C(t0) is positive definite."""
    n, d, _, temporal = matrix.shape
    flat = matrix.transpose(0, 3, 1, 2).reshape(-1, d, d)
    values, vectors = np.linalg.eigh(flat)
    values = np.maximum(values, rel * values[:, -1:])
    fixed = (vectors * values[:, None, :]) @ vectors.conj().swapaxes(-1, -2)
    return fixed.reshape(n, temporal, d, d).transpose(0, 2, 3, 1)


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    times = [3, 4, 5, 6]
    for split in ('development', 'validation'):
        path = RUN / f'{split}.npz'
        if not path.exists():
            continue
        z = np.load(path)
        matrix = regularize(build_matrix(z['correlators'], z['powers']))
        rows = gevp_energies(matrix, t0=2, times=times)
        print(split, 'matrix', matrix.shape[1:3])
        for state in range(matrix.shape[1]):
            result = summary(rows, state=state)
            print(f'  state {state}: E = {result["energy"]:.4f} +- {result["se_boot"]:.4f} '
                  f'per-time { {t: round(v, 4) for t, v in result["per_time"].items()} }')


if __name__ == '__main__':
    main()
