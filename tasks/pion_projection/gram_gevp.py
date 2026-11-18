"""GEVP on the Gram matrix of two operator families (C2, CPU).

gram[config, i, j, t] = <K_i S, K_j S> with K_i from H^n and APE-smoothed H^n.
A tiny eigenvalue floor removes round-off negatives before the generalized
eigenproblem.
"""
import os
from pathlib import Path

import numpy as np

from .gevp import gevp_energies, summary

RUN = Path(os.environ.get('PION_GRAM_RUN', 'results/pion_projection/s16t16_gram'))


def floored(gram, rel=1e-10):
    n, d, _, temporal = gram.shape
    flat = gram.transpose(0, 3, 1, 2).reshape(-1, d, d)
    values, vectors = np.linalg.eigh(0.5 * (flat + flat.conj().swapaxes(-1, -2)))
    values = np.maximum(values, rel * values[:, -1:])
    fixed = (vectors * values[:, None, :]) @ vectors.conj().swapaxes(-1, -2)
    return fixed.reshape(n, temporal, d, d).transpose(0, 2, 3, 1)


def main():
    times = [3, 4, 5, 6]
    for split in ('development', 'validation'):
        path = RUN / f'{split}.npz'
        if not path.exists():
            continue
        z = np.load(path)
        matrix = floored(z['gram'])
        rows = gevp_energies(matrix, t0=2, times=times)
        print(split, 'matrix', matrix.shape[1:3])
        for state in range(min(3, matrix.shape[1])):
            result = summary(rows, state=state)
            print(f'  state {state}: E = {result["energy"]:.4f} +- {result["se_boot"]:.4f} '
                  f'per-time { {t: round(v, 4) for t, v in result["per_time"].items()} }')


if __name__ == '__main__':
    main()
