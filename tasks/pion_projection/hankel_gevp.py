"""Hankel GEVP from the existing multi-sink data (todo C2, CPU).

For sink kernels K_i = H^{n_i} the cross-correlator is
C_ij(t) = <K_i S, K_j S> = <S, H^{n_i+n_j} S> = C_{0, n_i+n_j}(t),
so a GEVP matrix can be built from the measured channels whenever every sum
n_i + n_j is among the measured powers. With powers {0,2,4,8,16,32,64} the
2x2 set {0, 8} works; larger subsets would need new sums (C_24, ...).
"""
import json
import os
from pathlib import Path

import numpy as np

from .gevp import gevp_energies, summary
from .statistics import blocks, fold

RUN = Path(os.environ.get('PION_RUN', 'results/pion_projection/s16t16_v1'))


def build_hankel(correlators, powers, ops=(0, 8)):
    if correlators.ndim != 3:
        raise ValueError('expected (nblocks, nchannels, T)')
    index = {power: i for i, power in enumerate(powers)}
    missing = [a + b for a in ops for b in ops if a + b not in index]
    if missing:
        raise ValueError(f'cross channels for sums {sorted(set(missing))} were not measured')
    matrix = np.empty((len(correlators), len(ops), len(ops), correlators.shape[-1]))
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            matrix[:, i, j] = correlators[:, index[a + b]]
    return matrix


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    raw = np.load(RUN / 'development.npz')['correlators']
    data = blocks(fold(raw), p['block_length'])
    times = [3, 4, 5, 6]
    for ops in ((0, 8, 16), (0, 8)):
        try:
            matrix = build_hankel(data, p['powers'], ops=ops)
        except ValueError as exc:
            print(f'ops {ops}: skipped ({exc})')
            continue
        rows = gevp_energies(matrix, t0=2, times=times)
        print(f'ops {ops}: matrix size {len(ops)}')
        for state in range(len(ops)):
            result = summary(rows, state=state)
            print(f'  state {state}: E = {result["energy"]:.4f} +- {result["se_boot"]:.4f} '
                  f'(per time { {t: round(v, 4) for t, v in result["per_time"].items()} })')
    print('reference single-cosh 1.0862, two-state 1.0570')


if __name__ == '__main__':
    main()
