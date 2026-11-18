"""Compare the sink-sink Gram matrix with the standard source-sink correlator.

Writes results/pion_projection/gram_identity_check.json (todo P0.1/P0.2).
"""
import json
from pathlib import Path

import numpy as np

MS = Path('results/pion_projection/s16t16_ms/development.npz')
GRAM = Path('results/pion_projection/s16t16_gram3/development.npz')
OUT = Path('results/pion_projection/gram_identity_check.json')


def main():
    g = np.load(GRAM)
    names = [str(x) for x in g['names'][:3]]
    gram = g['gram'][:, :3, :3, :].mean(axis=0)          # sink-sink M_ij
    m = np.load(MS)
    powers = [int(x) for x in m['powers']]
    index = [powers.index(0), powers.index(4), powers.index(16)]
    source_sink = m['correlators'][:, :, index, :].mean(axis=0)  # C[source j, sink i]
    rows = []
    for t in (3, 4, 5, 6):
        M = gram[:, :, t]
        C = source_sink[:, :, t].T                        # rows i (sink), cols j (source)
        scale = float(np.abs(C).max())
        rows.append({
            't': t,
            'max_abs_gram_minus_standard_over_max': float(np.abs(M - C).max() / scale),
            'gram_symmetry': float(np.abs(M - M.T).max() / np.abs(M).max()),
            'standard_antihermitian': float(np.abs(C - C.conj().T).max() / np.abs(C).max()),
        })
    out = {'schema': 'pion-gram-identity-v1', 'operators': names, 'temporal': 16,
           'rows': rows,
           'note': 'M_ij = <K_i S, K_j S> (both kernels on the sink) versus the standard '
                   'source-sink C_ij = <O_i O_j^dagger>; they coincide only for a '
                   'translation-invariant commuting family up to boundary effects'}
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
