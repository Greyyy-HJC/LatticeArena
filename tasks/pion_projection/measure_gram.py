"""Gram-matrix measurement for a GEVP with two operator families (C2, GPU).

For each configuration, time slice and kernel pair,
    M_ij(t) = <K_i S, K_j S>
is measured from one point-source propagator. The kernel families are H^n and
H^n applied after APE smoothing of the spatial links, which are genuinely
different operators (not just powers of H).
"""
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import digest, fingerprint, indices, protocol, write_json
from ..wilson_loop.kernels import ape


def kernel_specs(p):
    """Three genuinely different operator families: H, APE-H and a stronger H."""
    specs = []
    for n in (0, 4, 16):
        specs.append({'name': f'H{n}', 'power': n, 'ape': None, 'alpha': 0.2})
    for n in (0, 4, 16):
        specs.append({'name': f'APE0.5x3_H{n}', 'power': n, 'ape': (0.5, 3), 'alpha': 0.2})
    for n in (0, 4, 16):
        specs.append({'name': f'H{n}_a0.4', 'power': n, 'ape': None, 'alpha': 0.4})
    return specs


def measure_gram(output, split, limit=None):
    import cupy as cp
    from mpi4py import MPI
    from pyquda_utils import core, io

    from .kernels import build_kernel

    p = protocol()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    core.init([1, 1, 1, 1], p['geometry'], resource_path='.cache/quda')
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError('Pilot supports one MPI rank')
    latt = core.LatticeInfo(p['geometry'], p['t_boundary'], 1.0)
    dirac = core.getWilson(latt, 1/(2*p['kappa'])-4, p['tolerance'], p['max_iterations'])
    specs = kernel_specs(p)
    ids = indices(split)[:limit] if limit else indices(split)
    rows, records = [], []
    for cfg in ids:
        start = perf_counter()
        gauge = io.readNERSCGauge(f"{p['ensemble']}/wilson_b6.{cfg}")
        links_all = cp.asarray(gauge.lexico())[:3]
        with dirac.useGauge(gauge):
            prop = core.invert(dirac, 'point', p['source'])
        fields = cp.asarray(prop.lexico())
        gram = np.zeros((len(specs), len(specs), p['geometry'][3]))
        for t in range(p['geometry'][3]):
            field = fields[t].transpose(0, 1, 2, 5, 3, 4, 6).reshape(*p['geometry'][:3], 3, 48)
            lk = links_all[:, t]
            smeared = []
            for spec in specs:
                use = lk if spec['ape'] is None else ape(lk[:, None], *spec['ape'])[:, 0]
                kernel = build_kernel(use, {'powers': [spec['power']], 'weights': [1.0], 'alpha': spec['alpha']})
                smeared.append(kernel.apply(field))
            for i in range(len(specs)):
                for j in range(len(specs)):
                    gram[i, j, t] = cp.vdot(smeared[i], smeared[j]).real.get()
        rows.append(gram)
        records.append({'id': int(cfg), 'gauge_sha256': digest(f"{p['ensemble']}/wilson_b6.{cfg}"),
                        'seconds': perf_counter()-start})
        np.savez_compressed(output / f'{split}.npz', ids=np.asarray(ids[:len(rows)]),
                            gram=np.asarray(rows), names=np.asarray([s['name'] for s in specs]),
                            protocol_hash=fingerprint(p))
        write_json(output / f'{split}_provenance.json', {'protocol': p, 'records': records, 'specs': specs})
        print(f'gram cfg {cfg}: {records[-1]["seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='development')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    measure_gram(args.output, args.split, args.limit)
