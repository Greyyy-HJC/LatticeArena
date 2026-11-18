"""Measure C_{source,sink}(t) for several Gaussian source smearings (C2, GPU).

For a Hermitian sink kernel K_i = H^{n_i} the correlator with source j is
    C_{j,i}(t) = <S_j, H^{n_i} S_j> = || H^{n_i/2} S_j ||^2,
so one inversion per source is enough. Writes a new array; does not touch the
frozen single-source measurement.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import digest, fingerprint, indices, protocol, write_json


def measure_multisource(output, split, limit=None):
    import cupy as cp
    from mpi4py import MPI
    from pyquda_utils import core, io, source

    p = protocol()
    sources = p['source_smear']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    core.init([1, 1, 1, 1], p['geometry'], resource_path='.cache/quda')
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError('Pilot supports one MPI rank')
    latt = core.LatticeInfo(p['geometry'], p['t_boundary'], 1.0)
    dirac = core.getWilson(latt, 1/(2*p['kappa'])-4, p['tolerance'], p['max_iterations'])
    ids = indices(split)[:limit] if limit else indices(split)
    rows, records = [], []
    for cfg in ids:
        start = perf_counter()
        gauge = io.readNERSCGauge(f"{p['ensemble']}/wilson_b6.{cfg}")
        if list(gauge.latt_info.global_size) != list(p['geometry']):
            raise ValueError('Wrong geometry')
        per_source = []
        with dirac.useGauge(gauge):
            for spec in sources:
                src = source.source12(latt, 'point', p['source'])
                if spec['steps']:
                    src = source.gaussian12(src, gauge, spec['rho'], spec['steps'])
                prop = core.invertPropagator(dirac, src, 12)
                residual = 0.0
                for spin in range(4):
                    for color in range(3):
                        b = src.getFermion(spin, color)
                        r = dirac.mat(prop.getFermion(spin, color)) - b
                        residual = max(residual, float((r.norm2()/b.norm2())**0.5))
                limit_tol = max(1e-8, 10 * p['tolerance'])
                if residual > limit_tol:
                    raise ValueError(f'Residual {residual} above {limit_tol} for cfg {cfg}')
                c = np.zeros((len(p['powers']), p['geometry'][3]))
                with gauge.use() as gdirac:
                    for spin in range(4):
                        for color in range(3):
                            field = prop.getFermion(spin, color)
                            previous = 0
                            for j, power in enumerate(p['powers']):
                                n = power // 2
                                if n > previous:
                                    field = gdirac.wuppertalSmear(field, n-previous, p['alpha'])
                                density = cp.sum(cp.abs(field.data)**2, axis=(-2, -1))
                                c[j] += density.sum(axis=(0, 2, 3, 4)).get()
                                previous = n
                per_source.append(c)
        rows.append(per_source)
        records.append({'id': int(cfg), 'gauge_sha256': digest(f"{p['ensemble']}/wilson_b6.{cfg}"),
                        'seconds': perf_counter()-start})
        np.savez_compressed(output / f'{split}.npz', ids=np.asarray(ids[:len(rows)]),
                            correlators=np.asarray(rows), powers=np.asarray(p['powers']),
                            sources=np.asarray([s['name'] for s in sources]),
                            protocol_hash=fingerprint(p))
        write_json(output / f'{split}_provenance.json',
                   {'protocol': p, 'records': records, 'source_smear': sources})
        print(f'multisource cfg {cfg}: {records[-1]["seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='development')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    measure_multisource(args.output, args.split, args.limit)
