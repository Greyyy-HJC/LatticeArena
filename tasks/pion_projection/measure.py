"""Trusted one-rank PyQUDA measurement and checked point-propagator cache."""
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import digest, fingerprint, indices, protocol, write_json


def measure(output, split, limit=None):
    import cupy as cp
    from mpi4py import MPI
    from pyquda_utils import core, io, source
    from pyquda.enum_quda import QudaVerbosity

    p = protocol()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if split == 'validation':
        frozen = json.loads((output / 'submission.json').read_text())
        if frozen['protocol_hash'] != fingerprint(p):
            raise ValueError('Submission protocol mismatch')
        if frozen['development_sha256'] != digest(output / 'development.npz'):
            raise ValueError('Development measurements changed after submission freeze')
    cache = Path('.cache/pion_projection/propagators')
    cache.mkdir(parents=True, exist_ok=True)
    Path('.cache/quda').mkdir(parents=True, exist_ok=True)
    core.init([1, 1, 1, 1], p['geometry'], resource_path='.cache/quda')
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError('Pilot supports one MPI rank')
    cp.get_default_memory_pool().set_limit(size=p['gpu_memory_limit_gib'] * 1024**3)
    software = {name: version(name) for name in ['PyQUDA', 'PyQUDA-Utils', 'cupy-cuda12x', 'numpy', 'mpi4py']}
    software['libquda_sha256'] = digest('/home/jinchen/software/quda/build/lib/libquda.so')
    software['mpi'] = MPI.Get_library_version().strip('\x00\n')
    software['measurement_sha256'] = digest(__file__)
    software['kernel_sha256'] = digest(Path(__file__).with_name('kernels.py'))
    latt = core.LatticeInfo(p['geometry'], p['t_boundary'], 1.0)
    dirac = core.getWilson(latt, 1/(2*p['kappa'])-4, p['tolerance'], p['max_iterations'])
    dirac.setVerbosity(QudaVerbosity.QUDA_SILENT)
    ids = indices(split)
    if limit is not None:
        ids = ids[:limit]
    correlators, records = [], []
    for cfg_id in ids:
        begin = perf_counter()
        filename = Path(p['ensemble']) / f'wilson_b6.{cfg_id}'
        cfg_hash = digest(filename)
        keydata = {'gauge': cfg_hash, 'geometry': p['geometry'], 'kappa': p['kappa'],
                   'boundary': p['t_boundary'], 'source': p['source'], 'tol': p['tolerance'],
                   'maxiter': p['max_iterations'], 'software': software}
        key = fingerprint(keydata)
        cachefile = cache / f'{key}.npy'
        metafile = cache / f'{key}.json'
        gauge = io.readNERSCGauge(str(filename))
        if gauge.latt_info.global_size != p['geometry']:
            raise ValueError('Wrong geometry')
        plaq = float(gauge.plaquette()[0])
        cache_hit = cachefile.exists() and metafile.exists()
        with dirac.useGauge(gauge):
            solve_start = perf_counter()
            if cache_hit:
                metadata = json.loads(metafile.read_text())
                if metadata['keydata'] != keydata or digest(cachefile) != metadata['array_sha256']:
                    raise ValueError('Cache provenance/checksum mismatch')
                prop = core.LatticePropagator(latt, cp.asarray(np.load(cachefile, allow_pickle=False)))
            else:
                prop = core.invert(dirac, 'point', p['source'])
            residuals = []
            for spin in range(4):
                for color in range(3):
                    b = source.source(latt, 'point', p['source'], spin, color)
                    r = dirac.mat(prop.getFermion(spin, color)) - b
                    residuals.append(float((r.norm2()/b.norm2())**0.5))
            if not np.all(np.isfinite(residuals)) or max(residuals) > p['tolerance']:
                raise ValueError(f'Failed residual check for {filename}: {residuals}')
            solve_seconds = perf_counter() - solve_start
        if not cache_hit:
            tmp = cachefile.with_suffix('.tmp.npy')
            np.save(tmp, prop.getHost(), allow_pickle=False)
            tmp.replace(cachefile)
            write_json(metafile, {'keydata': keydata, 'array_sha256': digest(cachefile)})
        sink_start = perf_counter()
        c = np.zeros((len(p['powers']), p['geometry'][3]))
        # H is Hermitian: <S,H^(2n)S> = ||H^n S||^2. Incremental QUDA
        # smearing is an exact fast path for these even diffusion powers.
        with gauge.use() as gdirac:
            for spin in range(4):
                for color in range(3):
                    field = prop.getFermion(spin, color)
                    previous = 0
                    for j, power in enumerate(p['powers']):
                        n = power // 2
                        if power % 2:
                            raise ValueError('QUDA fast path requires even powers')
                        if n > previous:
                            field = gdirac.wuppertalSmear(field, n-previous, p['alpha'])
                        density = cp.sum(cp.abs(field.data)**2, axis=(-2, -1))
                        c[j] += density.sum(axis=(0, 2, 3, 4)).get()
                        previous = n
        direct = cp.sum(cp.abs(prop.data)**2, axis=(-4, -3, -2, -1)).sum(axis=(0, 2, 3, 4)).get()
        np.testing.assert_allclose(c[0], direct, rtol=1e-12, atol=0)
        if not np.all(np.isfinite(c)) or np.any(c <= 0):
            raise ValueError('Invalid basis correlator')
        # Check actual time-slice submission API against QUDA on the first
        # configuration of each run, independently of the norm fast path.
        if int(cfg_id) == int(ids[0]):
            from .kernels import build_kernel
            links = gauge.lexico(force_numpy=False)[:3, 2]
            # (z,y,x,sink_spin,source_spin,sink_color,source_color)
            lexico = prop.lexico(force_numpy=False)[2]
            field = lexico.transpose(0, 1, 2, 5, 3, 4, 6).reshape(16, 16, 16, 3, 48)
            kernel = build_kernel(links, {'powers': [2], 'weights': [1.0], 'alpha': p['alpha']})
            reference = cp.vdot(field, kernel.apply(field)).get()
            np.testing.assert_allclose(reference, c[1, 2], rtol=1e-10, atol=1e-13)
        cp.cuda.Stream.null.synchronize()
        elapsed = perf_counter() - begin
        if elapsed > p['wall_seconds_per_configuration']:
            raise RuntimeError('Configuration wall-time budget exceeded')
        record = {'id': int(cfg_id), 'gauge_sha256': cfg_hash, 'plaquette': plaq,
                  'residual_max': max(residuals), 'cache_key': key, 'cache_hit': cache_hit,
                  'solve_and_residual_seconds': solve_seconds, 'sink_seconds': perf_counter()-sink_start,
                  'total_seconds': elapsed, 'cupy_pool_peak_reserved_bytes': cp.get_default_memory_pool().total_bytes()}
        records.append(record)
        correlators.append(c)
        print(json.dumps(record), flush=True)
        # Checkpoint completed rows, but consumers reject incomplete splits.
        np.savez_compressed(output / f'{split}.npz', ids=ids[:len(records)],
                            correlators=np.asarray(correlators), protocol_hash=fingerprint(p))
        write_json(output / f'{split}_provenance.json', {'protocol': p, 'software': software, 'records': records})
        del prop, gauge
    return records
