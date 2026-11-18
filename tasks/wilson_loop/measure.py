"""Measure Wilson loops W(R,T) from smeared spatial lines and original temporal links."""
import os
import subprocess
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import digest, fingerprint, indices, protocol, read_nersc_with_status, write_json
from core.ledger import Ledger

from .identity import check_resume, generation, loops_sha256, measurement_spec, write_marker
from .kernels import _lib, build_kernel, dagger, shift


def loops(links, temporal, p):
    """links/temporal: (t, z, y, x, 3, 3); returns W with shape (len(R), len(T))."""
    R_list, T_list = p['r_values'], p['t_values']
    xp = _lib(links)
    out = xp.zeros((len(R_list), len(T_list)))
    volume = float(np.prod(temporal.shape[:4]))
    for d in range(3):
        axis = 3 - d
        K, cur = {1: links[d].copy()}, links[d].copy()
        for R in range(2, max(R_list) + 1):
            cur = shift(cur, 1, axis)
            K[R] = K[R - 1] @ cur
        P, cur = {1: temporal.copy()}, temporal.copy()
        for T in range(2, max(T_list) + 1):
            cur = shift(cur, 1, 0)
            P[T] = P[T - 1] @ cur
        for i, R in enumerate(R_list):
            for j, T in enumerate(T_list):
                w = K[R] @ shift(P[T], R, axis) @ dagger(shift(K[R], T, 0)) @ dagger(P[T])
                out[i, j] += xp.trace(w, axis1=-2, axis2=-1).real.sum()
    out = out / (9 * volume)
    return out.get() if hasattr(out, 'get') else out


def git_state():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True,
                                             stderr=subprocess.DEVNULL).strip())
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': 'unavailable', 'dirty': None}


def measure(output, split, limit=None):
    p = protocol()
    try:
        import cupy as xp
    except ImportError:
        xp = np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = indices(split)[:limit] if limit else indices(split)
    spec = measurement_spec(p, split)
    if limit:
        spec = dict(spec, ids=spec['ids'][:limit])
    completed = check_resume(output, split, spec)
    if completed:
        print(f'resuming {split} from row {completed}/{len(ids)}', flush=True)
    pending = ids[completed:]
    rows, records = [], []
    if completed:
        with np.load(output / f'{split}.npz', allow_pickle=False) as f:
            rows = f['loops'].tolist()
        records = json.loads((output / f'{split}_provenance.json').read_text())['records'][:completed]
    kernel_sha = digest(Path(__file__).with_name('kernels.py'))
    ledger = Ledger(device='cupy' if xp is not np else 'cpu', precision='complex128', parallel=1)
    for cfg in pending:
        gauge_path = Path(p['ensemble']) / f'wilson_b6.{cfg}'
        cfg_start = perf_counter()
        u, checksum_status = read_nersc_with_status(gauge_path, expected_geometry=p['geometry'])
        u = xp.asarray(u)
        setup = perf_counter() - cfg_start
        measured = perf_counter()
        try:
            row = [loops(build_kernel(u[:3], k), u[3], p) for k in p['kernels']]
        except ValueError as exc:
            raise ValueError(f'cfg {cfg}: {exc}') from exc
        rows.append(row)
        ledger.add('setup', setup).add('kernel', perf_counter() - measured)
        records.append({'id': int(cfg), 'gauge_path': str(gauge_path), 'checksum_status': checksum_status,
                        'gauge_sha256': digest(gauge_path), 'loops_sha256': loops_sha256(np.asarray(row)),
                        'setup_seconds': setup, 'measure_seconds': perf_counter() - measured,
                        'total_seconds': perf_counter() - cfg_start})
        saved = completed + len(rows)
        if saved != len(records) or saved != len(spec['ids'][:saved]):
            raise RuntimeError(f'row accounting mismatch: {saved=} {len(rows)=} {len(records)=}')
        tmp = output / f'.{split}.tmp.npz'
        np.savez_compressed(tmp, ids=np.asarray(spec['ids'][:saved]),
                            loops=np.asarray(rows), protocol_hash=fingerprint(p),
                            generation=generation(spec))
        os.replace(tmp, output / f'{split}.npz')
        write_json(output / f'{split}_provenance.json',
                   {'protocol': p, 'kernel_sha256': kernel_sha, 'ensemble': p['ensemble'],
                    'git': git_state(), 'generation': generation(spec),
                    'dependencies': {name: digest(Path(__file__).with_name(name))
                                     for name in ('kernels.py', 'statistics.py', 'measure.py',
                                                  'common.py', 'identity.py')},
                    'records': records, 'ledger': ledger.summary()})
        write_marker(output, split, output / f'{split}.npz', output / f'{split}_provenance.json',
                     spec, np.asarray(rows))
        print(f'cfg {cfg}: {records[-1]["total_seconds"]:.2f}s', flush=True)
    return records
