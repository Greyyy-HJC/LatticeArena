"""Generate pure-gauge SU(3) configurations with PyQUDA's HMC (DCU/QUDA backend).

Writes NERSC-format files named wilson_b6.<id> so that LatticeArena's
tasks/pion_projection/measure.py can read them unchanged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from pyquda.action import GaugeAction, abstract
from pyquda.hmc import HMC, Integrator
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T


class BAB(Integrator):
    """BAB (2nd order) leapfrog, as in PyQUDA examples/1_Quenched_HMC.py."""

    def integrate(self, updateGauge, updateMom, t):
        dt = t / self.n_steps
        for _ in range(self.n_steps):
            updateMom(dt / 2)
            updateGauge(dt)
            updateMom(dt / 2)


def plaq(hmc):
    return float(hmc.plaquette()[0])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', default='16,16,16,16')
    ap.add_argument('--beta', type=float, default=6.0)
    ap.add_argument('--trajectories', type=int, default=800)
    ap.add_argument('--thermal', type=int, default=300)
    ap.add_argument('--save-every', type=int, default=5)
    ap.add_argument('--max-save', type=int, default=100)
    ap.add_argument('--start-id', type=int, default=0)
    ap.add_argument('--n-steps', type=int, default=120)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--start-config', default=None)
    ap.add_argument('--seed', type=int, default=20261109)
    ap.add_argument('--out', default='data/S16T16')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    if args.smoke:
        args.trajectories, args.thermal, args.save_every, args.max_save = 3, 0, 1, 1

    latt = [int(x) for x in args.lattice.split(',')]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path('.cache/quda').mkdir(parents=True, exist_ok=True)

    core.init([1, 1, 1, 1], latt, resource_path='.cache/quda')
    info = core.LatticeInfo(latt)
    loop = abstract.LoopParam(
        [[X, Y, -X, -Y], [X, Z, -X, -Z], [Y, Z, -Y, -Z],
         [X, T, -X, -T], [Y, T, -Y, -T], [Z, T, -Z, -T]],
        [1.0] * 6,
    )
    hmc = HMC(info, [GaugeAction(info, loop, args.beta)], BAB(args.n_steps))
    gauge = core.LatticeGauge(info)
    if args.start_config:
        gauge = io.readNERSCGauge(args.start_config)
    else:
        rng = np.random.default_rng(args.seed)
        shape = gauge.data.shape
        a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        q, _ = np.linalg.qr(a)
        q = q / np.linalg.det(q)[..., None, None] ** (1 / 3)
        gauge.data[:] = cp.asarray(q.astype(np.complex128))
    hmc.initialize(args.seed, gauge)
    print(f'INIT lattice={latt} beta={args.beta} n_steps={args.n_steps} '
          f'seed={args.seed} tau={args.tau} plaquette={plaq(hmc):.8f}', flush=True)

    manifest_path = out / 'manifest.json'
    manifest = {'lattice': latt, 'beta': args.beta, 'n_steps': args.n_steps,
                'seed': args.seed, 'ensemble': 'LatticeArena-pilot',
                'configs': []}
    accepted = saved = 0
    t_start = time.perf_counter()
    for traj in range(args.trajectories):
        t0 = time.perf_counter()
        hmc.gaussMom()
        e_old = hmc.momAction() + hmc.gaugeAction()
        hmc.integrate(args.tau, 2e-14)
        e_new = hmc.momAction() + hmc.gaugeAction()
        acc = hmc.accept(e_new - e_old)
        if acc or traj < args.thermal:
            hmc.saveGauge(gauge)
        else:
            hmc.loadGauge(gauge)
        accepted += int(acc)
        dt = time.perf_counter() - t0
        if traj % 20 == 0 or traj == args.trajectories - 1:
            print(f'TRAJ {traj + 1}/{args.trajectories} acc={acc} '
                  f'plaquette={plaq(hmc):.8f} dE={e_new - e_old:.4e} '
                  f'sec={dt:.3f} acc_rate={accepted / (traj + 1):.3f}', flush=True)
        if traj >= args.thermal and (traj - args.thermal) % args.save_every == 0:
            cfg_id = args.start_id + saved
            path = out / f'wilson_b6.{cfg_id}'
            io.writeNERSCGauge(str(path), gauge, ensemble_id='LatticeArena-pilot',
                               ensemble_label=f'beta{args.beta}', sequence_number=cfg_id)
            plaq_val = plaq(hmc)
            manifest['configs'].append({'id': cfg_id, 'trajectory': traj + 1,
                                        'plaquette': plaq_val, 'sha256': sha256(path)})
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
            saved += 1
            print(f'SAVED {path} id={cfg_id} traj={traj + 1} plaquette={plaq_val:.8f}', flush=True)
            if saved >= args.max_save:
                break
    total = time.perf_counter() - t_start
    print(f'DONE saved={saved} trajectories={args.trajectories} '
          f'accept_rate={accepted / args.trajectories:.3f} '
          f'wall_seconds={total:.1f} sec_per_traj={total / args.trajectories:.3f}', flush=True)


if __name__ == '__main__':
    main()
