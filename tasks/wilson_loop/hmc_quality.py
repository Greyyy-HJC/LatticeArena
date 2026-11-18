"""HMC production-quality audit (todo F4): manifest, autocorrelation, toy integrator.

The toy SU(3) plaquette action on a 2^4 lattice checks the leapfrog algorithm
(reversibility and energy-error order) with numerical gradients; it is not a
re-run of the production PyQUDA HMC.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

OUT = Path('results/wilson_loop/hmc_quality_v1')
MANIFEST = OUT / 'manifest_snapshot.json'

GM = []
for a, b in [(0, 1), (0, 2), (1, 2)]:
    m = np.zeros((3, 3), dtype=complex)
    m[a, b] = m[b, a] = 1
    GM.append(m / np.sqrt(2))
for a, b in [(0, 1), (0, 2), (1, 2)]:
    m = np.zeros((3, 3), dtype=complex)
    m[a, b] = -1j
    m[b, a] = 1j
    GM.append(m / np.sqrt(2))
GM.append(np.diag([1, 1, -2]) / np.sqrt(6))
GM.append(np.diag([1, -1, 0]) / np.sqrt(2))


def random_su3(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    return q / np.linalg.det(q) ** (1 / 3)


def action(u, beta=6.0):
    total = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            p = (u[mu] @ np.roll(u[nu], -1, axis=mu)
                 @ np.roll(u[mu], -1, axis=nu).conj().transpose(0, 1, 2, 3, 5, 4)
                 @ u[nu].conj().transpose(0, 1, 2, 3, 5, 4))
            total += np.real(np.trace(p, axis1=-2, axis2=-1)).sum() / 3
    return beta * (6 * u.shape[1] ** 3 - total)


def force(u, beta=6.0, delta=1e-5):
    f = np.zeros_like(u, dtype=complex)
    for mu in range(4):
        for a in range(8):
            for x in range(u.shape[1]):
                for y in range(u.shape[2]):
                    for z in range(u.shape[3]):
                        for t in range(u.shape[4]):
                            plus, minus = u.copy(), u.copy()
                            plus[mu, x, y, z, t] = expm(1j * delta * GM[a]) @ u[mu, x, y, z, t]
                            minus[mu, x, y, z, t] = expm(-1j * delta * GM[a]) @ u[mu, x, y, z, t]
                            f[mu, x, y, z, t] += -(action(plus, beta) - action(minus, beta)) / (2 * delta) * GM[a]
    return f


def drift(u, p, eps):
    out = u.copy()
    for mu in range(4):
        for x in range(u.shape[1]):
            for y in range(u.shape[2]):
                for z in range(u.shape[3]):
                    for t in range(u.shape[4]):
                        out[mu, x, y, z, t] = expm(1j * eps * p[mu, x, y, z, t]) @ u[mu, x, y, z, t]
    return out


def leapfrog(u, p, force_fn, eps, steps):
    # force_fn returns F = -dS/dU, so the canonical kick is p += eps * F.
    p = p + 0.5 * eps * force_fn(u)
    for i in range(steps):
        u = drift(u, p, eps)
        p = p + (eps * force_fn(u) if i < steps - 1 else 0.5 * eps * force_fn(u))
    return u, p


def toy_integrator(seed=5):
    rng = np.random.default_rng(seed)
    u = np.array([[[[[random_su3(rng) for _ in range(2)] for _ in range(2)]
                    for _ in range(2)] for _ in range(2)] for _ in range(4)])
    u = u.reshape(4, 2, 2, 2, 2, 3, 3)
    base = np.zeros((4, 2, 2, 2, 2, 3, 3), dtype=complex)
    base[...] = np.eye(3)
    p = np.einsum('...ij,...jk->...ik', np.array([[[[[rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
                                                    for _ in range(2)] for _ in range(2)]
                                                   for _ in range(2)] for _ in range(2)] for _ in range(4)]).reshape(4, 2, 2, 2, 2, 3, 3),
                  base)
    p = 0.05 * (p + p.conj().transpose(0, 1, 2, 3, 4, 6, 5))
    p = p - np.trace(p, axis1=-2, axis2=-1)[..., None, None] / 3 * np.eye(3)
    rows = []
    for eps in (0.1, 0.05, 0.025, 0.0125):
        steps = max(int(round(0.5 / eps)), 1)
        start = action(u)
        u1, p1 = leapfrog(u.copy(), p.copy(), force, eps, steps)
        end = action(u1)
        dH = abs((end + 0.5 * float(np.vdot(p1, p1).real)) - (start + 0.5 * float(np.vdot(p, p).real)))
        u2, p2 = leapfrog(u1, -p1, force, eps, steps)
        revers = float(np.max(np.abs(u2 - u)))
        rows.append({'eps': eps, 'steps': steps, 'delta_H': dH, 'reversibility_error': revers})
    return rows


def autocorrelation(values):
    x = np.asarray(values, dtype=float) - np.mean(values)
    n = len(x)
    acf = [float(np.dot(x[:n - k], x[k:]) / np.dot(x, x)) for k in range(1, n // 2)]
    tau = 0.5
    for k, value in enumerate(acf, start=1):
        tau += value
        if k >= 5 * tau:
            break
    return acf, float(tau)


def manifest_audit():
    data = json.loads(MANIFEST.read_text())
    configs = data['configs']
    trajectories = [c['trajectory'] for c in configs]
    spacing = sorted({b - a for a, b in zip(trajectories, trajectories[1:])})
    acf, tau = autocorrelation([c['plaquette'] for c in configs])
    return {'lattice': data['lattice'], 'beta': data['beta'], 'n_steps': data['n_steps'],
            'seed': data['seed'], 'n_configs': len(configs),
            'first_trajectory': trajectories[0], 'save_spacing': spacing,
            'missing': [name for name in ('delta_H', 'acceptance', 'thermalization_declared',
                                          'tau_int_declared') if name not in data],
            'tau_int_plaquette': tau, 'plaquette_mean': float(np.mean([c['plaquette'] for c in configs])),
            'plaquette_std': float(np.std([c['plaquette'] for c in configs], ddof=1)),
            'autocorrelation': acf[:10]}


def order_check(eps_list=(0.02, 0.01), total=0.1, seed=7):
    """Small, fast second-order check: Delta H ratio for two step sizes."""
    rows = []
    for eps in eps_list:
        rng = np.random.default_rng(seed)
        u = np.array([[[[[random_su3(rng) for _ in range(2)] for _ in range(2)] for _ in range(2)]
                        for _ in range(2)] for _ in range(4)]).reshape(4, 2, 2, 2, 2, 3, 3)
        r = rng.normal(size=u.shape) + 1j * rng.normal(size=u.shape)
        p = 0.05 * (r + r.conj().transpose(0, 1, 2, 3, 4, 6, 5))
        p = p - np.trace(p, axis1=-2, axis2=-1)[..., None, None] / 3 * np.eye(3)
        steps = max(int(round(total / eps)), 1)
        start = action(u) + 0.5 * float(np.vdot(p, p).real)
        u1, p1 = leapfrog(u, p, force, eps, steps)
        rows.append({'eps': eps, 'steps': steps,
                     'delta_H': abs(action(u1) + 0.5 * float(np.vdot(p1, p1).real) - start)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = manifest_audit()
    toy = toy_integrator()
    (OUT / 'hmc_quality.json').write_text(json.dumps({'manifest': audit, 'toy_integrator': toy}, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plaq = [c['plaquette'] for c in json.loads(MANIFEST.read_text())['configs']]
    axes[0].plot(plaq, lw=0.8)
    axes[0].set(xlabel='saved configuration', ylabel='plaquette', title='plaquette history (S16T16C)')
    axes[0].grid(alpha=0.2)
    eps = [r['eps'] for r in toy]
    axes[1].loglog(eps, [r['delta_H'] for r in toy], 'o-', label='|ΔH|')
    axes[1].loglog(eps, [r['reversibility_error'] for r in toy], 's--', label='reversibility error')
    axes[1].set(xlabel='leapfrog step ε', ylabel='error', title='toy SU(3) integrator (2^4)')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'hmc_quality.png', dpi=150)
    print(json.dumps({'manifest': {k: v for k, v in audit.items() if k != 'autocorrelation'},
                      'toy': toy}, indent=2))
    print('wrote', OUT)


if __name__ == '__main__':
    main()


MANIFEST_FIELDS = ('seed', 'first_trajectory', 'spacing', 'tau_int', 'n_configs',
                   'acceptance_rate', 'delta_h', 'thermalization')


def audit_manifest(manifest):
    """Classify an HMC manifest by which production fields are actually present."""
    reasons = []
    for field in MANIFEST_FIELDS:
        if manifest.get(field) is None:
            reasons.append(f'{field}_missing')
    fields = {field: manifest.get(field) for field in MANIFEST_FIELDS}
    return {'status': 'verified' if not reasons else 'archived_diagnostic',
            'reasons': reasons, 'fields': fields,
            'note': ('unknown fields are reported as missing, not filled with local defaults')}
