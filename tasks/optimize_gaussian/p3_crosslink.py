"""Cross-link teacher experiment (todo CPU package C).

Independent train/test gauge links and one deterministic gauge-conditioned
operator rule. The target is Q[U] = H(U)^2 + eps * B(U) H(U) B(U), where H is the
summed traceless plaquette and B a single-plane plaquette. Feature class A is
H^0..H^7 (contains the eps=0 target exactly); class B adds nested-space blocks
G0, G0 H, H G0 (contains the eps>0 term). Train on train links, freeze, predict
on independent test links; SLSQP/Powell plus a least-squares oracle.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

OUT = Path('results/optimize_gaussian/cpu_next_p3_crosslink')
SHAPE = (4, 4, 4, 4)
EPSILON_SCAN = (0.0, 0.01, 0.05, 0.1)
POWERS = range(0, 8)
SEEDS = (0, 1, 2)


def su3(rng, eps):
    a = rng.normal(size=(*SHAPE, 4, 3, 3))
    g = (a + a.transpose(0, 1, 2, 3, 4, 6, 5).conj()) / 2.0
    g = g - np.trace(g, axis1=-2, axis2=-1)[..., None, None] / 3.0 * np.eye(3)
    u = np.eye(3) + 1j * eps * g
    q, _ = np.linalg.qr(u)
    det = np.linalg.det(q)
    return q / (det ** (1 / 3))[..., None, None]


def _plane(links, x, mu, nu):
    xpm = list(x); xpm[mu] = (x[mu] + 1) % SHAPE[mu]
    xpn = list(x); xpn[nu] = (x[nu] + 1) % SHAPE[nu]
    p = (links[tuple(x)][mu] @ links[tuple(xpm)][nu]
         @ links[tuple(xpn)][mu].conj().T @ links[tuple(x)][nu].conj().T)
    p = (p + p.conj().T) / 2
    return p - np.trace(p) / 3 * np.eye(3)


def H_of(links, x):
    return sum((_plane(links, x, mu, nu) for mu in range(4) for nu in range(mu + 1, 4)),
               np.zeros((3, 3), dtype=complex))


def B_of(links, x):
    return _plane(links, x, 0, 1)


def target(H, B, eps):
    g0 = B @ H @ B
    q = H @ H + eps * (g0 + g0.conj().T) / 2
    return q - np.trace(q) / 3 * np.eye(3)


def samples(rng, links, n, eps, augmented):
    rows, targets = [], []
    for _ in range(n):
        x = tuple(int(rng.integers(0, SHAPE[i])) for i in range(4))
        h, b = H_of(links, x), B_of(links, x)
        g0 = (b @ h @ b)
        feats = [np.linalg.matrix_power(h, p) for p in POWERS]
        if augmented:
            feats += [g0, g0 @ h, h @ g0]
        rows.append(np.concatenate([f.reshape(-1) for f in feats]))
        targets.append(target(h, b, eps).reshape(-1))
    return np.asarray(rows), np.asarray(targets)


def fit(X, Y, method):
    xsd = X.std(0)
    keep = xsd > 1e-8
    xmu, xsd = X.mean(0)[keep], xsd[keep]
    ysd = Y.std(0)
    ykeep = ysd > 1e-8
    ymu, ysd = Y.mean(0)[ykeep], ysd[ykeep]
    Xtr = (X[:, keep] - xmu) / xsd
    Ytr = (Y[:, ykeep] - ymu) / ysd
    if method == 'oracle':
        theta, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
        return theta, 1, True, (keep, xmu, xsd, ykeep, ymu, ysd)
    def objective(flat):
        theta = flat.reshape(Xtr.shape[1], Ytr.shape[1])
        r = (Xtr @ theta - Ytr).ravel()
        return float(np.real(np.vdot(r, r)))
    n = Xtr.shape[1] * Ytr.shape[1]
    result = minimize(objective, np.zeros(n), method=method, bounds=[(-5.0, 5.0)] * n,
                      options={'maxiter': 300})
    theta = result.x.reshape(Xtr.shape[1], Ytr.shape[1])
    return theta, int(result.nfev), bool(result.success), (keep, xmu, xsd, ykeep, ymu, ysd)


def predict(X, theta, meta):
    keep, xmu, xsd, ykeep, ymu, ysd = meta
    return ((X[:, keep] - xmu) / xsd @ theta) * ysd + ymu


def run(seed, eps, n_train=200, n_test=100):
    rng = np.random.default_rng(seed)
    train_links = su3(rng, eps) @ su3(rng, 0.5)
    test_links = su3(rng, eps) @ su3(rng, 0.5)
    out = {}
    for augmented in (False, True):
        X_train, Y_train = samples(rng, train_links, n_train, eps, augmented)
        X_test, Y_test = samples(rng, test_links, n_test, eps, augmented)
        klass = 'B_nested' if augmented else 'A_spectral'
        for method in ('SLSQP', 'Powell', 'oracle'):
            theta, nfev, success, meta = fit(X_train, Y_train, method)
            error = float(np.linalg.norm(predict(X_test, theta, meta) - Y_test[:, meta[3]])
                          / (np.linalg.norm(Y_test[:, meta[3]]) + 1e-12))
            out[f'{klass}/{method}'] = {'test_error': error, 'nfev': nfev, 'success': success,
                                        'train_error': float(np.linalg.norm(predict(X_train, theta, meta)
                                                                            - Y_train[:, meta[3]])
                                                             / (np.linalg.norm(Y_train[:, meta[3]]) + 1e-12))}
    return out


def attributes():
    rng = np.random.default_rng(1)
    links = su3(rng, 0.1) @ su3(rng, 0.5)
    h, b = H_of(links, (1, 2, 1, 2)), B_of(links, (1, 2, 1, 2))
    q = target(h, b, 0.1)
    return {'q_hermitian': float(np.linalg.norm(q - q.conj().T)),
            'q_traceless': float(abs(np.trace(q))),
            'h_hermitian': float(np.linalg.norm(h - h.conj().T)),
            'b_hermitian': float(np.linalg.norm(b - b.conj().T))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'p3-crosslink-v2', 'shape': SHAPE, 'attributes': attributes(),
              'seeds': list(SEEDS), 'epsilon_scan': list(EPSILON_SCAN), 'rows': {}}
    for eps in EPSILON_SCAN:
        for seed in SEEDS:
            result['rows'][f'eps{eps}/seed{seed}'] = run(seed, eps)
            print(f'eps{eps} seed{seed}', {k: round(v['test_error'], 6)
                                           for k, v in result['rows'][f'eps{eps}/seed{seed}'].items()})
    (OUT / 'crosslink.json').write_text(json.dumps(result, indent=2) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
