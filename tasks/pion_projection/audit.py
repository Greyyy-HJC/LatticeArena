"""Read-only CPU audit of archived pion runs (todo Phase B).

Writes a new diagnostics directory; never modifies the archived runs.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .experiment import reflection_odd
from .statistics import ReferenceFit, blocks, bootstrap_indices, fold

OUT = Path('results/pion_projection/score_diagnostics_v2')
RUNS = ['s16t16_v1', 's16t16_twostate', 's16t16_t32']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    out = {'path': str(root)}
    for split in ('development', 'validation'):
        path = root / f'{split}.npz'
        if not path.exists():
            continue
        z = np.load(path)
        c = z['correlators']
        out[split] = {'shape': list(c.shape), 'dtype': str(c.dtype),
                      'finite': bool(np.all(np.isfinite(c))), 'sha256': sha256(path),
                      'ids': z['ids'].tolist() if 'ids' in z else None}
    return out


def protocol_of(root):
    return json.loads((root / 'development_provenance.json').read_text())['protocol']


def reference_scan(root):
    p = protocol_of(root)
    raw = np.load(root / 'development.npz')['correlators']
    data = blocks(fold(raw), p['block_length'])
    rows, spectra = [], {}
    for window in p['reference_windows']:
        model = ReferenceFit(data, p, window)
        result, boot = model.bootstrap(bootstrap_indices(len(data), 200, p['seed']))
        cov = np.atleast_2d(np.cov(model.samples, rowvar=False, ddof=1)) / len(model.samples)
        ev = np.linalg.eigvalsh(cov)
        spectra[str(window)] = {'eigenvalues': ev.tolist(),
                                'condition_number': float(ev[-1] / max(ev[0], 1e-300)),
                                'below_floor': int(np.sum(ev < ev[-1] * 1e-10))}
        rows.append({k: result[k] for k in ('window', 'mass', 'se', 'chi2', 'dof', 'p_approx')})
    return {'protocol_version': p.get('version'), 'n_blocks': len(data), 'windows': rows,
            'covariance_spectrum': spectra}


def reflection_check(root):
    p = protocol_of(root)
    raw = np.load(root / 'development.npz')['correlators']
    odd = blocks(reflection_odd(raw), p['block_length'])
    se = odd.std(axis=0, ddof=1) / np.sqrt(len(odd))
    z = np.abs(odd.mean(axis=0) / se)
    archived = None
    path = root / 'submission.json'
    if path.exists():
        archived = json.loads(path.read_text()).get('calibration', {}).get('reflection_max_z')
    return {'temporal': raw.shape[-1], 'max_z': float(np.max(z)),
            'z_by_t': z.max(axis=0).tolist(), 'archived_max_z': archived}


def folded_plot(root, name):
    p = protocol_of(root)
    raw = np.load(root / 'development.npz')['correlators']
    folded = fold(raw)
    t = np.arange(raw.shape[-1])
    tf = np.arange(folded.shape[-1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k in range(raw.shape[1]):
        axes[0].plot(t, raw[:, k].mean(0), 'o-', ms=3, label=f'$n={p["powers"][k]}$')
        axes[1].plot(tf, folded[:, k].mean(0), 'o-', ms=3)
    for ax in axes:
        ax.set_yscale('log')
        ax.grid(alpha=0.2)
        ax.set_xlabel('$t/a$')
    axes[0].set_ylabel('mean correlator (configuration mean)')
    axes[0].set_title('raw')
    axes[1].set_title('folded $[C(t)+C(T-t)]/2$')
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f'folded_vs_raw_{name}.png', dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inv, scan, refl = {}, {}, {}
    for name in RUNS:
        root = Path('results/pion_projection') / name
        if not (root / 'development.npz').exists():
            continue
        inv[name] = inventory(root)
        scan[name] = reference_scan(root)
        refl[name] = reflection_check(root)
        folded_plot(root, name)
    (OUT / 'inventory.json').write_text(json.dumps(inv, indent=2) + '\n')
    (OUT / 'reference_scan.json').write_text(json.dumps(scan, indent=2) + '\n')
    (OUT / 'reflection_check.json').write_text(json.dumps(refl, indent=2) + '\n')
    for name, rows in scan.items():
        print(name, [(r['window'], round(r['mass'], 4), round(r['p_approx'], 4)) for r in rows['windows']])
        print('  reflection', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in refl[name].items() if k != 'z_by_t'})


if __name__ == '__main__':
    main()
