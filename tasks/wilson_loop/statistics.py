import numpy as np

from core.stats import blocks, bootstrap_indices  # noqa: F401  (re-export)


def potential(w, window, t_values):
    """V(R) from -d log W / dT over a fixed T window; w: (..., nR, nT)."""
    t = np.asarray(window, dtype=float)
    y = np.log(w[..., [list(t_values).index(x) for x in window]])
    x = t - t.mean()
    return -(y * x).sum(-1) / (x * x).sum()


def risk(w_blocks, index, windows, t_values, v_ref, nominal=1):
    """Precision (bootstrap SE) plus plateau spread across T windows."""
    mean = w_blocks.mean(0)
    boot_mean = w_blocks[index].mean(1)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(boot_mean)):
        raise ValueError('non-finite W in the risk inputs')
    if np.any(mean <= 0) or np.any(boot_mean <= 0):
        raise ValueError(f'non-positive W (mean {int((mean <= 0).sum())}, '
                         f'bootstrap {int((boot_mean <= 0).sum())}); log is undefined')
    v = potential(mean, windows[nominal], t_values)
    vb = potential(boot_mean, windows[nominal], t_values)
    se = vb.std(axis=0, ddof=1)
    v_win = np.stack([potential(mean, w, t_values) for w in windows], axis=-1)
    spread = v_win.std(axis=-1)
    v_ref = np.asarray(v_ref)
    return {'risk': float(np.mean((se / v_ref) ** 2 + (spread / v_ref) ** 2)),
            'v': v.tolist(), 'se': se.tolist(), 'spread': spread.tolist(),
            'v_windows': v_win.tolist()}
