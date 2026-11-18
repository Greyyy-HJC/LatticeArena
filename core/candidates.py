"""One candidate validator for optimizers and LLM proposals (todo F2)."""
import numpy as np


def validate_weights(u, size, l1_bound=8.0, tolerance=1e-6):
    """Return (ok, reason). Rejects NaN/Inf, wrong shape, wrong sum, L1 overflow."""
    try:
        array = np.asarray(u, dtype=float)
    except (TypeError, ValueError):
        return False, 'not a numeric vector'
    if array.ndim != 1 or array.size != size:
        return False, f'expected a one-dimensional vector of length {size}, got shape {array.shape}'
    if not np.all(np.isfinite(array)):
        return False, 'non-finite coefficient'
    if abs(float(array.sum()) - 1.0) > tolerance:
        return False, f'coefficients must sum to 1, got {float(array.sum()):.6g}'
    if float(np.abs(array).sum()) > l1_bound + tolerance:
        return False, f'L1 norm exceeds {l1_bound}, got {float(np.abs(array).sum()):.6g}'
    return True, 'ok'
