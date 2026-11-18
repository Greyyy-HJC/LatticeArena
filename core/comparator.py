"""Shared paired comparator (todo F1, slice 4).

Draw arrays must share one length; the comparator never broadcasts. The log
lower bound is the same statistic used by the pion certificate.
"""
import hashlib

import numpy as np


def check_draws(risks, draw_ids=None, finite=True):
    """Every array must be 1-D, the same length, and finite; no broadcasting."""
    arrays = {name: np.asarray(value) for name, value in risks.items()}
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f'draw array {name!r} must be one-dimensional, got shape {array.shape}')
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f'draw arrays have different lengths {sorted(shapes)}')
    n = shapes.pop()[0]
    if finite:
        for name, array in arrays.items():
            if not np.all(np.isfinite(array.astype(float))):
                raise ValueError(f'draw array {name!r} has non-finite entries')
    if draw_ids is not None:
        ids = np.asarray(draw_ids)
        if ids.ndim != 1 or ids.size != n:
            raise ValueError('draw_ids must be a one-dimensional array of the same length')
        if np.unique(ids).size != n:
            raise ValueError('draw_ids contain duplicates; each draw must be unique')
        digest = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
        return n, digest
    return n, None


def paired_ratio(baseline, candidate, draw_ids=None):
    """Median and central 95% interval of baseline/candidate per shared draw."""
    check_draws({'baseline': baseline, 'candidate': candidate}, draw_ids)
    ratio = np.asarray(baseline, dtype=float) / np.asarray(candidate, dtype=float)
    return {'median': float(np.median(ratio)),
            'ci95': np.quantile(ratio, [0.025, 0.975]).tolist(),
            'exceedance_fraction': float(np.mean(ratio > 1)),
            'draws': int(ratio.size)}


def log_lower_bound(baseline, candidate, level, floor=0.0):
    """Lower quantile of log(base+floor) - log(cand+floor) at the given level."""
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    valid = np.isfinite(base) & np.isfinite(cand) & (base > 0) & (cand > 0)
    if not np.any(valid):
        return None, int((~valid).sum())
    difference = np.log(base[valid] + floor) - np.log(cand[valid] + floor)
    return float(np.quantile(difference, level)), int((~valid).sum())
