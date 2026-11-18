"""Count invalid loop samples and declare rejection rules (todo P2)."""
import json
from pathlib import Path

import numpy as np

from .statistics import blocks, bootstrap_indices

T_VALUES = [4, 5, 6, 7, 8]
WINDOWS = [[4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
RUNS = {'v2': 'results/wilson_loop/s16t16_v2', 'v3': 'results/wilson_loop/s16t16_v3',
        'v4': 'results/wilson_loop/s16t16_v4'}


def count_invalid(data, index):
    rows = []
    for k in range(data.shape[1]):
        w = data[:, k]
        mean = w.mean(axis=0)
        boot = w[index].mean(axis=1)
        invalid = np.zeros(len(boot), dtype=bool)
        for window in WINDOWS:
            cols = [T_VALUES.index(t) for t in window]
            invalid |= (boot[:, :, cols] <= 0).any(axis=2).any(axis=1)
        rows.append({
            'kernel': k,
            'mean_nonpositive': int((mean <= 0).sum()),
            'mean_nan': int((~np.isfinite(mean)).sum()),
            'bootstrap_nonpositive': int((boot <= 0).sum()),
            'bootstrap_nan': int((~np.isfinite(boot)).sum()),
            'replicas_with_nonpositive_in_window': int(invalid.sum()),
        })
    return rows


def audit_run(run):
    root = Path(RUNS[run])
    data = blocks(np.load(root / 'validation.npz')['loops'], 5)
    index = bootstrap_indices(len(data), 500, 11)
    return count_invalid(data, index)


def main():
    out = {}
    for run in RUNS:
        if (Path(RUNS[run]) / 'validation.npz').exists():
            out[run] = audit_run(run)
    dest = Path('results/wilson_loop/invalid_audit_v1')
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'invalid_audit.json').write_text(json.dumps(
        {'runs': out,
         'rules': {'mean_nonpositive': 'invalidate the window (log undefined)',
                   'bootstrap_nonpositive': 'drop the replica from the SE term, record the count',
                   'nan': 'reject the kernel; never silently score NaN',
                   'zero_denominator': 'reject the replica (V_ref or W == 0)'}}, indent=2) + '\n')
    for run, rows in out.items():
        print(run)
        for row in rows:
            print(' ', row)


if __name__ == '__main__':
    main()
