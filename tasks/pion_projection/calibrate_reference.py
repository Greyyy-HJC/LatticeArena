"""Compare the asymptotic and simulation-calibrated reference p-values."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from tasks.pion_projection.calibration import calibrated_p, fitted_mean
from tasks.pion_projection.statistics import ReferenceFit, blocks, bootstrap_indices, fold

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/pion_projection/s16t16_v1')


def main():
    p = json.loads((RUN / 'development_provenance.json').read_text())['protocol']
    data = blocks(fold(np.load(RUN / 'development.npz')['correlators']), p['block_length'])
    out = {'windows': [], 'n_blocks': len(data)}
    nominal = p['reference_nominal_window']
    for i, window in enumerate(p['reference_windows']):
        model = ReferenceFit(data, p, window)
        mean = model.samples.mean(0)
        mass, cost, _ = fitted_mean(model, mean)
        dof = len(mean) - len(model.basis) - 1
        cal, _, costs = calibrated_p(model, mean, nsim=300, seed=100 + i)
        out['windows'].append({'window': window, 'mass': float(mass), 'cost': float(cost),
                               'dof': int(dof), 'p_asymptotic': float(chi2.sf(cost, dof)),
                               'p_calibrated': cal,
                               'null_median': float(np.median(costs)),
                               'null_q95': float(np.quantile(costs, 0.95))})
        print(json.dumps(out['windows'][-1]), flush=True)

    # calibration check: simulate 50 datasets from the nominal fitted model
    model = ReferenceFit(data, p, p['reference_windows'][nominal])
    mean = model.samples.mean(0)
    _, _, mu = fitted_mean(model, mean)
    cov = np.linalg.pinv(model.inv)
    cov = 0.5 * (cov + cov.T) + 1e-13 * np.eye(len(mu))
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(7)
    fails = 0
    n = 50
    for _ in range(n):
        sim = mu + chol @ rng.normal(size=len(mu))
        fails += calibrated_p(model, sim, nsim=100, seed=int(rng.integers(1e6)))[0] < 0.01
    out['false_fail_rate_at_1pct'] = fails / n
    (RUN / 'calibration.json').write_text(json.dumps(out, indent=2) + '\n')
    print('FALSE_FAIL', out['false_fail_rate_at_1pct'])


if __name__ == '__main__':
    main()
