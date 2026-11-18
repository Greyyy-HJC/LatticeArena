"""Run the bootstrap GEVP estimator on the Gram data (exploratory; todo P0.2/P0.3)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .gevp_stats import bootstrap_gevp, dedupe, summarise

RUN = Path('results/pion_projection/s16t16_gram3')
OUT = Path('results/pion_projection/gram_stats_v2')


def block_index(rng, ncfg, block=5, replicas=200):
    count = ncfg // block
    starts = rng.integers(0, count, (replicas, count))
    return (starts[:, :, None] * block + np.arange(block)).reshape(replicas, count * block)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(RUN / 'development.npz')
    names = [str(x) for x in z['names']]
    raw = z['gram']
    matrix, keep = dedupe(raw)
    kept = [names[i] for i in keep]
    spectrum = np.linalg.eigvalsh(matrix[:, :, :, 2].mean(axis=0))[::-1]
    effective_rank = int((spectrum > 1e-6 * spectrum[0]).sum())
    rank = min(effective_rank, 4)
    rng = np.random.default_rng(20261118)
    index = block_index(rng, matrix.shape[0])
    result = {'kept': kept, 'effective_rank': effective_rank, 'rank_used': rank,
              'metric_spectrum': spectrum.tolist(), 'estimators': {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(np.arange(1, len(spectrum) + 1), spectrum, 'o-', label='deduped')
    raw_spec = np.linalg.eigvalsh(raw[:, :, :, 2].mean(axis=0))[::-1]
    axes[0].semilogy(np.arange(1, len(raw_spec) + 1), raw_spec, 's--', label='raw')
    axes[0].set(xlabel='mode', ylabel='eigenvalue at $t_0=2$', title='metric spectrum')
    axes[0].legend()
    for model in ('forward', 'cosh'):
        energies, failures = bootstrap_gevp(matrix, index, t0=2, times=(3, 4, 5, 6),
                                            model=model, rank=rank)
        summary = summarise(energies, (3, 4, 5, 6), mode=0)
        result['estimators'][model] = {'summary': summary, 'failures': len(failures),
                                       'failure_reasons': failures[:5]}
        per_time = [summary['per_time'][t] for t in (3, 4, 5, 6)]
        per_se = [summary['per_time_se'][t] for t in (3, 4, 5, 6)]
        axes[1].errorbar([3, 4, 5, 6], per_time, yerr=per_se, fmt='o-', capsize=3, label=model)
        print(model, 'E0 %.4f +- %.4f' % (summary['energy'], summary['se']),
              {t: round(v, 3) for t, v in summary['per_time'].items()}, 'failures', len(failures))
    axes[1].set(xlabel='t', ylabel='$E_0(t)$', title='bootstrap GEVP ($t_0=2$)')
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / 'gevp_bootstrap.png', dpi=150)
    (OUT / 'gevp_bootstrap.json').write_text(json.dumps(result, indent=2) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
