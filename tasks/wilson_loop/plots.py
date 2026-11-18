"""Static-potential figures from recorded results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import protocol


def plot(output):
    output = Path(output)
    p = protocol()
    report = json.loads((output / 'validation_results.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, value in report['results'].items():
        v = value['v']
        err = [1.96 * s for s in value['se']]
        axes[0].errorbar(p['r_values'], v, yerr=err, marker='o', capsize=3, label=name)
    axes[0].set(xlabel='$R/a$', ylabel='$aV(R)$', title='Static potential (validation)')
    axes[0].legend(fontsize=8)
    names = list(report['results'])
    axes[1].bar(range(len(names)), [report['results'][n]['risk'] for n in names])
    axes[1].set_yscale('log')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[1].set(ylabel='risk', title='Validation risk (lower is better)')
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Wilson loop pilot: {report['n_validation']} configurations, "
                 f"reference {'passed' if report['reference_passed'] else 'failed'}")
    fig.tight_layout()
    fig.savefig(output / 'potential.png', dpi=160)
    plt.close(fig)
    print(f'wrote {output / "potential.png"}')


def paired_plot(output):
    output = Path(output)
    paired = json.loads((output.parent / f'paired_{output.name}' / 'paired.json').read_text())
    names = list(paired['candidates'])
    med = [100 * c['median'] for c in (paired['candidates'][n] for n in names)]
    low = [100 * c['ci95'][0] for c in (paired['candidates'][n] for n in names)]
    high = [100 * c['ci95'][1] for c in (paired['candidates'][n] for n in names)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(names)), med, color='#087F8C')
    err = [[med[i] - low[i] for i in range(len(names))],
           [high[i] - med[i] for i in range(len(names))]]
    ax.errorbar(range(len(names)), med, yerr=err, fmt='none', ecolor='k', capsize=4)
    ax.axhline(100, color='k', ls='--', lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set(ylabel='score relative to identity', title='Paired improvement (95% interval)')
    ax.grid(alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(output.parent / f'paired_{output.name}' / 'score_paired.png', dpi=150)
    plt.close(fig)
    print(f'wrote {output.parent / f"paired_{output.name}" / "score_paired.png"}')
