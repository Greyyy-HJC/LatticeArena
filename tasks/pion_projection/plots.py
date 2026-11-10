"""Static, exportable figures from recorded results; no tuning or rescoring."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import load_data
from .statistics import blocks, fold, effective_mass


def plot(output):
    output = Path(output)
    frozen = json.loads((output/'submission.json').read_text())
    report = json.loads((output/'validation_results.json').read_text())
    raw = load_data(output, 'validation')
    p = report['protocol']
    data = blocks(fold(raw), p['block_length'])
    labels = {'local': 'Local sink', 'tuned_gaussian': 'Tuned Gaussian ($H^{32}$)',
              'agent_signed_mixture': 'Agent signed mixture'}
    colors = {'local': '#536879', 'tuned_gaussian': '#D28B26', 'agent_signed_mixture': '#087F8C'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for name, item in report['results'].items():
        if 'risk' not in item:
            continue
        color, label = colors[name], labels[name]
        axes[0, 0].errorbar(p['score_window'], item['q'], yerr=item['q_se'], marker='o',
                           color=color, label=label, capsize=3)
        axes[0, 1].plot(np.arange(1, 9), np.array(item['snr'])[1:], 'o-', color=color, label=label)
        values = np.einsum('nbt,b->nt', data, frozen['submissions'][name])
        mean = values.mean(axis=0)
        meff = effective_mass(mean)
        axes[1, 0].plot(np.arange(1, 8), meff, 'o-', color=color, label=label)
    axes[0, 0].axhline(1, color='black', lw=.8, ls='--')
    axes[0, 0].set(xlabel='$t/a$', ylabel='$C(t)/(A f_0(t))$', title='Departure from provisional reference')
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].set(xlabel='$t/a$', ylabel=r'$|\overline{C}| / \mathrm{SE}(\overline{C})$',
                   title='Validation signal-to-noise', yscale='log')
    axes[1, 0].axhline(report['reference']['mass'], color='black', lw=.8, ls='--', label='Provisional $m_0$')
    axes[1, 0].set(xlabel='$t/a$', ylabel=r'$a m_{\mathrm{eff}}$', title='Cosh effective mass of mean correlator')
    axes[1, 0].legend(fontsize=9)
    windows = frozen['calibration']['windows']
    axes[1, 1].errorbar([w['window'][0] for w in windows], [w['mass'] for w in windows],
                       yerr=[w['se'] for w in windows], marker='o', color='#B34142', capsize=4)
    axes[1, 1].set(xlabel=r'Reference fit $t_{\min}/a$ ($t_{\max}/a=8$)', ylabel='Fitted $a m_0$',
                   title='Development reference fails window stability', xticks=[4, 5, 6])
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('S16T16 pilot: 30 development + 15 validation configurations\n'
                 'Reference checks failed: diagnostics only, no official score', fontsize=13)
    fig.savefig(output/'pilot_results.png', dpi=180)
    fig.savefig(output/'pilot_results.pdf')
    plt.close(fig)
