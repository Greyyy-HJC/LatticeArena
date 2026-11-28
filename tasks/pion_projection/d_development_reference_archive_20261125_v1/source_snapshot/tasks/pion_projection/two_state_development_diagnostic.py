"""Post-failure, development-only comparison of the registered single- and two-state fits."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from .common import load_data
from .reference import make_reference
from .statistics import blocks, fold


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output, protocol_path, bootstrap_path=None):
    output, protocol_path = Path(output), Path(protocol_path)
    p = json.loads(protocol_path.read_text())
    raw = load_data(output, 'development', p)
    data = blocks(fold(raw), p['block_length'])
    bootstrap_path = Path(bootstrap_path) if bootstrap_path else output / 'reference_bootstrap.npz'
    with np.load(bootstrap_path, allow_pickle=False) as z:
        indices = np.asarray(z['indices'], dtype=int)
    single = json.loads((output / 'submission.json').read_text())['calibration']
    rows, boot_by_window = [], []
    for window in p['reference_windows']:
        model = make_reference(data, p, window, 'twostate')
        full = model.fit_full(model.samples.mean(axis=0))
        dof = model.nchan * model.ntime - 2 - 2 * model.nchan
        masses = np.full((len(indices), 2), np.nan)
        failures = []
        for j, draw in enumerate(indices):
            try:
                fit = model.fit_full(model.samples[draw].mean(axis=0))
                masses[j] = [fit['mass'], fit['mass2']]
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                failures.append({'draw': j, 'reason': repr(exc)})
        valid = np.isfinite(masses).all(axis=1)
        se = masses[valid].std(axis=0, ddof=1) if valid.sum() > 1 else [np.nan, np.nan]
        row = {
            'window': list(window), 'mass': full['mass'], 'mass2': full['mass2'], 'gap': full['gap'],
            'mass_se': float(se[0]), 'mass2_se': float(se[1]), 'chi2': full['cost'], 'dof': dof,
            'p_approx': float(chi2.sf(full['cost'], dof)), 'condition': full['condition'],
            'amplitudes': full['amplitudes'], 'at_boundary': full['at_boundary'],
            'bootstrap_valid': int(valid.sum()), 'bootstrap_total': len(indices),
            'bootstrap_failures': failures,
            'mass_bounds_reached': bool(np.any((masses[valid] <= p['mass_bounds'][0] + 1e-4)
                                               | (masses[valid] >= p['mass_bounds'][1] - 1e-4))) if valid.any() else None,
        }
        rows.append(row)
        boot_by_window.append(masses)
        print(f"two-state window={window} mass={full['mass']:.6g} excited={full['mass2']:.6g} valid={valid.sum()}/{len(indices)}", flush=True)

    nominal = p['reference_nominal_window']
    nominal_boot = boot_by_window[nominal][:, 0]
    failures = []
    for j, row in enumerate(rows):
        if j == nominal:
            continue
        candidate = boot_by_window[j][:, 0]
        paired = np.isfinite(candidate) & np.isfinite(nominal_boot)
        delta = candidate[paired] - nominal_boot[paired]
        paired_se = float(delta.std(ddof=1)) if len(delta) > 1 else np.nan
        z = abs(row['mass'] - rows[nominal]['mass']) / paired_se if paired_se > 0 else np.inf
        row['paired_shift_z'] = float(z)
    for row in rows:
        if row['bootstrap_valid'] < len(indices):
            failures.append(f"Window {row['window']} has bootstrap fit failures ({row['bootstrap_valid']}/{len(indices)} valid)")
        if row['mass_se'] / abs(row['mass']) > p['reference_max_relative_error']:
            failures.append(f"Window {row['window']} ground-state relative bootstrap error exceeds threshold")
        if row['p_approx'] < p['reference_min_fit_p']:
            failures.append(f"Window {row['window']} approximate fit p-value is below threshold")
        if row.get('paired_shift_z', 0) > p['reference_max_window_z']:
            failures.append(f"Window {row['window']} shifts by {row['paired_shift_z']:.2f} paired standard errors")
        if row['mass_bounds_reached'] or row['at_boundary']:
            failures.append(f"Window {row['window']} reaches a mass bound")
    report = {
        'schema': 'pion-two-state-development-diagnostic-v1',
        'status': 'EXPLORATORY_DIAGNOSTIC_ONLY',
        'scope': 'post-failure comparison on D development IDs 0-99; no anchor, validation, gap, or shared holdout data read',
        'protocol_sha256': sha256(protocol_path),
        'measurement_sha256': sha256(output / 'development.npz'),
        'analysis_amendment_sha256': sha256(Path(__file__)),
        'bootstrap_indices_sha256': sha256(bootstrap_path),
        'n_development_configurations': len(raw), 'n_development_blocks': len(data),
        'block_length': p['block_length'], 'bootstrap_draws': len(indices),
        'single_state_reference_gate_passed': bool(single['reference_passed']),
        'single_state_windows': single['windows'],
        'two_state_windows': rows,
        'two_state_threshold_screen': {'passed': not failures, 'failures': failures,
                                       'interpretation': 'screen only; the two-state model was not part of the frozen primary reference, so a pass cannot reopen the anchor or certify a score'},
        'warning': 'A two-state fit can absorb excited-state curvature, but window robustness and fit p-values on this same development sample are not independent validation. Amplitudes are unconstrained; inspect signs and conditioning.',
    }
    (output / 'two_state_diagnostic.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    np.savez_compressed(output / 'two_state_bootstrap.npz', indices=indices,
                        masses=np.asarray(boot_by_window), windows=np.array(json.dumps(p['reference_windows'])))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].errorbar(x - .07, [r['mass'] for r in single['windows']],
                     yerr=[r['se'] for r in single['windows']], fmt='o-', capsize=3,
                     color='#B34142', label='single state')
    axes[0].errorbar(x + .07, [r['mass'] for r in rows], yerr=[r['mass_se'] for r in rows],
                     fmt='s-', capsize=3, color='#087F8C', label='two state (exploratory)')
    axes[0].set_xticks(x, [str(w) for w in p['reference_windows']])
    axes[0].set(xlabel='fit window', ylabel=r'ground-state mass $a m_0$', title='Window stability')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=.25)
    for label, values, marker, color in (
        ('single state', [r['p_approx'] for r in single['windows']], 'o', '#B34142'),
        ('two state', [r['p_approx'] for r in rows], 's', '#087F8C'),
    ):
        axes[1].plot(x, values, marker=marker, color=color, label=label)
    axes[1].axhline(p['reference_min_fit_p'], color='black', linestyle='--', linewidth=1,
                    label='frozen minimum p')
    axes[1].set_yscale('log')
    axes[1].set_xticks(x, [str(w) for w in p['reference_windows']])
    axes[1].set(xlabel='fit window', ylabel='approximate fit p-value', title='Fit adequacy (diagnostic)')
    axes[1].grid(alpha=.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('D development only: two-state model is exploratory after primary gate failure')
    image = output / 'two_state_diagnostic.png'
    fig.savefig(image, dpi=180)
    plt.close(fig)

    lines = [
        '# Two-state reference diagnostic on D development', '',
        '**Status: exploratory only; the frozen single-state reference gate remains failed.**', '',
        f"The two-state ansatz was fitted to the same 100 D-development configurations ({len(data)} configuration blocks) and the same three frozen time windows. Bootstrap indices were reused across windows to show paired shifts. No E anchor, D validation, gap, or shared holdout data were opened.", '',
        '![Single- and two-state window diagnostics](two_state_diagnostic.png)', '',
        '| Window | Two-state $am_0$ ± SE | $am_1$ ± SE | Approx. p | Paired shift | Condition | Bound reached? | Valid bootstrap |',
        '|---|---:|---:|---:|---:|---:|:---:|---:|',
    ]
    for row in rows:
        shift = row.get('paired_shift_z')
        shift_text = 'reference' if shift is None else f"{shift:.3g}σ"
        bound = 'yes' if row['mass_bounds_reached'] or row['at_boundary'] else 'no'
        lines.append(f"| {row['window']} | {row['mass']:.6g} ± {row['mass_se']:.3g} | {row['mass2']:.6g} ± {row['mass2_se']:.3g} | {row['p_approx']:.3g} | {shift_text} | {row['condition']:.3g} | {bound} | {row['bootstrap_valid']}/{row['bootstrap_total']} |")
    lines += ['', '## Interpretation boundary', '',
              '- A two-state fit can explain curvature caused by excited-state contamination, but improved in-sample fit is not proof that the second state is physically identified.',
              '- The `[6,7,8]` fit reaches the excited-state upper mass bound in its bootstrap and has condition number about $3\\times10^5$; this is a direct identifiability warning, despite the good in-sample p-value.',
              '- Read the amplitude signs and condition numbers in `two_state_diagnostic.json`; unconstrained amplitudes and short windows can produce unstable decompositions.',
              '- Even if this screen meets the numerical thresholds, it does not amend the frozen primary protocol, validate the independent E chain, or permit a physical score.',
              '- The next legitimate confirmation requires a separate frozen protocol/model hash and independent-chain check before opening E. Do not tune model or thresholds after E is measured.', '']
    (output / 'two_state_diagnostic.md').write_text('\n'.join(lines))
    provenance = {
        'schema': 'pion-two-state-development-diagnostic-provenance-v1',
        'analysis_source_sha256': sha256(Path(__file__)),
        'protocol_sha256': sha256(protocol_path),
        'measurement_sha256': sha256(output / 'development.npz'),
        'measurement_provenance_sha256': sha256(output / 'development_provenance.json'),
        'bootstrap_indices_sha256': sha256(bootstrap_path),
        'outputs_sha256': {name: sha256(output / name) for name in
                           ('two_state_diagnostic.json', 'two_state_bootstrap.npz',
                            'two_state_diagnostic.png', 'two_state_diagnostic.md')},
        'scope': report['scope'],
    }
    (output / 'two_state_diagnostic_provenance.json').write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--bootstrap', type=Path)
    args = parser.parse_args()
    result = run(args.output, args.protocol, args.bootstrap)
    print(json.dumps({'status': result['status'], 'threshold_screen': result['two_state_threshold_screen']}, indent=2))
