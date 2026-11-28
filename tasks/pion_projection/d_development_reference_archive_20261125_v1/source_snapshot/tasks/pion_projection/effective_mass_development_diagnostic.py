"""Development-only cosh effective-mass diagnostic for the D-chain pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import digest
from .statistics import blocks, fold


def run(run_dir, protocol_path, bootstrap_path, output_dir):
    run_dir = Path(run_dir).resolve()
    protocol_path = Path(protocol_path).resolve()
    bootstrap_path = Path(bootstrap_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    p = json.loads(protocol_path.read_text())
    provenance = json.loads((run_dir / 'development_provenance.json').read_text())
    with np.load(run_dir / 'development.npz', allow_pickle=False) as saved:
        raw = np.asarray(saved['correlators'], dtype=float)
        ids = np.asarray(saved['ids'], dtype=int)
        generation = str(saved['generation'])
        protocol_hash = str(saved['protocol_hash'])
    with np.load(bootstrap_path, allow_pickle=False) as saved:
        indices = np.asarray(saved['indices'], dtype=int)
    if generation != provenance['generation'] or protocol_hash != provenance['protocol_hash']:
        raise ValueError('development NPZ and provenance identity differ')
    if raw.ndim != 3 or len(raw) != len(ids) or len(raw) != len(provenance['records']):
        raise ValueError('unexpected development measurement shape or record count')
    if ids.tolist() != list(range(100)):
        raise ValueError('this diagnostic is frozen for D development IDs 0-99')
    if raw.shape[-1] != int(p['geometry'][3]):
        raise ValueError('correlator temporal extent differs from the protocol')
    if [int(row['id']) for row in provenance['records']] != ids.tolist():
        raise ValueError('development record IDs differ from the NPZ')
    block_length = int(p['block_length'])
    folded = fold(raw)
    grouped = blocks(folded, block_length)
    if indices.ndim != 2 or indices.shape[1] != len(grouped):
        raise ValueError('bootstrap indices do not match the registered complete blocks')
    if np.any(indices < 0) or np.any(indices >= len(grouped)):
        raise ValueError('bootstrap index is out of range')
    draws = grouped[indices].mean(axis=1)
    point = grouped.mean(axis=0)
    times = np.arange(1, folded.shape[-1] - 1)
    # For C(t) proportional to cosh[m(T/2-t)],
    # (C(t-1)+C(t+1))/(2 C(t)) = cosh(m).
    def transform(c):
        center = c[..., 1:-1]
        ratio = (c[..., :-2] + c[..., 2:]) / (2 * center)
        valid = np.isfinite(ratio) & (ratio >= 1.0) & (center > 0)
        value = np.full(ratio.shape, np.nan, dtype=float)
        value[valid] = np.arccosh(ratio[valid])
        return value

    point_mass = transform(point)
    boot_mass = transform(draws)
    powers = p.get('powers', list(range(raw.shape[1])))
    if len(powers) != raw.shape[1]:
        raise ValueError('protocol powers do not match correlator channels')
    submission = json.loads((run_dir / 'submission.json').read_text())
    nominal = submission['calibration']['reference']
    fit_mass = float(nominal['mass'])
    fit_se = float(nominal['se'])
    channel_results = []
    for channel, power in enumerate(powers):
        row = {'channel': channel, 'sink_power': power, 'times': []}
        for it, time in enumerate(times):
            values = boot_mass[:, channel, it]
            values = values[np.isfinite(values)]
            row['times'].append({
                't': int(time), 'point_am': float(point_mass[channel, it]),
                'valid_bootstrap': int(len(values)), 'total_bootstrap': int(len(indices)),
                'median_am': float(np.median(values)) if len(values) else None,
                'q16_am': float(np.quantile(values, 0.16)) if len(values) else None,
                'q84_am': float(np.quantile(values, 0.84)) if len(values) else None,
                'q50_am': float(np.quantile(values, 0.50)) if len(values) else None,
            })
        channel_results.append(row)

    report = {
        'schema': 'pion-effective-mass-development-diagnostic-v1',
        'status': 'EXPLORATORY_DIAGNOSTIC_ONLY',
        'scope': 'D-chain development IDs 0-99 only; no E anchor, gap, validation, or shared holdout opened',
        'measurement_sha256': digest(run_dir / 'development.npz'),
        'measurement_provenance_sha256': digest(run_dir / 'development_provenance.json'),
        'measurement_generation': generation, 'measurement_protocol_hash': protocol_hash,
        'analysis_protocol_sha256': digest(protocol_path),
        'bootstrap_indices_sha256': digest(bootstrap_path),
        'analysis_script_sha256': digest(Path(__file__)),
        'configuration_ids': ids.tolist(), 'n_configurations': int(len(ids)),
        'n_complete_blocks': int(len(grouped)), 'block_length': block_length,
        'n_bootstrap_draws': int(len(indices)), 'temporal_extent': int(raw.shape[-1]),
        'effective_mass_definition': 'arccosh((C_folded(t-1)+C_folded(t+1))/(2*C_folded(t)))',
        'fit_reference_for_visual_comparison': {'single_state_am': fit_mass, 'bootstrap_se': fit_se,
                                                'fit_window': nominal['window']},
        'channels': channel_results,
        'warning': 'The cosh effective-mass ratio is a diagnostic transform of the same D-development correlators. It is correlated across time and smearing channels and does not independently validate a plateau or override the frozen failed reference gate.'
    }
    json_path = output_dir / 'effective_mass_diagnostic.json'
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    nchan = len(powers)
    ncols = 2
    nrows = (nchan + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.0*nrows), sharex=True,
                             sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for channel, (ax, row) in enumerate(zip(axes, channel_results)):
        valid_points = [item for item in row['times'] if item['valid_bootstrap'] > 0]
        x = np.array([item['t'] for item in valid_points])
        med = np.array([item['median_am'] for item in valid_points])
        lo = np.array([item['q16_am'] for item in valid_points])
        hi = np.array([item['q84_am'] for item in valid_points])
        point_values = np.array([item['point_am'] for item in valid_points])
        ax.errorbar(x, point_values, fmt='o', color='#345995', markersize=4,
                    label='full-sample effective mass')
        ax.fill_between(x, lo, hi, color='#4C78A8', alpha=.2,
                        label='bootstrap 16–84% interval')
        ax.plot(x, med, color='#4C78A8', linewidth=1.2)
        ax.axhspan(fit_mass-fit_se, fit_mass+fit_se, color='#E45756', alpha=.12)
        ax.axhline(fit_mass, color='#E45756', linestyle='--', linewidth=1,
                   label='failed single-state reference ±1 SE')
        for window in p['reference_windows']:
            for t in window:
                ax.axvline(t, color='grey', alpha=.08, linewidth=.7)
        ax.set_title(f'sink power {row["sink_power"]}')
        ax.set_ylabel(r'$a m_{eff}(t)$')
        ax.grid(alpha=.2)
    for ax in axes[nchan:]:
        ax.set_visible(False)
    for ax in axes[:nchan]:
        ax.set_xlabel('central time $t$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.015), fontsize=8)
    fig.suptitle('Pion D development only: cosh effective-mass scan (exploratory)', y=1.04)
    fig.savefig(output_dir / 'effective_mass_diagnostic.png', dpi=180,
                bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    late_times = np.arange(3, folded.shape[-1] - 1)
    colors = plt.cm.viridis(np.linspace(.08, .92, nchan))
    for row, color in zip(channel_results, colors):
        selected = [item for item in row['times'] if item['t'] in late_times]
        x = np.asarray([item['t'] for item in selected])
        center = np.asarray([item['q50_am'] for item in selected])
        lo = np.asarray([item['q16_am'] for item in selected])
        hi = np.asarray([item['q84_am'] for item in selected])
        ax.errorbar(x, center, yerr=np.vstack((center-lo, hi-center)), fmt='o-',
                    capsize=2, markersize=4, linewidth=1, color=color,
                    label=f"sink power {row['sink_power']}")
    ax.axhspan(fit_mass-fit_se, fit_mass+fit_se, color='#E45756', alpha=.12,
               label='failed single-state reference ±1 SE')
    ax.axhline(fit_mass, color='#E45756', linestyle='--', linewidth=1)
    ax.set(xlim=(2.7, float(late_times[-1])+.3), ylim=(.95, 1.55),
           xticks=late_times, xlabel='central time $t$', ylabel=r'$a m_{eff}(t)$',
           title='Late-time effective mass with bootstrap intervals')
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output_dir / 'effective_mass_late_time.png', dpi=180,
                bbox_inches='tight')
    plt.close(fig)

    invalid = sum(1 for channel in channel_results for item in channel['times']
                  if item['valid_bootstrap'] != len(indices))
    lines = [
        '# Pion effective-mass scan on D development', '',
        '**Status: exploratory diagnostic only; the frozen single-state reference gate remains failed.**', '',
        '![Cosh effective-mass scan](effective_mass_diagnostic.png)', '',
        '![Late-time effective masses with bootstrap intervals](effective_mass_late_time.png)', '',
        f"- Data: {len(ids)} configurations (IDs {ids.min()}–{ids.max()}), {len(grouped)} complete blocks of length {block_length}, {len(indices)} paired bootstrap draws.",
        f"- Channels: sink powers {', '.join(map(str, powers))}; plotted points use all D-development configurations, with the 16–84% bootstrap band from the same complete block resamples.",
        f"- Invalid effective-mass bootstrap channel/time cells: {invalid}; invalid draws are retained in the JSON counts rather than clipped into a mass estimate.",
        f"- The red band/line marks the failed single-state reference, $am={fit_mass:.6g}\\pm{fit_se:.3g}$, fitted on window {nominal['window']}.", '',
        'The estimator uses the folded periodic-cosh identity $m_{eff}(t)=\\operatorname{arccosh}[(C(t-1)+C(t+1))/(2C(t))]$. Time points and channels are correlated; the plot is a spectral-shape diagnostic, not an independent plateau test. It cannot replace the frozen reference gate or authorize E, validation, or shared-holdout access.', '',
        f"Hashes and per-channel/time values are in `effective_mass_diagnostic.json` (measurement `{report['measurement_sha256']}`, source `{report['analysis_script_sha256']}`).", ''
    ]
    (output_dir / 'effective_mass_diagnostic.md').write_text('\n'.join(lines))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--bootstrap', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run_dir, args.protocol, args.bootstrap, args.output_dir)
    print(json.dumps({'status': result['status'], 'n_bootstrap_draws': result['n_bootstrap_draws'],
                      'channels': len(result['channels'])}, indent=2))
