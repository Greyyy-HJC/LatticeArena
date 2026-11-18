"""Anchor sensitivity with real block draws (C3).

Replaces the pseudo-replication in energy_experiment.anchor_sensitivity: the
baseline/candidate energies come from block-bootstrap draws of independent
datasets, the anchor is sampled independently and shared by both candidates, and
the naive bootstrap interval is compared with the independent-dataset interval.
"""
import json
from pathlib import Path

import numpy as np

from .energy_experiment import correlators
from .energy_target import lower_bound
from .statistics import ReferenceFit, blocks, bootstrap_indices

OUT = Path('results/pion_projection/anchor_calibration_v1')
PROTOCOL = json.loads(Path('results/pion_projection/s16t16_v1/development_provenance.json').read_text())['protocol']
WINDOW = PROTOCOL['reference_windows'][PROTOCOL['reference_nominal_window']]
TAU_E = 0.01


def mass_draws(raw, rng, n_boot=100):
    data = blocks(raw, PROTOCOL['block_length'])
    ref = ReferenceFit(data, PROTOCOL, WINDOW)
    index = rng.integers(0, len(data), (n_boot, len(data)))
    _, masses = ref.bootstrap(index)
    return np.asarray(masses)


def run_cell(rng, candidate_noise, sigma_anchor, bias_anchor, n_dataset=30, n_boot=100, e0=0.80):
    naive_hits = 0
    dataset_deltas = []
    for _ in range(n_dataset):
        e_b = mass_draws(correlators(rng, noise=0.02), rng, n_boot)
        e_c = mass_draws(correlators(rng, noise=candidate_noise), rng, n_boot)
        anchor = e0 + bias_anchor + (rng.normal(0, sigma_anchor, len(e_b)) if sigma_anchor else 0.0)
        delta = ((e_b - anchor) ** 2 - (e_c - anchor) ** 2) / TAU_E ** 2
        if len(delta) >= 10:
            naive_hits += lower_bound(delta, 0.05) > 0
        dataset_deltas.append(float(delta.mean()))
    dataset_deltas = np.asarray(dataset_deltas)
    return {'candidate_noise': candidate_noise, 'sigma_anchor': sigma_anchor, 'bias_anchor': bias_anchor,
            'n_dataset': n_dataset, 'n_boot': n_boot,
            'naive_certified_fraction': naive_hits / n_dataset,
            'dataset_mean': float(dataset_deltas.mean()),
            'dataset_se': float(dataset_deltas.std(ddof=1) / np.sqrt(n_dataset)),
            'dataset_certified': (bool(lower_bound(dataset_deltas, 0.05) > 0)
                                  if len(dataset_deltas) >= 10 else None)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261121)
    rows = []
    for scenario, candidate_noise in (('improvement', 0.01), ('null', 0.02)):
        for sigma in (0.0, 0.005, 0.01, 0.02):
            row = run_cell(rng, candidate_noise, sigma, 0.0)
            row['scenario'] = scenario
            rows.append(row)
            print(scenario, 'sigma', sigma, 'naive', row['naive_certified_fraction'],
                  'dataset_certified', row['dataset_certified'])
    for bias in (0.01, 0.02, 0.05):
        row = run_cell(rng, 0.01, 0.0, bias)
        row['scenario'] = 'anchor_bias'
        rows.append(row)
        print('bias', bias, 'naive', row['naive_certified_fraction'])
    scaling = []
    for n_boot in (50, 200, 800):
        widths = []
        for _ in range(10):
            e_b = mass_draws(correlators(rng, noise=0.02), rng, n_boot)
            e_c = mass_draws(correlators(rng, noise=0.01), rng, n_boot)
            anchor = 0.80 + rng.normal(0, 0.01, len(e_b))
            delta = ((e_b - anchor) ** 2 - (e_c - anchor) ** 2) / TAU_E ** 2
            widths.append(float(delta.mean() - lower_bound(delta, 0.05)))
        scaling.append({'n_boot': n_boot, 'mean_naive_interval_width': float(np.mean(widths)),
                        'relative_to_50': float(np.mean(widths) / scaling[0]['mean_naive_interval_width']) if scaling else 1.0})
    result = {'rows': rows, 'nboot_scaling': scaling,
              'note': ('dataset_se is the independent-dataset standard error; naive uses '
                       'the bootstrap draws as if independent and shrinks with Nboot')}
    (OUT / 'anchor_calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    print('scaling', scaling)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
