"""End-to-end calibration of the candidate improvement certificate (C2).

Each simulation generates independent development/validation worlds, fits the
reference and its gate on development, runs joint draws, and evaluates the
certificate. Scenarios: no improvement, a known improvement (lower excited-state
contamination), and a misspecified single-cosh reference.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .reference import make_reference
from .scoring import improvement_certificate
from .statistics import blocks, bootstrap_indices, risk

OUT = Path('results/pion_projection/certificate_calibration_v1')
PROTOCOL = json.loads(Path('results/pion_projection/s16t16_v1/development_provenance.json').read_text())['protocol']
E0, E1 = 1.0, 1.5
AMPS = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15])
CONTAM = np.array([1.0, 0.85, 0.7, 0.5, 0.35, 0.25, 0.15])   # excited-state fraction per channel


def world(rng, ncfg, contam=None, noise=0.02, cfg_sigma=0.01, time_rho=0.5):
    T = PROTOCOL['geometry'][3]
    t = np.arange(T)
    r = (CONTAM if contam is None else contam)[None, :]
    signal = (AMPS[:, None] * (np.exp(-E0 * t) + np.exp(-E0 * (T - t)))[None, :]
              + r.T * AMPS[:, None] * (np.exp(-E1 * t) + np.exp(-E1 * (T - t)))[None, :])
    cfg = rng.normal(size=(ncfg, len(AMPS), 1)) * cfg_sigma
    eps = rng.normal(size=(ncfg, len(AMPS), T))
    for k in range(1, T):
        eps[:, :, k] = time_rho * eps[:, :, k - 1] + np.sqrt(1 - time_rho ** 2) * rng.normal(size=(ncfg, len(AMPS)))
    return signal[None] * (1 + cfg + noise * eps)


def gate(dev_blocks):
    ref = make_reference(dev_blocks, PROTOCOL,
                         PROTOCOL['reference_windows'][PROTOCOL['reference_nominal_window']], 'single')
    mean = ref.samples.mean(axis=0)
    mass, cost = ref.fit(mean)
    dof = ref.samples.shape[1] - len(PROTOCOL['reference_basis']) - 1
    return ref, mass, float(chi2.sf(cost, dof))


def run_simulation(rng, contam_baseline, contam_candidate, outer=60, inner=24, blocks_dev=24):
    ncfg = blocks_dev * PROTOCOL['block_length']
    dev = blocks(world(rng, ncfg, contam_baseline), PROTOCOL['block_length'])
    val = blocks(world(rng, 12 * PROTOCOL['block_length'], contam_baseline), PROTOCOL['block_length'])
    ref, mass, p_value = gate(dev)
    if p_value < 0.05:
        return {'abstain': True, 'p_value': p_value}
    baseline = np.zeros(len(AMPS))
    baseline[1] = 1.0
    candidate = np.zeros(len(AMPS))
    candidate[2] = 1.0
    draw_ids = np.arange(outer)
    risks = {'base': np.zeros(outer), 'cand': np.zeros(outer)}
    for b in range(outer):
        dev_idx = rng.integers(0, len(dev), len(dev))
        ref_b = make_reference(dev[dev_idx], PROTOCOL,
                               PROTOCOL['reference_windows'][PROTOCOL['reference_nominal_window']], 'single')
        fit_b, masses = ref_b.bootstrap(rng.integers(0, len(dev), (inner, len(dev))))
        val_idx = rng.integers(0, len(val), len(val))
        inner_idx = rng.integers(0, len(val), (inner, len(val)))
        for name, weights in (('base', baseline), ('cand', candidate)):
            risks[name][b] = risk(val[val_idx], weights, fit_b['mass'], masses, inner_idx, PROTOCOL)['risk']
    spec = {'schema': 'pion-score-v1', 'version': 'c2', 'estimand': 'conditional',
            'baseline_id': 'base', 'accepted_models': ['single'], 'candidate_ids': ['cand'],
            'analysis_version': 'c2', 'alpha': 0.05,
            'comparison_family': {'candidates': 1, 'models': 1},
            'min_baseline_risk': 1e-9, 'invalid_tolerance': 0.05, 'min_improvement': 1.0,
            'risk_floor': 0.0, 'require_draw_ids': True}
    certificate = improvement_certificate({'single': {'base': risks['base'], 'cand': risks['cand']}},
                                          spec, draw_ids=draw_ids)
    return {'abstain': False, 'p_value': p_value, 'certified': bool(certificate['candidates']['cand']['certified']),
            'lower_bound': certificate['candidates']['cand']['improvement_lower_bound'],
            'point_ratio': float(risks['base'].mean() / risks['cand'].mean()),
            'risks': {name: values.tolist() for name, values in risks.items()}}


def world_ground_weak(rng, ncfg, ground_amp=0.0, excited_amp=1.0, noise=0.02):
    """Two-state world where the excited state dominates the correlator."""
    T = PROTOCOL['geometry'][3]
    t = np.arange(T)
    profile = (ground_amp * (np.exp(-E0 * t) + np.exp(-E0 * (T - t)))
               + excited_amp * (np.exp(-E1 * t) + np.exp(-E1 * (T - t))))
    signal = np.ones((len(AMPS), 1)) * profile[None, :]
    cfg = rng.normal(size=(ncfg, len(AMPS), 1)) * 0.01
    eps = rng.normal(size=(ncfg, len(AMPS), T))
    for k in range(1, T):
        eps[:, :, k] = 0.5 * eps[:, :, k - 1] + np.sqrt(1 - 0.5 ** 2) * rng.normal(size=(ncfg, len(AMPS)))
    return signal[None] * (1 + cfg + noise * eps)


def run_wrong_state(rng, n_dataset=8, outer=40, inner=16):
    """Gate passes on the excited state; check whether a null candidate certifies."""
    gate_pass = wrong_mass = certified = abstain = 0
    masses = []
    for _ in range(n_dataset):
        dev = blocks(world_ground_weak(rng, 24 * PROTOCOL['block_length']), PROTOCOL['block_length'])
        val = blocks(world_ground_weak(rng, 12 * PROTOCOL['block_length']), PROTOCOL['block_length'])
        ref, mass, p_value = gate(dev)
        masses.append(float(mass))
        if p_value < 0.05:
            abstain += 1
            continue
        gate_pass += 1
        if abs(mass - E0) > 0.1:
            wrong_mass += 1
        weights = np.eye(len(AMPS))[1]
        draws = np.zeros(outer)
        for b in range(outer):
            dev_idx = rng.integers(0, len(dev), len(dev))
            ref_b = make_reference(dev[dev_idx], PROTOCOL,
                                   PROTOCOL['reference_windows'][PROTOCOL['reference_nominal_window']], 'single')
            fit_b, boot = ref_b.bootstrap(rng.integers(0, len(dev), (inner, len(dev))))
            draws[b] = risk(val[rng.integers(0, len(val), len(val))], weights, fit_b['mass'], boot,
                            rng.integers(0, len(val), (inner, len(val))), PROTOCOL)['risk']
        # identical baseline/candidate: a certificate would be a false improvement
        spec = {'schema': 'pion-score-v1', 'version': 'c2-wrong-state', 'estimand': 'conditional',
                'baseline_id': 'local', 'accepted_models': ['single'], 'candidate_ids': ['cand'],
                'analysis_version': 'c2-wrong-state', 'alpha': 0.05,
                'comparison_family': {'candidates': 1, 'models': 1},
                'min_baseline_risk': 1e-9, 'invalid_tolerance': 0.05, 'min_improvement': 1.0,
                'risk_floor': 0.0, 'require_draw_ids': True,
                'draw_ids_hash': hashlib.sha256(np.arange(outer).astype(np.int64).tobytes()).hexdigest()}
        certificate = improvement_certificate({'single': {'local': draws, 'cand': draws}},
                                              spec, draw_ids=np.arange(outer))
        certified += bool(certificate['candidates']['cand']['certified'])
    return {'n_dataset': n_dataset, 'abstain': abstain, 'gate_pass': gate_pass,
            'wrong_mass_gt_0.1': wrong_mass, 'certified': certified,
            'masses': masses,
            'note': ('ground amplitude 0, excited 1.0: the data are a pure excited state, the '
                     'single-cosh gate passes at E1, and only an independent-mass guard can '
                     'flag the wrong state')}

def main(n_dataset=8, blocks_list=(6, 12, 24), seed=20261120):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    flat = np.zeros(7)
    same = flat.copy(); same[1] = same[2] = 0.5
    better = flat.copy(); better[1] = 0.7; better[2] = 0.2
    miss = flat.copy(); miss[[0, 3, 4]] = 0.5; miss[[1, 2]] = 0.5
    scenarios = {
        'no_improvement': (same, same),
        'known_improvement': (better, better),
        'misspecified_reference': (miss, miss),
    }
    rows = {}
    for name, (base_c, cand_c) in scenarios.items():
        for blocks in blocks_list:
            results = []
            for _ in range(n_dataset):
                results.append(run_simulation(rng, base_c, cand_c, blocks_dev=blocks))
            valid = [r for r in results if not r['abstain']]
            certified = sum(r['certified'] for r in valid)
            key = f'{name}/blocks_{blocks}'
            rows[key] = {'n_dataset': n_dataset, 'blocks': blocks, 'abstain': n_dataset - len(valid),
                         'valid': len(valid), 'certified': certified,
                         'unconditional_certified_rate': certified / n_dataset,
                         'conditional_certified_rate': certified / len(valid) if valid else None,
                         'median_lower_bound': float(np.median([r['lower_bound'] for r in valid])) if valid else None,
                         'details': results}
            print(key, {k: rows[key][k] for k in ('n_dataset', 'abstain', 'valid', 'certified',
                                                  'unconditional_certified_rate')})
    family = run_family(rng)
    rows['family_J2_K2'] = family
    wrong = run_wrong_state(rng)
    rows['wrong_excited_state'] = wrong
    print('family', family)
    (OUT / 'certificate_calibration.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('wrote', OUT)




def run_family(rng, n_dataset=8, outer=40, inner=16, alpha=0.05):
    """J=2 candidates (one null, one better) and K=2 models (single, twostate)."""
    from .reference import make_reference

    contamination = np.zeros(7)
    contamination[[0, 3, 4]] = 0.0          # reference channels stay single-cosh
    contamination[1] = contamination[2] = 0.5   # baseline and null candidate
    contamination[5] = 0.2                  # the better candidate
    certified_null = certified_better = certified_any = abstain = 0
    for _ in range(n_dataset):
        dev = blocks(world(rng, 24 * PROTOCOL['block_length'], contamination), PROTOCOL['block_length'])
        val = blocks(world(rng, 12 * PROTOCOL['block_length'], contamination), PROTOCOL['block_length'])
        ref, mass, p_value = gate(dev)
        if p_value < 0.05:
            abstain += 1
            continue
        weights = {'base': np.eye(7)[1], 'null': np.eye(7)[2], 'better': np.eye(7)[5]}
        risks = {m: {name: np.zeros(outer) for name in weights} for m in ('single', 'twostate')}
        for b in range(outer):
            dev_idx = rng.integers(0, len(dev), len(dev))
            val_idx = rng.integers(0, len(val), len(val))
            inner_idx = rng.integers(0, len(val), (inner, len(val)))
            for m in ('single', 'twostate'):
                ref_b = make_reference(dev[dev_idx], PROTOCOL,
                                       PROTOCOL['reference_windows'][PROTOCOL['reference_nominal_window']], m)
                fit_b, masses = ref_b.bootstrap(rng.integers(0, len(dev), (inner, len(dev))))
                for name, w in weights.items():
                    risks[m][name][b] = risk(val[val_idx], w, fit_b['mass'], masses, inner_idx,
                                             PROTOCOL, require_valid=False)['risk']
        draw_ids = np.arange(outer)
        spec = {'schema': 'pion-score-v1', 'version': 'c2-family', 'estimand': 'conditional',
                'baseline_id': 'base', 'accepted_models': ['single', 'twostate'],
                'candidate_ids': ['null', 'better'], 'analysis_version': 'c2-family',
                'alpha': alpha, 'comparison_family': {'candidates': 2, 'models': 2},
                'min_baseline_risk': 1e-9, 'invalid_tolerance': 0.05, 'min_improvement': 1.0,
                'risk_floor': 0.0, 'require_draw_ids': True,
                'draw_ids_hash': hashlib.sha256(np.asarray(draw_ids, dtype=np.int64).tobytes()).hexdigest()}
        certificate = improvement_certificate(risks, spec, draw_ids=draw_ids)
        null_ok = bool(certificate['candidates']['null']['certified'])
        better_ok = bool(certificate['candidates']['better']['certified'])
        certified_null += null_ok
        certified_better += better_ok
        certified_any += (null_ok or better_ok)
    return {'n_dataset': n_dataset, 'outer': outer, 'inner': inner, 'alpha': alpha,
            'abstain': abstain, 'valid': n_dataset - abstain,
            'certified_null': certified_null, 'certified_better': certified_better,
            'family_false_certification': certified_null / (n_dataset - abstain) if n_dataset > abstain else None,
            'power': certified_better / (n_dataset - abstain) if n_dataset > abstain else None}


if __name__ == '__main__':
    main()
