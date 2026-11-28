"""Fresh synthetic coefficient-fit demonstration, isolated from production.

Run with --out pointing to a new task-local directory. No archive/cache access.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from . import trainable_w as model
from .features import dagger, directional_features, shift


def su3(rng, shape):
    raw = rng.normal(size=(*shape, 3, 3)) + 1j * rng.normal(size=(*shape, 3, 3))
    q, _ = np.linalg.qr(raw)
    q[..., :, 0] /= np.linalg.det(q)[..., None]
    return q


def transform(array, rotation):
    """Independent coordinate/link-anchor transform for cubic diagnostics."""
    n = array.shape[1]
    zz, yy, xx = np.indices((n, n, n))
    coordinates = np.stack([xx, yy, zz], axis=-1)
    old = np.einsum('ij,...j->...i', rotation.T, coordinates).astype(int) % n
    index = (old[..., 2], old[..., 1], old[..., 0])
    if array.ndim == 5:
        return array[index]
    links = []
    for direction in range(3):
        source = int(np.argmax(np.abs(rotation[direction])))
        sign = int(rotation[direction, source])
        if sign > 0:
            value = array[source]
        elif array.ndim == 4:  # a is an unoriented-edge scalar: shift, no sign or dagger
            value = shift(array[source], -1, source)
        else:
            value = dagger(shift(array[source], -1, source))
        links.append(value[index])
    return np.stack(links)


def symmetry_checks(links, W, v, c, seed=2026112105):
    rng = np.random.default_rng(seed)
    shape = links.shape[1:4]
    f = rng.normal(size=(*shape, 3, 2)) + 1j * rng.normal(size=(*shape, 3, 2))
    g = rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)
    gauge = su3(rng, shape)
    transformed_links = np.stack([gauge @ links[d] @ dagger(shift(gauge, 1, d)) for d in range(3)])
    transformed_field = gauge @ f
    z = directional_features(links)[None]
    a = model.coefficient_field(z, W, v, c)[0]
    ag = model.coefficient_field(directional_features(transformed_links)[None], W, v, c)[0]
    applied = model.apply(f, links, W, v, c)
    gauge_error = np.linalg.norm(model.apply(transformed_field, transformed_links, W, v, c) - gauge @ applied)
    gauge_error /= max(np.linalg.norm(applied), 1e-30)
    lhs = np.vdot(g, applied)
    rhs = np.vdot(model.apply(g, links, W, v, c), f)
    hermitian = abs(lhs - rhs) / max(np.linalg.norm(g) * np.linalg.norm(f), 1e-30)
    rows = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = np.zeros((3, 3), dtype=int)
            for axis, source in enumerate(permutation):
                rotation[axis, source] = signs[axis]
            expected = transform(applied, rotation)
            actual = model.apply(transform(f, rotation), transform(links, rotation), W, v, c)
            error = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)
            rotated_a = model.coefficient_field(directional_features(transform(links, rotation))[None], W, v, c)[0]
            rows.append({'permutation': list(permutation), 'signs': list(signs),
                         'determinant': int(round(np.linalg.det(rotation))), 'relative_error': float(error),
                         'coefficient_max_abs': float(np.max(np.abs(rotated_a - transform(a, rotation))))})
    return {'feature_gauge_max_abs': float(np.max(np.abs(directional_features(transformed_links) - z[0]))),
            'coefficient_gauge_max_abs': float(np.max(np.abs(ag - a))),
            'coefficient_spatial_mean_minus_c_max_abs': float(np.max(np.abs(a.mean(axis=(1, 2, 3)) - c))),
            'kernel_gauge_relative': float(gauge_error), 'hermitian_relative': float(hermitian),
            'cubic_cases': rows, 'cubic_max_relative': max(row['relative_error'] for row in rows),
            'cubic_coefficient_max_abs': max(row['coefficient_max_abs'] for row in rows)}


def gradient_check(parameters, z, target, hidden, step=1e-6):
    value, analytic = model.loss_and_gradient(parameters, z, target, hidden)
    rows = []
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        numeric = (model.loss_and_gradient(plus, z, target, hidden)[0]
                   - model.loss_and_gradient(minus, z, target, hidden)[0]) / (2 * step)
        rows.append({'parameter': index, 'analytic': float(analytic[index]), 'finite_difference': float(numeric),
                     'absolute_error': float(abs(analytic[index] - numeric))})
    return {'loss': value, 'step': step, 'rows': rows,
            'max_absolute_error': max(row['absolute_error'] for row in rows)}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = {'classification': 'diagnostic_only', 'physical_score': None,
            'scope': 'noiseless synthetic coefficient fit, not gauge-ensemble physics risk; no SLSQP comparison',
            'shape_zyx': [3, 3, 3], 'hidden': 2, 'fit_links': 4, 'heldout_links': 3,
            'teacher_seed': 2026112101, 'fit_seed': 2026112102, 'heldout_seed': 2026112103,
            'initial_seed': 2026112104, 'maxiter': 400,
            'source_hashes': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ('trainable_w.py', 'trainable_w_diagnostic.py', 'features.py', 'kernels.py')}}
    (output / 'protocol.json').write_text(json.dumps(spec, indent=2) + '\n')
    teacher_rng = np.random.default_rng(spec['teacher_seed'])
    teacher_W, teacher_v, teacher_c = teacher_rng.normal(size=(2, 2)) * 2, teacher_rng.normal(size=2) * .3, .05
    train_rng, test_rng = np.random.default_rng(spec['fit_seed']), np.random.default_rng(spec['heldout_seed'])
    fit_links = [su3(train_rng, (3, 3, 3, 3)) for _ in range(spec['fit_links'])]
    heldout_links = [su3(test_rng, (3, 3, 3, 3)) for _ in range(spec['heldout_links'])]
    z = np.stack([directional_features(u) for u in fit_links])
    ztest = np.stack([directional_features(u) for u in heldout_links])
    target = model.coefficient_field(z, teacher_W, teacher_v, teacher_c)
    target_test = model.coefficient_field(ztest, teacher_W, teacher_v, teacher_c)
    initial_rng = np.random.default_rng(spec['initial_seed'])
    W0, v0, c0 = initial_rng.normal(size=(2, 2)) * .4, initial_rng.normal(size=2) * .2, -.02
    initial = model.pack(W0, v0, c0)
    gradient = gradient_check(initial, z, target, 2)
    result = model.fit(z, target, W0, v0, c0, maxiter=spec['maxiter'])
    initial_test = model.loss_and_gradient(initial, ztest, target_test, 2)[0]
    final_test = model.loss_and_gradient(np.array(result['parameters']), ztest, target_test, 2)[0]
    symmetry = symmetry_checks(heldout_links[0], np.array(result['W']), np.array(result['v']), result['c'])
    per_link = []
    for population, inputs, truth in (('fit', z, target), ('heldout', ztest, target_test)):
        for index in range(len(inputs)):
            before = model.loss_and_gradient(initial, inputs[index:index+1], truth[index:index+1], 2)[0]
            after = model.loss_and_gradient(np.array(result['parameters']), inputs[index:index+1], truth[index:index+1], 2)[0]
            per_link.append({'population': population, 'index': index, 'initial_half_mse': before,
                             'final_half_mse': after, 'finite': bool(np.isfinite(after))})
    raw = {'spec': spec, 'teacher_parameters': model.pack(teacher_W, teacher_v, teacher_c).tolist(),
           'fit': result, 'gradient_check': gradient, 'symmetry': symmetry, 'per_link': per_link,
           'heldout_initial_loss': initial_test, 'heldout_final_loss': final_test,
           'denominators': {'planned_fits': 1, 'returned_fits': 1, 'solver_non_success': int(not result['success']),
                            'fit_links': len(fit_links), 'heldout_links': len(heldout_links),
                            'gradient_components': len(initial), 'cubic_cases': len(symmetry['cubic_cases']),
                            'nonfinite_link_metrics': sum(not row['finite'] for row in per_link)}}
    (output / 'raw.json').write_text(json.dumps(raw, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(output / 'synthetic_inputs.npz', fit_features=z, fit_target=target,
                        heldout_features=ztest, heldout_target=target_test)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    trace = result['trace']
    axes[0].semilogy([r['call'] for r in trace], [r['loss'] for r in trace], label='raw fit loss')
    axes[0].semilogy([r['call'] for r in trace], np.minimum.accumulate([r['loss'] for r in trace]), label='best observed')
    axes[0].set(xlabel='objective+gradient calls', ylabel='half coefficient MSE', title='Synthetic fit trace')
    axes[0].legend()
    axes[1].bar(['fit initial', 'fit final', 'heldout initial', 'heldout final'],
                [result['initial_loss'], result['final_loss'], initial_test, final_test])
    axes[1].set(yscale='log', ylabel='half coefficient MSE', title='Independent links; no physics score')
    axes[1].tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(output / 'coefficient_fit.png', dpi=160)
    plt.close(fig)
    rows = '\n'.join(f"|{r['population']}|{r['index']}|{r['initial_half_mse']:.6g}|{r['final_half_mse']:.6g}|{r['finite']}|" for r in per_link)
    report = f'''# Trainable-W CPU parameterization prototype

Status: **CPU parameterization prototype completed / production integration and physics comparison still NOT_RUN**.
Classification: `diagnostic_only`; physical_score=null. This is one new deterministic coefficient-fit fixture, not repeated initializer benchmarking.

The model uses the actual link-centered `directional_features`, real shared W/v/c, spatial mean centering per configuration/direction,
and the existing paired-forward/backward `apply_kernel`. Spatial storage is (z,y,x), directions (x,y,z), fields (...,color,probe),
color=3. Spatial boundaries are periodic. No temporal/gamma/propagator convention or Gaussian normalization is tested here.
Centering uses the whole spatial slice: the map is time-local but not strictly spatial-local. a_mu is an unoriented-edge scalar,
not an ordinary polar vector; reversing a link requires shifting its anchor, without multiplying a by a reflection sign.

Input flow: independent teacher seed → noiseless coefficient labels on 4 fit links → joint analytic-gradient L-BFGS-B → 3 independently
generated heldout links. Heldout data do not enter optimization/selection. Teacher is in the same model family by construction;
low error is implementation recovery, not discovery of a physical smear. Hidden permutations and (W_h,v_h)→(-W_h,-v_h)
are exact parameter non-identifiabilities; parameter recovery is not claimed. Initialization draws both W and v independently and nonzero:
W=v=0 makes both nonconstant gradients vanish. Cold/homogeneous links have zero centered features and cannot test W learning.

![Coefficient fitting diagnostic](coefficient_fit.png)

|metric|observed|
|---|---|
|fit initial/final half MSE|{result['initial_loss']:.8g} / {result['final_loss']:.8g}|
|heldout initial/final half MSE|{initial_test:.8g} / {final_test:.8g}|
|W update norm|{result['W_update_norm']:.8g}|
|solver success/status/message|{result['success']} / {result['status']} / {result['message']}|
|objective+gradient calls/iterations|{result['nfev']} / {result['nit']}|
|elapsed CPU wall seconds|{result['elapsed_seconds']:.6g}|
|gradient finite-difference max absolute error ({len(initial)} components)|{gradient['max_absolute_error']:.6g}|
|feature/coefficient gauge max abs|{symmetry['feature_gauge_max_abs']:.6g} / {symmetry['coefficient_gauge_max_abs']:.6g}|
|kernel gauge covariance relative error|{symmetry['kernel_gauge_relative']:.6g}|
|Hermiticity relative error|{symmetry['hermitian_relative']:.6g}|
|48 cubic/parity signed permutations maximum relative error|{symmetry['cubic_max_relative']:.6g}|
|48 cubic/parity coefficient anchor max abs error|{symmetry['cubic_coefficient_max_abs']:.6g}|

## Per-link denominators and failure state

Planned fits=1, returned fits=1, solver non-success={int(not result['success'])}; fit links=4, heldout links=3, excluded links=0,
nonfinite reported link metrics={raw['denominators']['nonfinite_link_metrics']}. Gradient components=7; cubic cases=48 (24 rotations plus 24 improper).
These are fixture coverage counts, not statistical failure probabilities. All per-call traces, final status, parameters,
gradient components and all 48 symmetry cases remain in raw.json; no case is silently excluded. Fatal exceptions propagate;
this diagnostic script does not promise a complete report on interruption.

|population|link index|initial half MSE|final half MSE|finite|
|---|---|---|---|---|
{rows}

## Evidence boundary

The objective is half mean squared error on synthetic coefficient labels. It is **not** gauge-ensemble/pion physics risk,
not a reference-gate test, not a SLSQP comparison, and not evidence of physical or cost superiority. Maxiter/optimizer tolerances
are fixture controls; they do not freeze any production TBD. No confidence interval or gauge sampling calibration is claimed
from one fit and three heldout links. Hermiticity does not imply positivity. No production source/sink, GPU/QUDA, shared holdout,
trainable-W production integration, Gaussian sandwich, gradients through propagator measurements, or blind physics comparison was run.

Protocol/source hashes are in protocol.json; raw parameters/traces/metrics in raw.json; features/targets in synthetic_inputs.npz.
Regenerate only into a fresh directory: `python3 -B -m tasks.optimize_gaussian.trainable_w_diagnostic --out <new-directory>`.
'''
    (output / 'README.md').write_text(report)
    return raw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.out)
