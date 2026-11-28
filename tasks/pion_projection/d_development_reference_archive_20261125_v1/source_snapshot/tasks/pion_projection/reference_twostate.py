"""Two-state common-mass reference: A_k f(t;m0) + B_k f(t;m1)."""
import hashlib
import json
import platform
import scipy
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .statistics import inverse_covariance, shape


def _receipt_safe(value):
    """Convert nested fit diagnostics to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _receipt_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_receipt_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return 'NaN'
        if np.isposinf(number):
            return 'Infinity'
        if np.isneginf(number):
            return '-Infinity'
        return number
    if isinstance(value, np.integer):
        return int(value)
    return value


def _validate_optimizer_controls(maxiter, ftol, boundary_tolerance, feasibility_tolerance):
    controls = np.asarray([maxiter, ftol, boundary_tolerance, feasibility_tolerance], dtype=float)
    if not np.isfinite(controls).all():
        raise ValueError('optimizer controls must be finite')
    invalid = (
        int(maxiter) <= 0 or int(maxiter) != maxiter or ftol <= 0
        or boundary_tolerance < 0 or feasibility_tolerance < 0
    )
    if invalid:
        raise ValueError('invalid optimizer controls')


class TwoStateFit:
    def __init__(self, data_blocks, config, window):
        self.p = config
        self.window = np.asarray(window)
        self.basis = config['reference_basis']
        self.samples = data_blocks[:, self.basis][:, :, self.window]
        self.scales = self.samples.mean(0)[:, 0]
        if np.any(self.scales == 0):
            raise ValueError('Zero reference channel')
        self.samples = (self.samples / self.scales[None, :, None]).reshape(len(data_blocks), -1)
        self.inv = inverse_covariance(self.samples, config['covariance_shrinkage'])
        self.nchan = len(self.basis)
        self.ntime = len(self.window)
        # `inv` is symmetric positive definite after inverse_covariance's
        # eigenvalue floor.  Whitening lets us solve for linear amplitudes with
        # SVD instead of forming the more poorly conditioned normal equations.
        precision = 0.5 * (self.inv + self.inv.T)
        try:
            self.whitener = np.linalg.cholesky(precision).T
        except np.linalg.LinAlgError as exc:
            raise ValueError('Reference precision matrix is not positive definite') from exc

    def design(self, m0, m1):
        T = self.p['geometry'][3]
        f0 = shape(m0, self.window, T)
        f1 = shape(m1, self.window, T)
        return np.kron(np.eye(self.nchan), np.stack([f0, f1], axis=1))

    def linear_fit(self, m0, m1, mean):
        """Fit linear amplitudes at fixed masses using whitened least squares.

        Equal masses are accepted intentionally: at ``m1 == m0`` the two-state
        design is the nested one-state boundary and its amplitude design is
        rank deficient.  That point is reported as such, never as a resolved
        two-state fit.
        """
        m0, m1 = float(m0), float(m1)
        mean = np.asarray(mean, dtype=float)
        if not np.isfinite([m0, m1]).all() or m1 < m0:
            raise ValueError('Expected finite ordered masses m0 <= m1')
        if mean.shape != (self.samples.shape[1],) or not np.isfinite(mean).all():
            raise ValueError('mean must be finite and match the fitted data dimension')
        X = self.design(m0, m1)
        weighted_design = self.whitener @ X
        weighted_mean = self.whitener @ mean
        amp, _, rank, singular_values = np.linalg.lstsq(
            weighted_design, weighted_mean, rcond=None,
        )
        residual = weighted_mean - weighted_design @ amp
        cost = float(residual @ residual)
        nparam = weighted_design.shape[1]
        if rank == nparam and len(singular_values) and singular_values[-1] > 0:
            design_condition = float(singular_values[0] / singular_values[-1])
            # Preserve the historical `condition` meaning: cond(X^T W X).
            condition = design_condition * design_condition
        else:
            design_condition = float('inf')
            condition = float('inf')
        gap_tol = np.finfo(float).eps * max(1.0, abs(m0), abs(m1)) * 8
        if abs(m1 - m0) <= gap_tol:
            status = 'nested_m1_boundary'
        elif rank < nparam:
            status = 'rank_deficient'
        else:
            status = 'full_rank'
        return {
            'cost': cost,
            'amplitudes': amp,
            'rank': int(rank),
            'n_amplitudes': int(nparam),
            'design_condition': design_condition,
            'condition': condition,
            'fit_status': status,
        }

    def _cost(self, m0, m1, mean):
        return self.linear_fit(m0, m1, mean)['cost']

    def _search(self, mean):
        lo, hi = self.p['mass_bounds']
        best = (np.inf, None, None)
        for m0 in np.arange(max(lo + 0.2, 0.7), min(hi - 0.3, 1.4), 0.02):
            for gap in (0.2, 0.4, 0.6, 0.8, 1.0):
                m1 = m0 + gap
                if m1 >= hi:
                    continue
                c = self._cost(m0, m1, mean)
                if c < best[0]:
                    best = (c, m0, m1)
        cost, m0, m1 = best
        for step in (0.01, 0.003):
            improved = True
            while improved:
                improved = False
                for d0, d1 in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    c0, c1 = m0 + d0, m1 + d1
                    if not (lo < c0 < c1 < hi):
                        continue
                    c = self._cost(c0, c1, mean)
                    if c < cost - 1e-12:
                        cost, m0, m1 = c, c0, c1
                        improved = True
        return cost, m0, m1

    def fit_full(self, mean):
        cost, m0, m1 = self._search(mean)
        linear = self.linear_fit(m0, m1, mean)
        lo, hi = self.p['mass_bounds']
        return {'mass': float(m0), 'mass2': float(m1), 'gap': float(m1 - m0),
                'amplitudes': linear['amplitudes'].tolist(), 'cost': float(cost),
                'condition': linear['condition'],
                'design_condition': linear['design_condition'],
                'amplitude_rank': linear['rank'],
                'amplitude_count': linear['n_amplitudes'],
                'fit_status': linear['fit_status'],
                'at_boundary': bool(min(m0 - lo, hi - m1) < 1e-3)}

    def fit_full_multistart(self, mean, starts=None, *, maxiter=500, ftol=1e-10,
                            boundary_tolerance=1e-5, feasibility_tolerance=1e-8):
        """Run and retain a deterministic, constrained SLSQP mass-search family.

        The linear amplitudes are eliminated by :meth:`linear_fit` at every
        objective call.  Every optimizer endpoint is returned, including
        finite nonconverged and nested ``gap=0`` endpoints.  The selected
        lowest finite feasible endpoint is only a best-found diagnostic; this
        routine does not certify a global minimum or apply a physics gate.
        Starts are ``(mass0, gap)`` pairs and the constraints enforce
        ``lo <= mass0``, ``gap >= 0``, and ``mass0 + gap <= hi``.
        """
        lo, hi = map(float, self.p['mass_bounds'])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError('invalid mass bounds')
        _validate_optimizer_controls(maxiter, ftol, boundary_tolerance, feasibility_tolerance)
        mean = np.asarray(mean, dtype=float)
        if mean.shape != (self.samples.shape[1],) or not np.isfinite(mean).all():
            raise ValueError('mean must be finite and match the fitted data dimension')
        if starts is None:
            width = hi - lo
            mass0s = np.linspace(lo + 0.03 * width, hi - 0.60 * width, 5)
            gaps = np.linspace(0.0, 0.90 * width, 7)
            starts = [(m0, gap) for m0 in mass0s for gap in gaps
                      if m0 + gap <= hi - 1e-10]
        starts = np.asarray(starts, dtype=float)
        if starts.ndim != 2 or starts.shape[1] != 2 or len(starts) == 0:
            raise ValueError('starts must be a nonempty (n, 2) array of (mass0, gap)')
        if not np.isfinite(starts).all():
            raise ValueError('starts must be finite')
        for mass0, gap in starts:
            if mass0 < lo or gap < 0 or mass0 + gap > hi:
                raise ValueError(f'infeasible starting point {(mass0, gap)}')

        def objective(point):
            mass0, gap = map(float, point)
            return self._cost(mass0, mass0 + gap, mean)

        rows = []
        for start_id, (start_mass0, start_gap) in enumerate(starts):
            optimizer_exception = None
            try:
                result = minimize(
                    objective,
                    np.asarray([start_mass0, start_gap], dtype=float),
                    method='SLSQP',
                    bounds=[(lo, hi), (0.0, hi - lo)],
                    constraints=[{'type': 'ineq', 'fun': lambda point: hi - point[0] - point[1]}],
                    options={'maxiter': int(maxiter), 'ftol': float(ftol)},
                )
            except Exception as exc:
                optimizer_exception = f'{type(exc).__name__}: {exc}'
                result = type('OptimizerFailure', (), {
                    'x': np.asarray([start_mass0, start_gap], dtype=float),
                    'success': False, 'status': -1, 'message': optimizer_exception,
                    'nfev': 0, 'nit': 0,
                })()
            mass0, gap = map(float, result.x)
            mass1 = mass0 + gap
            violation = float(max(0.0, lo - mass0, -gap, mass1 - hi))
            try:
                linear = self.linear_fit(mass0, mass1, mean)
                finite = bool(np.isfinite(linear['cost']) and np.isfinite(linear['amplitudes']).all())
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                linear = {'cost': float('inf'), 'amplitudes': np.asarray([]), 'rank': 0,
                          'n_amplitudes': 2 * self.nchan, 'condition': float('inf'),
                          'design_condition': float('inf'), 'fit_status': 'invalid_endpoint'}
                finite = False
            feasible = bool(violation <= feasibility_tolerance)
            if not finite or not feasible:
                endpoint_status = 'invalid_endpoint'
            elif linear['fit_status'] == 'nested_m1_boundary':
                endpoint_status = 'nested_m1_boundary'
            elif gap <= boundary_tolerance:
                endpoint_status = 'near_nested_m1_boundary'
            elif linear['rank'] < linear['n_amplitudes']:
                endpoint_status = 'rank_deficient'
            elif not bool(result.success):
                endpoint_status = 'finite_nonconverged_endpoint'
            elif abs(mass0 - lo) <= boundary_tolerance or abs(mass1 - hi) <= boundary_tolerance:
                endpoint_status = 'converged_mass_boundary'
            else:
                endpoint_status = 'converged_full_rank'
            rows.append({
                'start_id': int(start_id),
                'start_mass0': float(start_mass0),
                'start_gap': float(start_gap),
                'mass0': mass0,
                'mass1': mass1,
                'gap': gap,
                'cost': float(linear['cost']),
                'finite_endpoint': finite,
                'feasible_endpoint': feasible,
                'solver_success': bool(result.success),
                'solver_status': int(result.status),
                'solver_message': str(result.message),
                'optimizer_exception': optimizer_exception,
                'nfev': int(getattr(result, 'nfev', 0)),
                'nit': int(getattr(result, 'nit', 0)),
                'constraint_violation': violation,
                'endpoint_status': endpoint_status,
                'amplitude_rank': int(linear['rank']),
                'amplitude_count': int(linear['n_amplitudes']),
                'design_condition': float(linear['design_condition']),
                'condition': float(linear['condition']),
                'amplitudes': np.asarray(linear['amplitudes'], dtype=float).tolist(),
            })

        valid = [row for row in rows if row['finite_endpoint'] and row['feasible_endpoint']]
        best = min(valid, key=lambda row: row['cost']) if valid else None
        if best is None:
            status = 'no_finite_feasible_endpoint'
        elif best['endpoint_status'] == 'nested_m1_boundary':
            status = ('best_found_on_nested_m1_boundary' if best['solver_success']
                      else 'best_found_at_nonconverged_nested_m1_boundary')
        elif best['endpoint_status'] == 'near_nested_m1_boundary':
            status = 'best_found_near_nested_m1_boundary'
        elif not best['solver_success']:
            status = 'best_found_at_nonconverged_endpoint'
        else:
            status = 'best_found_full_rank_or_mass_boundary'
        return {
            'classification': 'diagnostic_only',
            'status': status,
            'best_found': best,
            'rows': rows,
        }

    def iter_bootstrap_multistart(self, indices, starts, *, maxiter=500, ftol=1e-10,
                                  boundary_tolerance=1e-5, feasibility_tolerance=1e-8,
                                  start_draw=0):
        """Yield a manifest and every planned fixed-covariance bootstrap fit.

        `indices` is the complete integer block-index matrix; it is hashed
        before any draw is fitted. The caller owns the physical configuration
        IDs and must bind them in the outer run receipt. No draw is silently
        dropped, and this iterator does not make a reference-gate decision.
        """
        indices = np.asarray(indices)
        if (indices.ndim != 2 or indices.shape[0] == 0
                or indices.shape[1] != len(self.samples)
                or indices.dtype.kind not in 'iu'):
            raise ValueError('indices must be a nonempty integer (n_draws, n_blocks) matrix')
        if np.any(indices < 0) or np.any(indices >= len(self.samples)):
            raise ValueError('bootstrap block index is out of range')
        _validate_optimizer_controls(maxiter, ftol, boundary_tolerance, feasibility_tolerance)
        if (isinstance(start_draw, (bool, np.bool_))
                or not isinstance(start_draw, (int, np.integer))
                or start_draw < 0 or start_draw > indices.shape[0]):
            raise ValueError('start_draw must be an integer within the planned draw range')
        indices = np.ascontiguousarray(indices, dtype='<i8')
        starts = np.asarray(starts, dtype=float)
        lo, hi = map(float, self.p['mass_bounds'])
        if starts.ndim != 2 or starts.shape[1] != 2 or len(starts) == 0:
            raise ValueError('starts must be a nonempty (n, 2) array of (mass0, gap)')
        if not np.isfinite(starts).all():
            raise ValueError('starts must be finite')
        if any(mass0 < lo or gap < 0 or mass0 + gap > hi for mass0, gap in starts):
            raise ValueError('bootstrap start is outside the mass domain')
        starts = np.ascontiguousarray(starts, dtype='<f8')
        index_payload = np.asarray(indices.shape, dtype='<i8').tobytes() + indices.tobytes(order='C')
        metric = np.ascontiguousarray(self.inv, dtype='<f8')
        fit_configuration = {
            'geometry': np.asarray(self.p['geometry']).tolist(),
            'mass_bounds': np.asarray(self.p['mass_bounds']).tolist(),
            'reference_basis': np.asarray(self.basis).tolist(),
            'window': self.window.tolist(),
            'covariance_shrinkage': _receipt_safe(self.p.get('covariance_shrinkage')),
        }
        fit_configuration_bytes = json.dumps(
            fit_configuration, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ).encode('utf-8')
        with open(__file__, 'rb') as source_file:
            executor_source_sha256 = hashlib.sha256(source_file.read()).hexdigest()
        yield {
            'record_type': 'run_manifest',
            'schema': 'pion-twostate-multistart-bootstrap-v1',
            'classification': 'diagnostic_only',
            'planned_draws': int(indices.shape[0]),
            'blocks_per_draw': int(indices.shape[1]),
            'starts_per_draw': int(len(starts)),
            'index_matrix_sha256': hashlib.sha256(index_payload).hexdigest(),
            'starts_sha256': hashlib.sha256(starts.tobytes(order='C')).hexdigest(),
            'fixed_precision_sha256': hashlib.sha256(metric.tobytes(order='C')).hexdigest(),
            'fit_configuration_sha256': hashlib.sha256(fit_configuration_bytes).hexdigest(),
            'fit_configuration': fit_configuration,
            'executor_source_sha256': executor_source_sha256,
            'environment': {
                'python': sys.version,
                'platform': platform.platform(),
                'numpy': np.__version__,
                'scipy': scipy.__version__,
            },
            'optimizer_controls': {
                'method': 'SLSQP',
                'maxiter': int(maxiter),
                'ftol': float(ftol),
                'boundary_tolerance': float(boundary_tolerance),
                'feasibility_tolerance': float(feasibility_tolerance),
            },
            'covariance_policy': 'fixed precision estimated from the original block sample',
            'index_matrix': indices.tolist(),
            'starts': starts.tolist(),
        }
        for draw_id in range(int(start_draw), len(indices)):
            block_indices = indices[draw_id]
            mean = self.samples[block_indices].mean(axis=0)
            fit = self.fit_full_multistart(
                mean, starts, maxiter=maxiter, ftol=ftol,
                boundary_tolerance=boundary_tolerance,
                feasibility_tolerance=feasibility_tolerance,
            )
            yield {
                'record_type': 'bootstrap_draw',
                'draw_id': int(draw_id),
                'resampled_block_indices': block_indices.tolist(),
                'fit': _receipt_safe(fit),
            }

    def fit(self, mean):
        cost, m0, m1 = self._search(mean)
        return float(m0), float(cost)

    def bootstrap(self, index):
        mean = self.samples.mean(0)
        full = self.fit_full(mean)
        masses = np.array([self.fit(row)[0] for row in self.samples[index].mean(axis=1)])
        dof = self.nchan * self.ntime - 2 - 2 * self.nchan
        return {'mass': full['mass'], 'mass2': full['mass2'], 'gap': full['gap'],
                'amplitudes': full['amplitudes'], 'condition': full['condition'],
                'design_condition': full['design_condition'],
                'amplitude_rank': full['amplitude_rank'],
                'amplitude_count': full['amplitude_count'],
                'fit_status': full['fit_status'],
                'at_boundary': full['at_boundary'], 'se': float(masses.std(ddof=1)),
                'chi2': full['cost'], 'dof': dof,
                'p_approx': float(chi2.sf(full['cost'], dof)),
                'window': self.window.tolist()}, masses
