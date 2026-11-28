"""Isolated CPU trainable-W parameterization; no production physics objective.

Spatial axes are (z,y,x), link directions are (x,y,z). Inputs are real
directional_features with shape (N,3,z,y,x,2). W and v are shared over directions
and configurations; each feature is centered over space independently for each
configuration/direction. The coefficient MSE below is not pion risk.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .features import directional_features
from .kernels import apply_kernel, correlator


def _features(z):
    z = np.asarray(z)
    if (z.ndim != 6 or z.shape[1] != 3 or z.shape[-1] != 2
            or any(size == 0 for size in z.shape) or np.iscomplexobj(z)
            or not np.all(np.isfinite(z))):
        raise ValueError('z must be finite real (N,3,z,y,x,2) directional features')
    return z.astype(float, copy=False)


def pack(W, v, c):
    W, v = np.asarray(W), np.asarray(v)
    if (W.ndim != 2 or W.shape[1] != 2 or not W.shape[0] or v.shape != (W.shape[0],)
            or np.iscomplexobj(W) or np.iscomplexobj(v) or np.iscomplexobj(c)):
        raise ValueError('W must be real (hidden,2), v real (hidden,), c real scalar')
    parameters = np.concatenate([W.ravel(), v, np.asarray(c).reshape(1)]).astype(float)
    if not np.all(np.isfinite(parameters)):
        raise ValueError('parameters must be finite')
    return parameters


def unpack(parameters, hidden):
    p = np.asarray(parameters)
    if (not isinstance(hidden, int) or hidden < 1 or p.shape != (3 * hidden + 1,)
            or np.iscomplexobj(p) or not np.all(np.isfinite(p))):
        raise ValueError('parameters must be finite real (3*hidden+1,)')
    return p[:2 * hidden].reshape(hidden, 2), p[2 * hidden:3 * hidden], float(p[-1])


def _forward(z, W, v, c):
    phi = np.tanh(np.einsum('ndzyxj,hj->ndzyxh', z, W))
    centered = phi - phi.mean(axis=(2, 3, 4), keepdims=True)
    return c + np.einsum('ndzyxh,h->ndzyx', centered, v), phi, centered


def coefficient_field(z, W, v, c):
    """Return real a[N,d,z,y,x]; center separately for each N,d."""
    parameters = pack(W, v, c)
    W, v, c = unpack(parameters, len(v))
    return _forward(_features(z), W, v, c)[0]


def apply(field, links, W, v, c):
    """Apply I + sum(a T + shifted-a T†), periodic in three spatial axes.

    This prototype has no Gaussian sandwich/positivity/physics normalization
    contract. Real coefficients ensure Hermiticity of the existing paired hop.
    """
    z = directional_features(np.asarray(links))[None]
    a = coefficient_field(z, W, v, c)[0]
    return apply_kernel(np.asarray(field), np.asarray(links), a)


def correlator_and_gradient(field, links, W, v, c):
    """Evaluate the real hopping correlator and its exact W/v/c gradient.

    This differentiates the actual paired-forward/backward ``apply_kernel``
    operator, including the per-direction spatial centering in
    ``coefficient_field``. It is a CPU algebra/implementation check only; it
    does not differentiate the ensemble risk, reference fit, or QUDA solve.
    ``field`` has shape (z,y,x,3,n), and ``links`` has shape
    (3,z,y,x,3,3).
    """
    field = np.asarray(field)
    links = np.asarray(links)
    if (field.ndim != 5 or field.shape[-2] != 3 or np.iscomplexobj(field) is False
            or not np.all(np.isfinite(field))):
        raise ValueError('field must be finite complex (z,y,x,3,n)')
    if (links.shape != (3, *field.shape[:3], 3, 3)
            or not np.iscomplexobj(links) or not np.all(np.isfinite(links))):
        raise ValueError('links must be finite complex (3,z,y,x,3,3)')
    z = directional_features(links)[None, ...]
    parameters = pack(W, v, c)
    W, v, c = unpack(parameters, len(v))
    prediction, phi, centered = _forward(_features(z), W, v, c)
    a = prediction[0]
    value = float(correlator(field, links, a))

    # The coefficient a_mu(x) multiplies both the forward hopping at x and
    # its adjoint backward hopping from x+mu. Their real derivatives agree.
    q = np.empty((3, *field.shape[:3]), dtype=float)
    for direction in range(3):
        neighbor = np.roll(field, -1, axis=2 - direction)
        hopped = links[direction] @ neighbor
        q[direction] = 2.0 * np.real(np.sum(field.conj() * hopped, axis=(-2, -1)))

    # Centering removes each direction's spatial mean. Its adjoint therefore
    # subtracts the corresponding spatial mean from the coefficient gradient.
    q_centered = q - q.mean(axis=(1, 2, 3), keepdims=True)
    delta = q_centered[None, ..., None] * v[None, None, None, None, None, :]
    delta *= 1.0 - phi ** 2
    grad_W = np.einsum('ndzyxh,ndzyxj->hj', delta, z)
    grad_v = np.einsum('dzyx,ndzyxh->h', q, centered)
    grad_c = float(q.sum())
    gradient = np.concatenate([grad_W.ravel(), grad_v, [grad_c]])
    return value, gradient


def loss_and_gradient(parameters, z, target, hidden):
    """Half mean squared coefficient error and analytic W/v/c gradient.

    dL/dphi uses e minus its spatial mean: omitting this projection would give
    the wrong derivative of the per-direction spatial centering operation.
    """
    z = _features(z)
    target = np.asarray(target)
    if (target.shape != z.shape[:-1] or np.iscomplexobj(target)
            or not np.all(np.isfinite(target))):
        raise ValueError('target must be a finite real coefficient field matching z')
    W, v, c = unpack(parameters, hidden)
    prediction, phi, centered = _forward(z, W, v, c)
    error = prediction - target
    derivative = error / error.size
    projected = derivative - derivative.mean(axis=(2, 3, 4), keepdims=True)
    delta = projected[..., None] * v * (1 - phi ** 2)
    grad_W = np.einsum('ndzyxh,ndzyxj->hj', delta, z)
    grad_v = np.einsum('ndzyx,ndzyxh->h', derivative, centered)
    gradient = np.concatenate([grad_W.ravel(), grad_v, [derivative.sum()]])
    return float(.5 * np.mean(error ** 2)), gradient


def fit(z, target, W, v, c, *, maxiter=400):
    """Joint CPU L-BFGS-B coefficient fit with raw diagnostic status/trace.

    maxiter/tolerances here are prototype controls, not production thresholds.
    No dev/test data enter this function and no solver gate is implied.
    """
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError('maxiter must be a positive integer')
    initial = pack(W, v, c)
    hidden = len(v)
    initial_loss, _ = loss_and_gradient(initial, z, target, hidden)
    trace, started = [], time.perf_counter()

    def objective(parameters):
        value, gradient = loss_and_gradient(parameters, z, target, hidden)
        trace.append({'call': len(trace) + 1, 'elapsed_seconds': time.perf_counter() - started,
                      'loss': value, 'gradient_norm': float(np.linalg.norm(gradient))})
        return value, gradient

    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-9})
    final_loss, gradient = loss_and_gradient(result.x, z, target, hidden)
    final_W, final_v, final_c = unpack(result.x, hidden)
    return {'classification': 'diagnostic_only', 'physical_score': None,
            'method': 'L-BFGS-B coefficient MSE', 'success': bool(result.success),
            'status': int(result.status), 'message': str(result.message),
            'nfev': int(result.nfev), 'nit': int(result.nit),
            'elapsed_seconds': time.perf_counter() - started,
            'initial_parameters': initial.tolist(), 'parameters': result.x.tolist(),
            'W': final_W.tolist(), 'v': final_v.tolist(), 'c': final_c,
            'initial_loss': initial_loss, 'final_loss': final_loss,
            'final_gradient_norm': float(np.linalg.norm(gradient)),
            'W_update_norm': float(np.linalg.norm(final_W - np.asarray(W))),
            'trace': trace}
