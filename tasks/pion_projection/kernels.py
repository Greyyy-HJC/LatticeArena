"""Time-slice submission API; NumPy or CuPy arrays, no propagator-dependent weights.

Links: (3,z,y,x,3,3), ordered x,y,z. Fields: (z,y,x,3,nrhs).
K = sum_n weights[n] H**powers[n], H=(1+alpha*spatial_hopping)/(1+6*alpha).
"""
import numpy as np


class DiffusionKernel:
    def __init__(self, spatial_links, powers, weights, alpha=0.2):
        self.xp = np
        if type(spatial_links).__module__.split('.')[0] == 'cupy':
            import cupy
            self.xp = cupy
        self.links = spatial_links.copy()
        self.powers = tuple(powers)
        self.weights = tuple(weights)
        self.alpha = float(alpha)
        if spatial_links.ndim != 6 or spatial_links.shape[0] != 3 or spatial_links.shape[-2:] != (3, 3):
            raise ValueError('Expected spatial links with shape (3,z,y,x,3,3)')
        if len(powers) != len(weights) or not powers:
            raise ValueError('One weight is required per power')
        if any(not isinstance(n, int) or n < 0 for n in powers) or len(set(powers)) != len(powers):
            raise ValueError('Powers must be distinct nonnegative integers')
        if not np.isfinite(alpha) or alpha < 0 or not np.all(np.isfinite(weights)):
            raise ValueError('Nonfinite coefficients or negative alpha')
        if not any(w != 0 for w in weights):
            raise ValueError('Zero operator')

    def step(self, field):
        xp = self.xp
        out = field.copy()
        for mu, axis in enumerate((2, 1, 0)):
            u = self.links[mu]
            forward = xp.einsum('...ab,...br->...ar', u, xp.roll(field, -1, axis=axis))
            back_u = xp.roll(u, 1, axis=axis).swapaxes(-1, -2).conj()
            backward = xp.einsum('...ab,...br->...ar', back_u, xp.roll(field, 1, axis=axis))
            out += self.alpha * (forward + backward)
        return out / (1 + 6 * self.alpha)

    def apply(self, field):
        if field.ndim != 5 or field.shape[:3] != self.links.shape[1:4] or field.shape[-2] != 3:
            raise ValueError('Expected field with shape (z,y,x,3,nrhs)')
        weights = dict(zip(self.powers, self.weights))
        out = self.xp.zeros_like(field, dtype=self.xp.result_type(field.dtype, *self.weights))
        current = field
        for n in range(max(self.powers) + 1):
            if n in weights:
                out += weights[n] * current
            if n != max(self.powers):
                current = self.step(current)
        return out


def build_kernel(spatial_links, parameters):
    """Return a linear operator using only spatial links and frozen parameters."""
    return DiffusionKernel(spatial_links, parameters['powers'], parameters['weights'], parameters['alpha'])
