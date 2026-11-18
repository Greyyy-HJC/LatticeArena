"""LatticeArena submission interface for the v6 (A1+) learned smear.

build_kernel(spatial_links, parameters) -> object with .apply(field)
K = C_hop(a) G^m,  a_mu(x) = c + sum_h v_h phi_h(mu, x)
"""
import numpy as np

from .features import _lib, directional_features
from .kernels import apply_kernel


class LearnedSmear:
    def __init__(self, links, parameters):
        self.links = links
        self.W = np.asarray(parameters['W'])
        self.theta = np.asarray(parameters['theta'])
        self.steps = int(parameters['steps'])
        self.alpha = float(parameters['alpha'])

    def apply(self, field):
        xp = _lib(field)
        links = xp.asarray(self.links)
        z, y, x = field.shape[:3]
        a0 = xp.full((3, z, y, x), self.alpha)
        base = field
        for _ in range(self.steps):
            base = apply_kernel(base, links, a0)
        f = directional_features(links)
        phi = xp.stack([xp.tanh(f[mu] @ xp.asarray(self.W.T)) for mu in range(3)], axis=0)
        phi = phi - phi.mean(axis=(1, 2, 3), keepdims=True)
        a = xp.full((3, z, y, x), self.theta[0])
        for h in range(self.W.shape[0]):
            a += self.theta[1 + h] * phi[:, :, :, :, h]
        out = apply_kernel(base, links, a)
        for _ in range(self.steps):
            out = apply_kernel(out, links, a0)
        return out


def build_kernel(spatial_links, parameters):
    return LearnedSmear(spatial_links, parameters)
