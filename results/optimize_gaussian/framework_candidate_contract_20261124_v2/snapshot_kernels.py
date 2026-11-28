"""Learned gauge-invariant smearing: K = I + sum_mu (a_mu T_mu + b_mu T_mu^dag), b_mu(x) = a_mu(x-mu)."""
import numpy as np

from .features import _lib, dagger, shift


def apply_kernel(field, links, a):
    """field: (z, y, x, 3, n); links: (3, z, y, x, 3, 3); a: (3, z, y, x)."""
    xp = _lib(field)
    out = field
    for d in range(3):
        b = shift(a[d], -1, d)                       # b_mu(x) = a_mu(x - mu)
        forward = links[d] @ shift(field, 1, d)
        backward = dagger(shift(links[d], -1, d)) @ shift(field, -1, d)
        out = out + a[d][..., None, None] * forward + b[..., None, None] * backward
    return out


def correlator(field, links, a):
    out = apply_kernel(field, links, a)
    return _lib(field).vdot(field, out).real
