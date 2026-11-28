"""Gauge-invariant, time-local features, arranged in the A1+ cubic channel."""

FEATURE_VERSION = 'a1plus-link-centred-v3'
import numpy as np


def _lib(a):
    if type(a).__module__.startswith('cupy'):
        import cupy
        return cupy
    return np


def shift(a, k, d):
    """a[x + k*mu_d] for links/fields whose spatial axes are (z, y, x) = (0, 1, 2)."""
    return _lib(a).roll(a, -k, axis=2 - d)


def dagger(a):
    return a.swapaxes(-1, -2).conj()


def plaquettes(links):
    """links: (3, z, y, x, 3, 3) -> (z, y, x, 3) real traces of xy, xz, yz."""
    xp = _lib(links)
    out = []
    for mu, nu in ((0, 1), (0, 2), (1, 2)):
        loop = links[mu] @ shift(links[nu], 1, mu) @ dagger(shift(links[mu], 1, nu)) @ dagger(links[nu])
        out.append(xp.trace(loop, axis1=-2, axis2=-1).real / 3)
    return xp.stack(out, axis=-1)


def link_plaquettes(links):
    """Plaquettes averaged over the two sides of each link, in the planes containing it.

    For the link (x, mu) and transverse direction nu the two plaquettes are
    P_{mu nu}(x) and P_{mu nu}(x - nu). Their average is centred on the link,
    so a cube rotation maps it to the corresponding link-centred average
    without moving the anchor. Returns (3, z, y, x, 2):
    mu=x -> (xy, xz), mu=y -> (yx, yz), mu=z -> (zx, zy).
    """
    xp = _lib(links)
    p = plaquettes(links)

    def side(i, nu):
        return 0.5 * (p[..., i] + shift(p[..., i], -1, nu))

    return xp.stack([xp.stack([side(0, 1), side(1, 2)], axis=-1),
                     xp.stack([side(0, 0), side(2, 2)], axis=-1),
                     xp.stack([side(1, 0), side(2, 1)], axis=-1)], axis=0)


def directional_features(links):
    """Link-centred, exchange-symmetric inputs for each direction: (3, z, y, x, 2).

    The link-centred plaquette averages are rotation covariant (the anchor is
    the link, not a corner); the (sum, product) pair is invariant under
    exchanging the two transverse directions and sharing one function across
    the three directions makes the coefficient field an A1+ vector.
    """
    p = link_plaquettes(links)
    pairs = [p[0], p[1], p[2]]
    return _lib(links).stack([_lib(links).stack([q[..., 0] + q[..., 1], q[..., 0] * q[..., 1]], axis=-1)
                              for q in pairs], axis=0)
