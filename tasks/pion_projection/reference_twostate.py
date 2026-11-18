"""Two-state common-mass reference: A_k f(t;m0) + B_k f(t;m1)."""
import numpy as np
from scipy.stats import chi2

from .statistics import inverse_covariance, shape


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

    def design(self, m0, m1):
        T = self.p['geometry'][3]
        f0 = shape(m0, self.window, T)
        f1 = shape(m1, self.window, T)
        return np.kron(np.eye(self.nchan), np.stack([f0, f1], axis=1))

    def _cost(self, m0, m1, mean):
        X = self.design(m0, m1)
        XtW = X.T @ self.inv
        amp = np.linalg.solve(XtW @ X, XtW @ mean)
        r = mean - X @ amp
        return float(r @ self.inv @ r)

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
        X = self.design(m0, m1)
        XtW = X.T @ self.inv
        amp = np.linalg.solve(XtW @ X, XtW @ mean)
        lo, hi = self.p['mass_bounds']
        return {'mass': float(m0), 'mass2': float(m1), 'gap': float(m1 - m0),
                'amplitudes': amp.tolist(), 'cost': float(cost),
                'condition': float(np.linalg.cond(XtW @ X)),
                'at_boundary': bool(min(m0 - lo, hi - m1) < 1e-3)}

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
                'at_boundary': full['at_boundary'], 'se': float(masses.std(ddof=1)),
                'chi2': full['cost'], 'dof': dof,
                'p_approx': float(chi2.sf(full['cost'], dof)),
                'window': self.window.tolist()}, masses
