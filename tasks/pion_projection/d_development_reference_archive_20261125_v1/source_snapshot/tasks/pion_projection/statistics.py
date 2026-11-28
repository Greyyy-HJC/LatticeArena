"""Fixed finite-time reference fit and block-bootstrap risk estimator.

The pilot restricts released submissions to real Hermitian diffusion polynomials.
No complex correlator is silently projected onto its real part.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


def fold(data):
    data = np.asarray(data)
    if data.shape[-1] % 2 or not np.all(np.isfinite(data)):
        raise ValueError('Finite even-length correlators required')
    t = data.shape[-1]
    return np.stack([data[..., 0]] + [0.5*(data[..., j]+data[..., t-j])
                                    for j in range(1, t//2)] + [data[..., t//2]], axis=-1)


def blocks(data, size):
    if len(data) % size or len(data)//size < 3:
        raise ValueError('Need at least three complete blocks; no rows may be dropped')
    return data.reshape(len(data)//size, size, *data.shape[1:]).mean(axis=1)


def shape(mass, times, temporal=16):
    t = np.asarray(times)
    return np.exp(-np.asarray(mass)[..., None]*t) + np.exp(-np.asarray(mass)[..., None]*(temporal-t))


def inverse_covariance(samples, shrink=0.25):
    """Covariance of the block mean, with fixed diagonal shrinkage/eigen floor."""
    samples = np.asarray(samples)
    if samples.ndim != 2 or len(samples) < 3:
        raise ValueError('Expected at least three block vectors')
    cov = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1)) / len(samples)
    cov = (1-shrink)*cov + shrink*np.diag(np.diag(cov))
    eig, vec = np.linalg.eigh(cov)
    scale = float(np.max(eig))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Unidentifiable covariance')
    eig = np.maximum(eig, scale*1e-10)
    return (vec/eig) @ vec.T


class ReferenceFit:
    """Common mass, separate amplitudes, fixed covariance within resampling."""
    def __init__(self, data_blocks, config, window):
        self.p = config
        self.window = np.asarray(window)
        self.basis = config['reference_basis']
        self.samples = data_blocks[:, self.basis][:, :, self.window]
        self.scales = np.mean(self.samples, axis=0)[:, 0]
        if np.any(self.scales == 0):
            raise ValueError('Zero reference channel')
        self.samples = self.samples / self.scales[None, :, None]
        self.samples = self.samples.reshape(len(data_blocks), -1)
        self.inv = inverse_covariance(self.samples, config['covariance_shrinkage'])

    def fit(self, mean):
        def objective(mass):
            f = shape(mass, self.window, self.p['geometry'][3])
            design = np.kron(np.eye(len(self.basis)), f[:, None])
            amp = np.linalg.solve(design.T @ self.inv @ design, design.T @ self.inv @ mean)
            residual = mean - design @ amp
            return float(residual @ self.inv @ residual)
        fit = minimize_scalar(objective, bounds=self.p['mass_bounds'], method='bounded',
                              options={'xatol': 1e-9})
        if not fit.success or not np.isfinite(fit.fun):
            raise ValueError('Reference fit failed')
        return float(fit.x), float(fit.fun)

    def bootstrap(self, index):
        mean = self.samples.mean(axis=0)
        mass, cost = self.fit(mean)
        masses = np.array([self.fit(row)[0] for row in self.samples[index].mean(axis=1)])
        dof = len(mean)-len(self.basis)-1
        return {'mass': mass, 'se': float(masses.std(ddof=1)), 'chi2': cost,
                'dof': dof, 'p_approx': float(chi2.sf(cost, dof)),
                'window': self.window.tolist()}, masses


def bootstrap_indices(nblocks, count, seed):
    return np.random.default_rng(seed).integers(0, nblocks, size=(count, nblocks))


def risk(data_blocks, weights, mass, boot_masses, index, config, require_valid=True, late_inverse=None):
    """Central loss; bootstrap propagates both late-amplitude and mass errors."""
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)) or not np.any(weights):
        raise ValueError('Invalid or zero weights')
    values = np.einsum('nbt,b->nt', data_blocks, weights)
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite correlator')
    mean = values.mean(axis=0)
    late, early = config['amplitude_window'], config['score_window']
    inv = inverse_covariance(values[:, late], config['covariance_shrinkage']) if late_inverse is None else late_inverse
    f = shape(mass, np.arange(values.shape[-1]), config['geometry'][3])
    amp = float(f[late] @ inv @ mean[late] / (f[late] @ inv @ f[late]))
    if amp == 0 or not np.isfinite(amp):
        raise ValueError('Zero or invalid amplitude')
    boot_mean = values[index].mean(axis=1)
    fb = shape(boot_masses, np.arange(values.shape[-1]), config['geometry'][3])
    amps = np.einsum('bi,ij,bj->b', fb[:, late], inv, boot_mean[:, late]) / np.einsum(
        'bi,ij,bj->b', fb[:, late], inv, fb[:, late])
    if not np.all(np.isfinite(amps)) or np.any(amps == 0):
        raise ValueError('Invalid bootstrap amplitude')
    amp_se = float(amps.std(ddof=1))
    z = abs(amp)/amp_se if amp_se > 0 else 1e300
    if require_valid and z < config['amplitude_z_min']:
        raise ValueError(f'Amplitude not identified: z={z:.3f}')
    q = mean[early]/(amp*f[early])
    qb = boot_mean[:, early]/(amps[:, None]*fb[:, early])
    variance = float(qb.var(axis=0, ddof=1).mean())
    bias = float(np.mean((q-1)**2))
    se = boot_mean.std(axis=0, ddof=1)
    if not np.all(np.isfinite(qb)) or (require_valid and np.any(se == 0)):
        raise ValueError('Invalid bootstrap statistics')
    snr = np.divide(abs(mean), se, out=np.zeros_like(mean), where=se > 0)
    return {'risk': variance+bias, 'variance': variance, 'departure_squared': bias,
            'amplitude': amp, 'amplitude_se': amp_se, 'amplitude_z': z,
            'q': q.tolist(), 'q_se': qb.std(axis=0, ddof=1).tolist(),
            'mean': mean.tolist(), 'se': se.tolist(), 'snr': snr.tolist()}


def effective_mass(mean):
    ratio = (mean[:-2]+mean[2:])/(2*mean[1:-1])
    out = np.full_like(ratio, np.nan)
    valid = (ratio >= 1) & np.isfinite(ratio)
    out[valid] = np.arccosh(ratio[valid])
    return out
