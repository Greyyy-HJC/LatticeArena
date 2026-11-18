"""Simulation-calibrated p-value for the common-mass reference fit."""
import numpy as np

from .statistics import shape


def fitted_mean(model, mean):
    mass, cost = model.fit(mean)
    f = shape(mass, model.window, model.p['geometry'][3])
    design = np.kron(np.eye(len(model.basis)), f[:, None])
    amp = np.linalg.solve(design.T @ model.inv @ design, design.T @ model.inv @ mean)
    return mass, cost, design @ amp


def calibrated_p(model, mean, nsim=200, seed=0):
    """Empirical p-value of the fit statistic under the fitted model."""
    _, cost, mu = fitted_mean(model, mean)
    cov = np.linalg.pinv(model.inv)
    cov = 0.5 * (cov + cov.T) + 1e-13 * np.eye(len(mu))
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    sims = mu + rng.normal(size=(nsim, len(mu))) @ chol.T
    costs = np.array([model.fit(row)[1] for row in sims])
    return float(np.mean(costs >= cost)), float(cost), costs
