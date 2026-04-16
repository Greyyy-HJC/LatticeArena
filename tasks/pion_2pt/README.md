# Task: Pion 2pt (boosted)

Optimization goal: design a better **pion interpolating operator** that achieves the following for **boosted pions** (flagship example `|p| ~ 1 GeV`):

- Higher signal-to-noise ratio (SNR)
- Reduced excited-state contamination (earlier, flatter effective-mass plateau)

## Physics target

The measured correlator is:

$$
C_\pi(\mathbf{p}, t) = \langle O_\pi(\mathbf{p}, t_0+t)\, O_\pi^\dagger(\mathbf{p}, t_0) \rangle,
\quad
O_\pi(\mathbf{p}, t) = \sum_{\mathbf{x}} e^{i\mathbf{p}\cdot\mathbf{x}}\, \bar{d}(\mathbf{x},t)\,\Gamma\,u(\mathbf{x},t).
$$

You optimize the construction of $O_\pi$ (e.g. smearing profile, momentum injection strategy, Dirac structure, derivative operators) to improve the effective-mass plateau and statistical precision for boosted pions.

## What you implement

Implement `PionInterpolatingOperator` from `tasks/pion_2pt/interface.py`:

- `setup(...)`: one-time preprocessing per gauge configuration
- `build(...)`: for a given momentum (GeV) and source timeslice, return source/sink profiles and Dirac structure

## Baseline

`operators/plain.py` provides a simple baseline:

- Gaussian spatial profile
- Plane-wave momentum phase injection
- `Gamma = gamma_5`

## Suggested benchmark metrics

- **SNR at fixed t**: `mean(C) / std(C)`
- **Plateau onset**: earliest `t_min` at which the effective mass enters the plateau
- **Excited-state contamination**: excited-state amplitude from a 2-state fit
- **Dispersion consistency**: agreement of `E(p)` with the continuum dispersion relation

## Quick check

```bash
pytest tasks/pion_2pt/tests/
python tasks/pion_2pt/benchmark/run.py --operator plain
```
