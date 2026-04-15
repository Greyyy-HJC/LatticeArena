# Task: Pion 2pt Ground-State Fit

Optimize the **ground-state fit configuration** for pion two-point correlators.

This task is about analysis choices after the correlator is measured. You do
not implement a fitter or an operator. Instead, you submit one fixed set of
fit settings:

- fit range: `t_min`, `t_max`
- number of states: `N_states`
- priors for `E0`, excited-state gaps, and amplitudes

The framework then runs the same correlated Bayesian fit for every submission
and scores how robustly it extracts the true ground-state energy `E0` from
synthetic pion 2pt samples.

## Fit Model

The benchmark uses the periodic multi-state form:

```text
C(t) = sum_n A_n * [exp(-E_n t) + exp(-E_n (L_t - t))]
E_n = E_0 + sum_{k=1..n} Delta E_k
```

The fitter enforces:

- `A_n > 0`
- `E0 > 0`
- `DeltaE_n > 0`

## What You Implement

Implement `Pion2PtGroundStateFit` from `interface.py` by providing:

- `meta`
- `config`

The baseline in `operators/plain.py` shows the expected structure.

## Benchmark Target

The v1 benchmark uses **synthetic bootstrap/jackknife-like samples** with known
truth. Each case contains a pion-like correlator with different noise levels
and excited-state contamination.

Primary objective:

- extract `E0` with low bias

Secondary objectives:

- keep the uncertainty reasonable
- maintain acceptable `chi2/dof`
- avoid unstable fits across resamples

## Quick Check

```bash
pytest tasks/gsfit_2pt/tests/
python tasks/gsfit_2pt/benchmark/run.py --operator plain
```

## Fake Data

You can materialize the built-in synthetic benchmark cases as a real dataset
file:

```bash
python tasks/gsfit_2pt/scripts/generate_fake_data.py
```

This produces `tasks/gsfit_2pt/dataset/fake_data.npz`, which can be used by
the benchmark directly:

```bash
python tasks/gsfit_2pt/benchmark/run.py --operator plain \
  --dataset-file tasks/gsfit_2pt/dataset/fake_data.npz
```

## Example Fit Script

An example local script is provided to run a correlated `gvar`/`lsqfit` 2pt
fit on one fake-data case:

```bash
python tasks/gsfit_2pt/scripts/gsfit.py --operator plain \
  --dataset-file tasks/gsfit_2pt/dataset/fake_data.npz \
  --case boosted_clean
```

Optional meff plot output:

```bash
python tasks/gsfit_2pt/scripts/gsfit.py --operator plain \
  --dataset-file tasks/gsfit_2pt/dataset/fake_data.npz \
  --case boosted_clean \
  --plot-output tasks/gsfit_2pt/benchmark/results/boosted_clean_meff.png
```

## NN Optimization

A tiny NumPy MLP surrogate is included to search for stronger fit settings:

```bash
python tasks/gsfit_2pt/scripts/optimize_nn.py
python tasks/gsfit_2pt/benchmark/run.py --operator nn \
  --dataset-file tasks/gsfit_2pt/dataset/fake_data.npz
```
