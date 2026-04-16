# Task: gsfit_2pt

`gsfit_2pt` is an analysis task rather than a measurement task. The submission
is a fixed ground-state fit configuration, while the fitting pipeline itself is
framework-owned.

## Task Components

- `dataset/`: synthetic pion two-point benchmark cases
- `scripts/fit.py`: fixed `gvar`/`lsqfit` analysis workflow
- `interface.py`: `Pion2PtGroundStateFit`
- `submissions/`: baseline and optimized fit configurations
- `tests/`: configuration legality and benchmark regression checks
- `benchmark/`: score computation and benchmark CLI

## Optimization Target

Submissions choose one fixed fit configuration:

- fit window: `t_min`, `t_max`
- number of states: `n_states`
- priors for `E0`
- priors for energy gaps
- priors for amplitudes

The submission does not implement a custom fitter. `scripts/fit.py` is the
single source of truth for the fixed fit pipeline.

## Benchmark

The benchmark uses deterministic synthetic correlator cases with known truth and
scores each submission by combining:

- ground-state bias
- uncertainty size
- `chi2/dof`
- failure rate across resample refits

## Quick Start

```bash
pytest tasks/gsfit_2pt/tests/
python tasks/gsfit_2pt/scripts/fit.py --submission plain
python tasks/gsfit_2pt/benchmark/run.py --submission plain
```

The benchmark runner validates the submission before scoring. If validation
fails, it exits and tells you to run `pytest tasks/gsfit_2pt/tests/`.

Generate a saved synthetic dataset archive:

```bash
python tasks/gsfit_2pt/scripts/generate_fake_data.py
```

Run the benchmark against the saved archive:

```bash
python tasks/gsfit_2pt/benchmark/run.py \
  --submission plain \
  --dataset-file tasks/gsfit_2pt/dataset/fake_data.npz
```

Run the fixed analysis workflow on one case:

```bash
python tasks/gsfit_2pt/scripts/fit.py \
  --submission plain \
  --dataset-file tasks/gsfit_2pt/dataset/fake_data.npz \
  --case boosted_clean
```

`scripts/optimize_nn.py` is an optional framework helper for searching stronger
fit configurations. It is not part of the benchmark contract.
