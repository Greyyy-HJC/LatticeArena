# Task: Pion 2pt

`pion_2pt` is a measurement task for boosted pion two-point correlators. The
optimization target is the interpolating submission used by a fixed PyQUDA
measurement pipeline.

## Task Components

- `dataset/`: task-local dataset contract plus local-only quenched smoke tests
- `scripts/measure.py`: fixed PyQUDA measurement workflow
- `interface.py`: `PionInterpolatingOperator`
- `submissions/`: baseline interpolating submissions
- `tests/`: validation of shapes and normalization
- `benchmark/`: benchmark scoring and CLI entrypoint

## Optimization Target

Submissions define how the pion source and sink are built for boosted pions:

- spatial source profile
- spatial sink profile
- Dirac structure

The current reference submissions are:

- `submissions/plain.py`: pseudoscalar (`gamma_5`) with a point source and
  plane-wave sink projection
- `submissions/axial_smeared.py`: Gaussian momentum-smeared profile with a
  boost-aligned axial-style Dirac structure

Measurement scripts in this task are written against
[PyQUDA](https://github.com/CLQCD/PyQUDA).

## Current Status

- Interface: ready
- Reference submissions: ready
- Validation tests: ready
- Measurement pipeline: ready for local PyQUDA datasets
- Benchmark scoring: ready for local smoke tests

## Quick Start

```bash
pytest tasks/pion_2pt/tests/
python tasks/pion_2pt/scripts/measure.py --submission plain
python tasks/pion_2pt/benchmark/run.py --submission plain
```

Useful references:

- PyQUDA upstream: [CLQCD/PyQUDA](https://github.com/CLQCD/PyQUDA)
- PyQUDA notes (repo-wide): [`PYQUDA_NOTES.md`](../../PYQUDA_NOTES.md)
