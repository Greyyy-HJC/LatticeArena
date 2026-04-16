# Task: Pion 2pt

`pion_2pt` is a measurement-task skeleton for boosted pion two-point
correlators. The optimization target is the interpolating submission used by a
fixed measurement pipeline.

This task is intentionally incomplete, but its structure is already standardized
to match the rest of the repository.

## Task Components

- `dataset/`: placeholder dataset contract for gauge fields and propagator data
- `scripts/measure.py`: fixed measurement-pipeline placeholder
- `interface.py`: `PionInterpolatingOperator`
- `submissions/`: baseline interpolating submissions
- `tests/`: validation of shapes and normalization
- `benchmark/`: explicit WIP benchmark skeleton

## Optimization Target

Submissions define how the pion source and sink are built for boosted pions:

- spatial source profile
- spatial sink profile
- Dirac structure

The current baseline is `submissions/plain.py`, which provides a normalized
Gaussian profile multiplied by a plane-wave phase and uses `gamma_5`.

## Current Status

- Interface: ready
- Baseline submission: ready
- Validation tests: ready
- Measurement pipeline: WIP
- Benchmark scoring: WIP

## Quick Start

```bash
pytest tasks/pion_2pt/tests/
python tasks/pion_2pt/benchmark/run.py --submission plain
```

The benchmark CLI currently returns a clear WIP message because the fixed
PyQUDA measurement workflow is not implemented yet.
