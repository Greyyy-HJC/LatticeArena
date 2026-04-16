# Task: Wilson Loop

`wilson_loop` is the reference measurement-task layout in this repository.

The optimization target is a spatial Wilson-line submission used inside a fixed
Wilson-loop measurement workflow. Better submissions should improve ground-state
overlap, flatten the effective-mass plateau earlier, and preserve gauge
covariance.

## Task Components

- `dataset/`: pure-gauge SU(3) configurations for development and scoring
- `scripts/`: fixed Wilson-loop measurement workflow
- `interface.py`: `SpatialOperator`
- `submissions/`: baseline and contributed spatial-path submissions
- `tests/`: validation of gauge covariance and cold-config behavior
- `benchmark/`: metric computation and benchmark CLI

## Submission Contract

Implement `SpatialOperator` from `interface.py`.

- `setup(...)` runs once per gauge configuration
- `compute(...)` returns the spatial Wilson-line field for one `(r, direction, t)`

Gauge covariance is the key legality condition. On a cold gauge field, the
submission should reduce to the identity matrix field.

## Quick Start

```bash
pytest tasks/wilson_loop/tests/
python tasks/wilson_loop/scripts/measure.py --submission plain
python tasks/wilson_loop/benchmark/run.py --submission plain
```

To benchmark a specific dataset subset:

```bash
python tasks/wilson_loop/benchmark/run.py \
  --submission plain \
  --dataset-path tasks/wilson_loop/dataset/test_small \
  --r-values 1,2,3 \
  --t-values 0,1,2,3,4 \
  --max-configs 4
```

## Baseline

`submissions/plain.py` implements the straight Wilson line built from ordered
link products along a single spatial direction.
