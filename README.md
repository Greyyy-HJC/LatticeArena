# LatticeArena

LatticeArena is a benchmark suite for lattice QCD optimization targets. Each
task isolates one fixed workflow and one optimization interface, so new ideas
can be tested as comparable submissions instead of one-off scripts.

## Core Model

Every task is built from six first-class components:

- `dataset/`: task inputs and dataset contract
- `scripts/`: framework-owned fixed workflow
- `interface.py`: the submission interface for the optimization target
- `submissions/`: baselines and contributed implementations
- `tests/`: legality and validation checks
- `benchmark/`: score computation and benchmark outputs

The composition rule is:

```text
benchmark = scripts + submission + metrics
```

`scripts/` defines the fixed physics or analysis flow. `submissions/` contains
only the part we want to optimize. `benchmark/metrics.py` turns the workflow
output into a comparable score.

## Shared Framework

`core/` is the shared framework layer for the repository. It is not task-
specific physics code. Instead, it contains the cross-task utilities that hold
the benchmark collection together, such as:

- task registration and the shared `TaskBase` / `BenchmarkResult` contract
- leaderboard result loading and saving
- reusable testing helpers shared across tasks

## Task Layout

```text
tasks/<task_name>/
  dataset/
  scripts/
  interface.py
  submissions/
  tests/
    validation.py
    test_validation.py
  benchmark/
    metrics.py
    run.py
    results/
```

Incomplete tasks should still expose this full skeleton. Use explicit WIP
placeholders instead of leaving missing directories or ambiguous file names.

## Available Tasks

| Task | Optimization target | Status by component |
|---|---|---|
| [`wilson_loop`](tasks/wilson_loop/) | Spatial Wilson-line submission for static-potential Wilson loops | dataset: ready, scripts: ready, interface: ready, submissions: ready, tests: ready, benchmark: ready |
| [`pion_2pt`](tasks/pion_2pt/) | Boosted pion interpolating submission for two-point correlators | dataset: placeholder, scripts: WIP, interface: ready, submissions: baseline, tests: ready, benchmark: WIP |
| [`gsfit_2pt`](tasks/gsfit_2pt/) | Fixed ground-state fit configuration submission for pion 2pt analysis | dataset: ready, scripts: ready, interface: ready, submissions: ready, tests: ready, benchmark: ready |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[dev]"
```

Run task tests:

```bash
pytest tasks/wilson_loop/tests/
pytest tasks/pion_2pt/tests/
pytest tasks/gsfit_2pt/tests/
```

Run benchmark or workflow CLIs:

```bash
python tasks/wilson_loop/benchmark/run.py --submission plain
python tasks/gsfit_2pt/scripts/fit.py --submission plain
python tasks/gsfit_2pt/benchmark/run.py --submission plain
```

## Submission Workflow

1. Pick a task and read its `README.md`.
2. Inspect `interface.py` to understand the optimization target.
3. Use an existing file under `submissions/` as a baseline.
4. Add a new submission under `tasks/<task>/submissions/`.
5. Run `pytest tasks/<task>/tests/`.
6. Run `python tasks/<task>/benchmark/run.py --submission <name>` when that
   task has a live benchmark. The benchmark runner re-checks task validation
   and exits early if the submission fails.

## Environment Notes

- Python 3.10+
- Runtime dependencies: `numpy`, `scipy`, `gvar`, `lsqfit`, `gmpy2`,
  `pyquda`, `pyquda-utils`
- Optional plotting dependency: `matplotlib`
- GPU-dependent PyQUDA tasks require a local QUDA installation

The pure analysis task [`gsfit_2pt`](tasks/gsfit_2pt/) does not depend on
PyQUDA for its fixed workflow, but the repository dependency set still declares
the broader lattice stack.

## Leaderboard Page

```bash
python scripts/build_leaderboard_page.py
```

This writes `site/leaderboard.html`.

## License

MIT
