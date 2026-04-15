# AGENTS

Guidance for coding agents working in the `LatticeArena` repository.

## Purpose

`LatticeArena` is a multi-task benchmark for lattice QCD operator optimization.
Each task isolates one optimization problem, exposes a narrow submission
interface, validates legality with tests, and scores valid submissions with a
benchmark pipeline.

Primary source documents:

- `CLAUDE.md` for repository workflow and conventions
- `SPEC.md` for benchmark/task contracts

When these differ in specificity, prefer the more local and task-facing contract
in `SPEC.md`, then keep repository-level wording in sync when editing docs.

## Environment

Use the local virtual environment when running Python tooling:

```bash
source .venv/bin/activate
```

Common commands:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest tasks/wilson_loop/tests/
.venv/bin/python tasks/wilson_loop/benchmark/run.py --operator plain
```

Repository expectations:

- Python 3.10+
- Standard runtime deps include `numpy`, `scipy`, `gvar`, `lsqfit`, `gmpy2`,
  `pyquda`, and `pyquda-utils`
- On CUDA 12 systems, local PyQUDA script runs also require a matching CuPy
  wheel such as `cupy-cuda12x`
- PyQUDA depends on a working QUDA installation for GPU-backed lattice tasks

## Repo Layout

```text
latticearena/          Shared framework (task registration, leaderboard)
tasks/<task_name>/     Self-contained benchmark tasks
  dataset/             Data location and acquisition/generation docs
  scripts/             Physics measurement and generation scripts
  interface.py         Optimization target ABC
  operators/           Submission implementations
  tests/               Legality and regression tests
  benchmark/           Metrics, runs, and saved results
```

Agents should preserve this structure. New task-specific logic belongs inside
the corresponding `tasks/<task_name>/` subtree.

## Task Model

Every task should provide:

1. A clearly scoped interface in `interface.py`
2. Validation tests in `tests/`
3. Benchmark logic in `benchmark/`
4. Dataset instructions in `dataset/README.md`
5. Runnable scripts in `scripts/` for the underlying physics pipeline

Submission rules from `SPEC.md`:

1. A submission is one Python file in `tasks/<task_name>/operators/`
2. The submission must implement the task interface
3. The submission must pass task validation tests
4. The benchmark score determines leaderboard ranking
5. Additional dependencies should not be introduced without approval

## Current Tasks

- `wilson_loop`: optimize spatial Wilson line operators for static
  quark-antiquark Wilson loops
- `pion_2pt`: optimize boosted pion interpolating operators
- `gsfit_2pt`: optimize fixed ground-state fit configurations for pion 2pt data

## Data Ordering And Field Conventions

Be explicit about gauge-field ordering. Different layers use different layouts.

- PyQUDA internally commonly uses even/odd ordering
  `(Nd, 2, Lt/2, Lz, Ly, Lx/2, Nc, Nc)`
- Task-facing operator interfaces may instead require natural or lexicographic
  ordering
- For `wilson_loop`, `SPEC.md` defines the operator-facing gauge field shape as
  `(Nd, Lx, Ly, Lz, Lt, Nc, Nc)`

Do not assume one ordering from another file without checking the task contract.
If a script converts between layouts, document the conversion near the code and
test it.

## Task-Specific Notes

### `wilson_loop`

- Physics goal: improve ground-state overlap `|c_0|^2` for Wilson loops
- Dataset: pure-gauge quenched SU(3) ensembles with Wilson gauge action
- Operator contract:
  - `setup(gauge_field, latt_size)`
  - `compute(gauge_field, r, direction, t) -> (Lx, Ly, Lz, Nc, Nc)`
- Core legality requirements:
  - gauge equivariance
  - correct output shape
  - identity on a cold/unit gauge field
- Measurement pipeline should load configs, convert to the task ordering, call
  `setup`, evaluate `compute`, and assemble Wilson loops

### `pion_2pt`

- Focus on boosted pion operators near `|p| ~ 1 GeV`
- Submission interface returns source/sink profiles and a Dirac matrix

### `gsfit_2pt`

- This is an analysis task, not a gauge-generation or operator-construction task
- Submissions choose a fixed fit configuration rather than implementing a fitter

## Editing Guidance

- Prefer minimal, local changes that preserve task boundaries
- Keep public task interfaces stable unless the spec explicitly changes
- Update task docs when changing on-disk formats, CLI contracts, or benchmark
  behavior
- Add or update tests whenever behavior, contracts, or file layouts change
- Avoid importing heavy optional runtime dependencies at module import time when
  a script can lazy-load them instead

## Validation Expectations

Before finishing work, run the narrowest relevant checks with `.venv`:

- task unit tests for edited modules
- CLI smoke checks for new scripts
- formatting or import-sanity checks when full runtime dependencies are absent

If a check cannot run because QUDA or GPU runtime is unavailable, say so
explicitly and still verify everything else that is feasible in `.venv`.
