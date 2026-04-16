# LatticeArena Specification

## 1. Overview

LatticeArena is a multi-task benchmark where each task isolates one lattice QCD
optimization target behind a fixed workflow and a stable scoring contract.

## 1.1 Document map (where to find things)

- **This file (`SPEC.md`)**: the stable, cross-task contract (structure + rules).
- **Project log (`PROJECT_LOG.md`)**: see [`PROJECT_LOG.md`](PROJECT_LOG.md) for long-form, living notes:
  - detailed per-task explanations/contracts
  - important design decisions
  - important changes / migration notes
- **PyQUDA integration notes**: see [`PYQUDA_NOTES.md`](PYQUDA_NOTES.md) for shared PyQUDA / `pyquda-utils` constraints and reminders.
- **Task-local docs**: `tasks/<task>/dataset/README.md` is the canonical place for
  dataset acquisition/generation details and any large-file handling notes.

## 2. Standard Task Structure

Every task lives under `tasks/<task_name>/` and must contain:

| Path | Purpose |
|---|---|
| `dataset/` | Task inputs and dataset contract. Large files may be gitignored; `dataset/README.md` documents how data is obtained or generated. |
| `scripts/` | Framework-owned fixed workflow. This is the part of the task that is not optimized by submissions. |
| `interface.py` | Abstract interface for the optimization target. |
| `submissions/` | Baseline and contributed submission implementations. |
| `tests/` | Legality and validation checks. |
| `tests/validation.py` | Reusable validation helpers shared by task code and tests. |
| `tests/test_validation.py` | Main validation test module. |
| `benchmark/metrics.py` | Score computation. |
| `benchmark/run.py` | Benchmark CLI driver. |
| `benchmark/results/` | Saved benchmark outputs. |

The design rule is:

```text
benchmark = scripts + submission + metrics
```

### Structure Rules

1. Incomplete tasks must still expose the full skeleton above.
2. Use explicit WIP placeholders instead of omitting `benchmark/metrics.py` or
   `scripts/`.
3. Use the `submissions/` directory for all contributed implementations.
4. Use the `--submission` CLI flag consistently.
5. `benchmark/run.py` should remain a thin driver:
   - validate the submission
   - load the submission
   - load or generate task input data
   - invoke the fixed workflow
   - call `benchmark/metrics.py`
   - save a `BenchmarkResult`

## 3. Submission Rules

1. A submission is a Python module under `tasks/<task_name>/submissions/`.
2. The submission must implement the task interface in `interface.py`.
3. The submission must pass all tests in `tasks/<task_name>/tests/`.
4. The benchmark is run through `tasks/<task_name>/benchmark/run.py`.
   `benchmark/run.py` must validate the submission before scoring and exit
   early if validation fails.
5. Higher score ranks higher unless a task documents otherwise.
6. Task-specific dependencies beyond the repository standard stack must be
   documented explicitly.

## 4. Task contracts (where they live)

Task-specific detailed descriptions (interfaces, validation expectations, WIP
status, scoring notes) live in:

- [`PROJECT_LOG.md`](PROJECT_LOG.md) → **Task contracts (migrated from `SPEC.md`)**

Task-specific dataset contracts and data acquisition/generation live in:

- `tasks/<task_name>/dataset/README.md`

## 5. Shared Framework

`core.task.TaskBase` defines the registry-facing task interface:

```python
class TaskBase(ABC):
    @property
    def name(self) -> str: ...

    def validate(self, operator) -> bool: ...

    def benchmark(self, operator, dataset_path: str | Path | None = None) -> BenchmarkResult: ...
```

`BenchmarkResult` stores:

- `task_name`
- `submission_name`
- `score`
- `metrics`

`core.leaderboard` collects saved benchmark results from
`tasks/<task_name>/benchmark/results/`.

## 6. Adding a New Task

1. Create `tasks/<task_name>/`.
2. Add the standard directories and files from Section 2.
3. Define the optimization-target interface in `interface.py`.
4. Implement the fixed workflow in `scripts/`.
5. Add a baseline submission under `submissions/`.
6. Add reusable checks to `tests/validation.py`.
7. Add task tests to `tests/test_validation.py` and any additional task tests.
8. Add score computation to `benchmark/metrics.py`.
9. Add a benchmark CLI to `benchmark/run.py`.
