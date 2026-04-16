# LatticeArena Specification

## 1. Overview

LatticeArena is a multi-task benchmark where each task isolates one lattice QCD
optimization target behind a fixed workflow and a stable scoring contract.

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

## 4. Task Contracts

### 4.1 `wilson_loop`

Optimization target: a spatial Wilson-line submission that improves overlap
with the ground state in static quark-antiquark Wilson loops.

- `dataset/`: pure-gauge SU(3) configurations
- `scripts/measure.py`: fixed Wilson-loop measurement workflow
- `interface.py`: `SpatialOperator`
- `submissions/`: spatial path constructions
- `tests/validation.py`: gauge covariance and cold-config checks
- `benchmark/metrics.py`: effective-mass and overlap-oriented scoring

The submission interface remains:

```python
class SpatialOperator(ABC):
    @property
    def meta(self) -> SubmissionMeta: ...

    def setup(self, gauge_field: np.ndarray, latt_size: tuple[int, int, int, int]) -> None: ...

    def compute(
        self,
        gauge_field: np.ndarray,
        r: int,
        direction: int,
        t: int,
    ) -> np.ndarray: ...
```

Key validation expectations:

- correct output shape and dtype
- identity on a cold gauge field
- correct gauge-covariant transformation behavior

### 4.2 `pion_2pt`

Optimization target: a boosted pion interpolating submission for pion two-point
correlators.

- `dataset/`: gauge fields and propagator inputs, documented as a placeholder
  contract for now
- `scripts/measure.py`: fixed measurement pipeline placeholder
- `interface.py`: `PionInterpolatingOperator`
- `submissions/`: source/sink profile and Dirac-structure baselines
- `tests/validation.py`: shape and normalization checks
- `benchmark/metrics.py`: explicit WIP placeholder until the live measurement
  workflow is complete

The submission interface remains:

```python
class PionInterpolatingOperator(ABC):
    @property
    def meta(self) -> SubmissionMeta: ...

    def setup(
        self,
        gauge_field: np.ndarray,
        latt_size: tuple[int, int, int, int],
        lattice_spacing_fm: float,
    ) -> None: ...

    def build(
        self,
        gauge_field: np.ndarray,
        momentum_gev: tuple[float, float, float],
        t_source: int,
    ) -> OperatorComponents: ...
```

### 4.3 `gsfit_2pt`

Optimization target: a fixed ground-state fit configuration for pion two-point
correlators.

- `dataset/synthetic.py`: synthetic correlator-case definitions and I/O
- `scripts/fit.py`: fixed `gvar`/`lsqfit` analysis pipeline
- `interface.py`: `Pion2PtGroundStateFit`
- `submissions/`: fit-configuration submissions
- `tests/validation.py`: configuration legality checks
- `benchmark/metrics.py`: scoring by bias, uncertainty, fit quality, and
  resampling robustness

The fixed fit configuration contains:

- `t_min`
- `t_max`
- `n_states`
- `e0_prior`
- `delta_e_priors`
- `amplitude_priors`

The submission does not implement a custom fitter. The fitter is entirely owned
by `scripts/fit.py`.

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
