# LatticeArena Project Log

This document is the **long-form, living log** for LatticeArena.

- It collects **detailed task explanations**, **design decisions**, and **important changes**.
- It is intentionally allowed to grow over time.
- `SPEC.md` should stay short and act as a **map** that points to the right place.

---

## How to write in this log

### Entry template (recommended)

Use this template when logging a design decision or a major change:

- **Date**: YYYY-MM-DD
- **Area**: task name (`wilson_loop` / `pion_2pt` / `gsfit_2pt`) or `core/` or repo-level
- **Type**: design / change / refactor / bugfix / docs
- **Context**: what problem triggered the work
- **Decision**: what we chose and why
- **Impact**: what changed for users/agents/submissions
- **Follow-ups**: what remains, links to issues/tasks

---

## Project-wide design notes

### Benchmark contract invariants

- **Invariant**: `benchmark = scripts + submission + metrics`
- **Meaning**:
  - `scripts/` is framework-owned fixed workflow (not optimized by submissions)
  - `submissions/` contains the optimization target implementations
  - `benchmark/metrics.py` defines scoring; higher score ranks higher unless stated

### Why keep `SPEC.md` short

`SPEC.md` is a navigation aid: it should define the stable cross-task contract and
point to the task-level details elsewhere, so it does not become an ever-growing
wall of text.

---

## Task contracts (migrated from `SPEC.md`)

> Notes:
>
> - These sections are intentionally detailed.
> - Keep task-specific evolution here (and/or in `tasks/<task>/dataset/README.md`).

### `wilson_loop`

Optimization target: a spatial Wilson-line submission that improves overlap
with the ground state in static quark-antiquark Wilson loops.

- **`dataset/`**: pure-gauge SU(3) configurations
- **`scripts/measure.py`**: fixed Wilson-loop measurement workflow
- **`interface.py`**: `SpatialOperator`
- **`submissions/`**: spatial path constructions
- **`tests/validation.py`**: gauge covariance and cold-config checks
- **`benchmark/metrics.py`**: effective-mass and overlap-oriented scoring

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

### `pion_2pt`

Optimization target: a boosted pion interpolating submission for pion two-point
correlators.

- **`dataset/`**: gauge fields plus task-local `ensemble.json` metadata (local-only
  quenched smoke tests are gitignored)
- **`scripts/measure.py`**: fixed PyQUDA measurement workflow (submission-aware)
- **`interface.py`**: `PionInterpolatingOperator`
- **`submissions/`**: source/sink design and Dirac-structure submissions
- **`tests/validation.py`**: spec legality plus profile normalization checks
- **`benchmark/metrics.py`**: scoring by signal-to-noise and excited-state proxies

Current reference submissions:

- `plain`: `gamma_5` with point source + plane-wave sink projection
- `temporal_axial`: the same point-source / plane-wave profile as `plain`,
  but with a temporal-axial `gamma_t gamma_5` bilinear

Current benchmark target:

- For ensembles without scale setting, boosted momentum is specified as a lattice
integer mode `(npx, npy, npz)`.
- The default local target is `(3, 3, 3)` (see `tasks/pion_2pt/dataset/README.md`).

The submission interface is:

```python
class PionInterpolatingOperator(ABC):
    @property
    def meta(self) -> SubmissionMeta: ...

    def setup(
        self,
        gauge_field: Any,
        latt_size: tuple[int, int, int, int],
        lattice_spacing_fm: float | None,
    ) -> None: ...

    def design_source(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> SourceSpec: ...

    def design_sink(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> SinkSpec: ...

    def gamma_matrix(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> np.ndarray: ...
```

### `gsfit_2pt`

Optimization target: a fixed ground-state fit configuration for pion two-point
correlators.

- **`dataset/synthetic.py`**: synthetic correlator-case definitions and I/O
- **`scripts/fit.py`**: fixed `gvar`/`lsqfit` analysis pipeline
- **`interface.py`**: `Pion2PtGroundStateFit`
- **`submissions/`**: fit-configuration submissions
- **`tests/validation.py`**: configuration legality checks
- **`benchmark/metrics.py`**: scoring by bias, uncertainty, fit quality, and
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

---

## Change log

### 2026-04-16 — `pion_2pt` measure follows `pion_disp.py`

- **Area**: `pion_2pt`
- **Type**: change / bugfix / interface change
- **Context**: the first task integration already used PyQUDA, but the
  submission contract still returned raw source/sink profiles through `build()`
  and the measurement flow drifted away from the simpler `pion_disp.py`
  reference shape. In addition, the temporal-axial baseline could produce all
  `nan` effective masses because the source-side gamma ordering in the fixed
  contraction was wrong for non-pseudoscalar bilinears.
- **Decision**:
  - replace `build()` with explicit `design_source()`, `design_sink()`, and
    `gamma_matrix()` hooks
  - add source/sink spec types so point sources and plane-wave sinks are
    represented directly instead of being encoded as raw arrays
  - make the reference workflow follow `pion_disp.py` more closely:
    `getClover(...)`, `dirac.loadGauge(...)`, `core.invert(..., "point", ...)`,
    fixed contraction, source-time rolling, and config averaging
  - keep a `colorvector` fallback only for arbitrary profile sources
  - fix the source-side contraction factor to use `Gamma^dagger @ gamma_5`
    instead of `gamma_5 @ Gamma^dagger`
- **Impact**:
  - `plain` now matches the point-source / plane-wave / `gamma_5` reference
    setup from `pion_disp.py`
  - `temporal_axial` differs from `plain` only by using `gamma_t gamma_5`
  - task tests now lock the new submission API and the contraction-order
    regression
  - this is a breaking submission-API change for `pion_2pt`

### 2026-04-16 — `pion_2pt` benchmark polish and temporal-axial rename

- **Area**: `pion_2pt`
- **Type**: change / docs / interface cleanup
- **Context**: the task still mixed hand-written gamma matrices, a misleading
  `axial_smeared` reference submission, and benchmark output that only saved
  JSON metrics without a quick-look meff artifact.
- **Decision**:
  - use `pyquda_utils.gamma.gamma(...)` as the single source of truth for task
    Dirac matrices
  - replace the old `axial_smeared` submission with `temporal_axial`, keeping
    the same point-source / plane-wave profiles as `plain` but changing the
    bilinear to `gamma_t gamma_5`
  - switch effective-mass analysis to the periodic/cosh definition and save a
    benchmark-side meff PDF artifact alongside the JSON result
  - keep the submission interface complete at the task level rather than moving
    solve logic into submissions
- **Impact**:
  - `tasks/pion_2pt/benchmark/results/` now has room for discoverable meff plot
    artifacts
  - the task no longer maintains any local handwritten gamma matrices
  - benchmark plotting now uses the shared repository plot style from
    `core/plot_settings.py`
  - submission loading now instantiates only classes defined in the target
    submission module, avoiding imported baseline classes shadowing renamed
    submissions such as `temporal_axial`
  - docs and tests now describe the completed interface and renamed submission

### 2026-04-16 — Docs re-organization

- **Area**: repo-level docs
- **Type**: docs
- **Decision**: move long task-contract text out of `SPEC.md` into this log
- **Impact**: `SPEC.md` becomes a short map; task details live here

### 2026-04-16 — `pion_2pt` measurement workflow wired

- **Area**: `pion_2pt`
- **Type**: change / refactor
- **Context**: `pion_2pt` existed as a skeleton but `scripts/measure.py` was a
standalone research script (hard-coded paths, `lametlat` dependency, no
submission loop), and benchmark scoring was WIP.
- **Decision**:
  - refactor the fixed workflow into a submission-aware PyQUDA pipeline
  - treat quenched ensembles without scale setting in lattice momentum modes
  - realize arbitrary source profiles via PyQUDA `colorvector` sources while
    keeping the solve and contraction in the framework
- **Impact**:
  - `tasks/pion_2pt/scripts/measure.py` provides a reusable API + CLI
  - `tasks/pion_2pt/benchmark/run.py` now validates, scores, and writes results
  - the interface uses `momentum_mode` (lattice units) and allows
  `lattice_spacing_fm=None`
- **Follow-ups**: refine score terms once larger ensembles and multi-source
averaging are standard in the dataset metadata.

### 2026-04-16 — `pion_2pt` default dataset path cleanup

- **Area**: `pion_2pt`
- **Type**: change / docs
- **Context**: the default dataset fallback still referenced `dataset/test_small`
  even after standardizing on the renamed local quenched ensemble directory.
- **Decision**:
  - remove `test_small` from default CLI dataset resolution in
    `benchmark/run.py` and `scripts/measure.py`
  - keep a single default local dataset path:
    `tasks/pion_2pt/dataset/quenched_wilson_b6_16x16`
  - align `.gitignore` and `dataset/README.md` with the same naming
- **Impact**: default runs are less ambiguous, docs and ignore rules now match the
  active local dataset convention, and users should pass `--dataset-path` when
  using any alternate local dataset.

### 2026-04-16 — `pion_2pt` default ensemble switched to local `8^3 x 32`

- **Area**: `pion_2pt`
- **Type**: change / dataset / docs
- **Context**: a new local quenched Wilson ensemble was provided under `/tmp/S8T32`,
  and the task default still pointed at the older `16^3 x 16` smoke-test set.
- **Decision**:
  - copy the local gauge files into
    `tasks/pion_2pt/dataset/quenched_wilson_b6_8x32`
  - add task-local metadata and README for the new ensemble
  - switch default dataset resolution in `scripts/measure.py`,
    `benchmark/run.py`, and `task.py` to `quenched_wilson_b6_8x32`
  - keep the older `quenched_wilson_b6_16x16` directory available as an alternate
    local dataset rather than the default
- **Impact**:
  - default local pion measurements now run on the `8^3 x 32` ensemble
  - docs and tests reflect the new default dataset naming
  - the new local gauge directory is ignored by git like the previous local-only
    smoke-test ensemble

### 2026-04-16 — `gsfit_2pt` benchmark pre-test gate

- **Area**: `gsfit_2pt`
- **Type**: change / bugfix
- **Context**: benchmark submission previously relied on `task.validate(...)`,
  which only checked interface/config legality and did not actually run task
  tests before scoring.
- **Decision**:
  - add a pre-benchmark pytest gate in `tasks/gsfit_2pt/benchmark/run.py` that
    runs `tasks/gsfit_2pt/tests/test_validation.py` by default
  - add `--skip-tests` as an explicit override for local experimentation
  - fail early with a clear message when the pre-test gate fails
- **Impact**:
  - default benchmark runs now execute real tests before score generation
  - invalid or regressing submissions are blocked earlier in the benchmark CLI
  - test coverage includes both default gate behavior and `--skip-tests` bypass

### 2026-04-16 — `pion_2pt` benchmark pre-test gate

- **Area**: `pion_2pt`
- **Type**: change / bugfix
- **Context**: benchmark submission only used `task.validate(...)` contract checks
  before scoring, so task tests were not executed by default in the benchmark CLI.
- **Decision**:
  - add a pre-benchmark pytest gate in `tasks/pion_2pt/benchmark/run.py` that runs
    `tasks/pion_2pt/tests/test_validation.py` by default
  - add `--skip-tests` as an explicit override for local experimentation
  - fail fast with a clear error when the pre-test gate fails
- **Impact**:
  - default `pion_2pt` benchmark runs now execute real tests before scoring
  - regressions are blocked earlier in the benchmark path
  - test coverage now validates both default gate behavior and the skip path

### 2026-04-16 — `wilson_loop` benchmark pre-test gate

- **Area**: `wilson_loop`
- **Type**: change / bugfix
- **Context**: benchmark CLI previously only checked submission validation helper
  and did not execute task tests before scoring.
- **Decision**:
  - add a pre-benchmark pytest gate in `tasks/wilson_loop/benchmark/run.py` that
    runs `tasks/wilson_loop/tests/test_validation.py` by default
  - add `--skip-tests` for explicit local override
  - fail fast with a clear error if the pre-test gate fails
- **Impact**:
  - default benchmark path now executes validation tests before score generation
  - behavior is aligned with `gsfit_2pt` and `pion_2pt`
  - tests cover both default gate execution and skip path
