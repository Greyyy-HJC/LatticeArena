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
- **`submissions/`**: source/sink profile and Dirac-structure submissions
- **`tests/validation.py`**: shape and normalization checks
- **`benchmark/metrics.py`**: scoring by signal-to-noise and excited-state proxies

Current reference submissions:

- `plain`: `gamma_5` with point source + plane-wave sink projection
- `axial_smeared`: Gaussian momentum-smeared profile with a boost-aligned
axial-style Dirac structure

Current benchmark target:

- For ensembles without scale setting, boosted momentum is specified as a lattice
integer mode `(npx, npy, npz)`.
- The default local target is `(3, 3, 3)` (see `tasks/pion_2pt/dataset/README.md`).

The submission interface remains:

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

    def build(
        self,
        gauge_field: Any,
        momentum_mode: tuple[int, int, int],
        t_source: int,
    ) -> OperatorComponents: ...
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
  - keep the submission interface centered on `(source_profile, sink_profile, gamma_matrix)`
  while realizing arbitrary profiles via PyQUDA `colorvector` sources
- **Impact**:
  - `tasks/pion_2pt/scripts/measure.py` provides a reusable API + CLI
  - `tasks/pion_2pt/benchmark/run.py` now validates, scores, and writes results
  - the interface uses `momentum_mode` (lattice units) and allows
  `lattice_spacing_fm=None`
- **Follow-ups**: refine score terms once larger ensembles and multi-source
averaging are standard in the dataset metadata.

