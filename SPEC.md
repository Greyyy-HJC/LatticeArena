# LatticeArena Specification

## 1. Overview

LatticeArena is a multi-task benchmark where each **task** isolates a specific lattice QCD optimization problem. Contributors submit optimized implementations through a well-defined interface. Submissions are validated by automated tests and ranked by benchmark metrics on a public leaderboard.

## 2. Task Structure

Every task lives under `tasks/<task_name>/` and contains:

| Directory/File | Purpose |
|---|---|
| `dataset/` | Input data (gauge ensembles, propagators, etc.). Large files are gitignored; `README.md` explains how to obtain them. |
| `scripts/` | Runnable scripts for the full measurement pipeline using PyQUDA. These implement the physics — loading configs, building observables, computing correlators. |
| `interface.py` | Defines the abstract base class (ABC) that contributors must implement. This is the **optimization target**. |
| `operators/` | Submitted implementations of the interface. Contributors only modify files here. |
| `tests/` | Validation tests. A submission is **legal** if and only if it passes all tests. |
| `benchmark/` | Metrics computation and scoring. Produces a numerical score for the leaderboard. |

### Submission Rules

1. A submission is a single Python file in `tasks/<task_name>/operators/` implementing the task's interface ABC.
2. The submission must pass all tests in `tasks/<task_name>/tests/`.
3. The submission is scored by running `tasks/<task_name>/benchmark/run.py`.
4. Higher score = better operator. The leaderboard ranks by score descending.
5. Submissions may use numpy, scipy, and PyQUDA. No additional dependencies without approval.

---

## 3. Task: `wilson_loop`

### 3.1 Physics Background

The Wilson loop `W_{r x t}` is a rectangular closed loop of gauge links with spatial extent `r` and temporal extent `t`. Its expectation value encodes the static quark-antiquark potential:

```
<tr W_{r x t}> = sum_n |c_n|^2 exp(-t * a * E_n(r))
```

where `E_n(r)` are static energies and `c_n` are overlap coefficients with the spatial operator. The ground-state energy `E_0(r)` gives the static potential.

**The problem**: The straight Wilson line has poor ground-state overlap `|c_0|^2`, causing:
- Excited-state contamination at small `t`
- Signal-to-noise degradation at large `t`

**The optimization target**: Design a spatial operator `S_hat(x, x+r, t)` that replaces the straight Wilson line, maximizing `|c_0|^2` while maintaining gauge covariance.

Reference: arXiv:2602.02436 (Bellscheidt et al., "Wilson loops with neural networks")

### 3.2 Dataset

Pure gauge (quenched) SU(3) lattice ensembles generated with the Wilson gauge action.

| Ensemble | Lattice | beta | Configs | Purpose |
|---|---|---|---|---|
| `test_small` | 8^3 x 16 | 5.8 | 100 | Fast validation and development |
| `benchmark` | 20^3 x 40 | 6.0 | 4000 | Official benchmark scoring |

See `tasks/wilson_loop/dataset/README.md` for download instructions.

### 3.3 Optimization Interface

File: `tasks/wilson_loop/interface.py`

```python
class SpatialOperator(ABC):
    """
    Replaces the spatial Wilson line S(x, x+r*e_dir, t) in the Wilson loop.

    The output must transform under gauge transformations as:
        S_hat(x, r, dir, t) -> G(x,t) @ S_hat(x, r, dir, t) @ G^dag(x+r*e_dir, t)

    This is the same transformation property as the straight Wilson line (product of links).
    """

    @abstractmethod
    def setup(self, gauge_field: np.ndarray, latt_size: tuple[int, int, int, int]) -> None:
        """One-time setup per gauge configuration. Use for precomputation, gauge fixing, etc."""
        ...

    @abstractmethod
    def compute(self, gauge_field: np.ndarray, r: int, direction: int, t: int) -> np.ndarray:
        """
        Compute the spatial operator for all spatial sites on timeslice t.

        Args:
            gauge_field: Full gauge field, shape (Nd, Lx, Ly, Lz, Lt, Nc, Nc) complex128
            r: Spatial separation in lattice units
            direction: Spatial direction (0=x, 1=y, 2=z)
            t: Timeslice index

        Returns:
            np.ndarray of shape (Lx, Ly, Lz, Nc, Nc) complex128
            Color matrix at each spatial site.
        """
        ...
```

**Key points**:
- `gauge_field` is provided in natural (lexicographic) ordering with shape `(Nd, Lx, Ly, Lz, Lt, Nc, Nc)`. The framework handles conversion from PyQUDA's internal even/odd format.
- `setup()` is called once per gauge configuration. Use it for expensive one-time operations (gauge fixing, precomputing smeared links, etc.).
- `compute()` is called for each `(r, direction, t)` combination. It must return the spatial operator at all spatial sites simultaneously.
- `Nc = 3` for SU(3). `Nd = 4` for 4 spacetime directions (0=x, 1=y, 2=z, 3=t).

### 3.4 Validation Tests

Tests in `tasks/wilson_loop/tests/test_validity.py`:

| Test | Description |
|---|---|
| **Gauge equivariance** | Apply random gauge transform `G(x)` to the config. Verify `S_hat` transforms as `G(x) @ S_hat @ G^dag(x+r*e_dir)`. This is the most critical physics constraint. |
| **Output shape** | `compute()` returns `(Lx, Ly, Lz, 3, 3)` complex array for every valid `(r, direction, t)`. |
| **Cold-config identity** | On a unit gauge (all links = identity), the operator should return the identity matrix. |

Tests run on a small lattice (4^3 x 8) for speed.

### 3.5 Benchmark Metrics

Metrics in `tasks/wilson_loop/benchmark/metrics.py`:

| Metric | Formula | Meaning |
|---|---|---|
| **Effective mass** | `m_eff(t) = -ln(C(t+1) / C(t))` | Should plateau at the ground-state energy. Earlier/flatter plateau = better operator. |
| **Signal-to-noise** | `mean(C) / std(C)` over configs | Higher = cleaner signal at large `t`. |
| **Plateau quality** | `chi^2 / dof` of constant fit to `m_eff` in `[t_min, t_max]` | Lower = flatter plateau = less excited-state contamination. |
| **Ground-state overlap** | `\|c_0\|^2` extracted from `C(t) / C(0)` fit | Higher = better projection onto ground state. |

The **composite score** combines these metrics (weighting TBD) into a single number for leaderboard ranking.

### 3.6 Measurement Pipeline

`tasks/wilson_loop/scripts/measure.py` implements:

1. Load gauge configuration from `dataset/`
2. Convert to natural ordering `(Nd, Lx, Ly, Lz, Lt, Nc, Nc)`
3. Call `operator.setup(gauge_field, latt_size)`
4. For each `(r, direction, t_source)`:
   - Compute spatial operator `S = operator.compute(gauge_field, r, direction, t_source)`
   - Build temporal Wilson lines `T(t_source, t_source+t, x)` by multiplying temporal links
   - Form Wilson loop: `W = S(x, t0) @ T(x+r, t0->t0+t) @ S^dag(x, t0+t) @ T^dag(x, t0->t0+t)`
   - Take trace, average over spatial sites `x`, directions, and source times `t0`
5. Output `C(r, t)` correlator array

## 3.7 Task: `pion_2pt`

### 3.7.1 Physics Background

The pion two-point correlator at nonzero momentum is

```
C_pi(p, t) = < O_pi(p, t0+t) O_pi^dag(p, t0) >
```

with

```
O_pi(p, t) = sum_x exp(i p.x) dbar(x,t) Gamma u(x,t).
```

The optimization target is the interpolating operator design (smearing profile,
Dirac structure, momentum injection strategy) for boosted pions, especially
for momentum scales around |p| ~ 1 GeV.

Desired outcomes:
- Higher signal-to-noise at moderate/large Euclidean time
- Earlier and flatter effective-mass plateau
- Reduced excited-state contamination in multi-state fits

### 3.7.2 Optimization Interface

File: `tasks/pion_2pt/interface.py`

Contributors implement `PionInterpolatingOperator` with:
- `setup(gauge_field, latt_size, lattice_spacing_fm)`
- `build(gauge_field, momentum_gev, t_source)`

`build(...)` returns source/sink spatial profiles and a Dirac matrix to define
the bilinear used in the 2pt contraction.

## 3.8 Task: `two_pt_gsfit`

### 3.8.1 Physics Background

This task targets the **analysis stage** of pion two-point correlators rather
than operator construction. Given bootstrap/jackknife-like correlator samples,
the goal is to choose a robust fixed configuration for extracting the
ground-state energy `E0`.

The framework-owned fit model is

```
C(t) = sum_n A_n [exp(-E_n t) + exp(-E_n (Lt - t))]
E_n = E0 + sum_{k <= n} DeltaE_k
```

with positivity constraints enforced internally:

- `A_n > 0`
- `E0 > 0`
- `DeltaE_n > 0`

### 3.8.2 Optimization Interface

File: `tasks/two_pt_gsfit/interface.py`

Contributors implement `Pion2PtGroundStateFit` and only submit one fixed
configuration:

- `t_min`, `t_max`
- `n_states`
- `e0_prior`
- `delta_e_priors`
- `amplitude_priors`

They do **not** implement a custom fitter.

### 3.8.3 Benchmark

The v1 benchmark uses deterministic synthetic pion correlator samples with
known truth instead of committed real datasets. The benchmark:

1. generates several pion-like cases with different noise levels and
   excited-state contamination
2. fits the sample mean correlator with a correlated Bayesian fit
3. refits resamples to estimate the uncertainty and failure rate
4. scores submissions by prioritizing:
   - low `E0` bias
   - reasonable uncertainty
   - acceptable `chi2/dof`
   - robustness across resamples

---

## 4. Framework Library

### `latticearena/task.py`

```python
class TaskBase(ABC):
    """Base class for all benchmark tasks."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def validate(self, operator) -> bool:
        """Run validation tests. Returns True if all pass."""
        ...

    @abstractmethod
    def benchmark(self, operator, dataset_path: str) -> dict:
        """Run benchmark. Returns dict with 'score' and individual metrics."""
        ...
```

### `latticearena/leaderboard.py`

Reads benchmark results from `tasks/<name>/benchmark/results/`, ranks submissions by score, outputs a summary table.

---

## 5. Adding a New Task

To add a new benchmark task (e.g., `glueball_spectrum`):

1. Create `tasks/glueball_spectrum/` with the standard structure
2. Define the optimization interface in `interface.py`
3. Write validation tests in `tests/`
4. Implement benchmark metrics in `benchmark/`
5. Provide measurement scripts in `scripts/`
6. Document the dataset in `dataset/README.md`
7. Submit a baseline operator in `operators/`
