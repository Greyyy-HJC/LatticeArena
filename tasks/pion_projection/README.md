# Executable S16T16 pion projection pilot

The scientific definition is in [the task specification](../../doc/pion_projection.md).
This directory implements a **local research pilot**, including fixed measurement,
cache verification, baseline search, a frozen agent submission, and held-out
diagnostic evaluation. It is not a secure multi-user competition service.

The first run is recorded in [the results report](../../doc/pion_projection_results.md).
The reference checks failed; the software consequently emits `score: null`.
Raw losses and uncertainty intervals are still useful diagnostics.

## Reproduce the run

From the repository root, after setting up [the environment](../../doc/environment.md):

```bash
.venv/bin/python -m pytest -q tests/test_pion_projection.py
bash scripts/run_pion.sh --output results/pion_projection/reproduction measure --split development
bash scripts/run_pion.sh --output results/pion_projection/reproduction develop
bash scripts/run_pion.sh --output results/pion_projection/reproduction measure --split validation
bash scripts/run_pion.sh --output results/pion_projection/reproduction evaluate
bash scripts/run_pion.sh --output results/pion_projection/reproduction plot
```

`--output` precedes the subcommand. Use a fresh output directory: `develop` refuses
to overwrite a frozen submission or reopen development after validation data
exist. Reproduction does not make these previously inspected configurations a
new hidden test set. `plot` uses recorded results and needs no new inversion.

Small result arrays, provenance, frozen proposals, and figures are stored under
`results/`. The approximately 6.4 GiB of point propagators for 45 configurations
live under ignored `.cache/pion_projection/propagators/`; no gauge files are copied
into the repository. Cache loading checks the gauge/software provenance, array
checksum, and, in the fixed measurement runner, all 12 explicit Dirac residuals.

## Fixed v1 protocol

`protocol.json` is the numerical source of truth and its hash accompanies data
and submissions. The protocol was fixed before the first development measurements.

| Setting | v1 value |
| --- | --- |
| Development / gap / validation IDs | 0–29 / 30–34 / 35–49 |
| Valence action | Wilson, kappa 0.14, periodic space, antiperiodic time |
| Source / residual / iteration limit | Origin, 12 spin-color columns, 1e-10, 10000 |
| Diffusion step | `H = (I + 0.2 * spatial_hopping) / 2.2` |
| Measured kernels | `H^n`, n = 0, 2, 4, 8, 16, 32, 64 |
| Score / amplitude windows | 2–5 / 6–8, folded per configuration |
| Common reference channels | n = 0, 8, 16; separate amplitudes, common mass |
| Reference windows | 4–8, 5–8 (nominal), 6–8 |
| Covariance | 25% diagonal shrinkage, eigenvalue floor 1e-10 times largest eigenvalue |
| Mass search | Bounded scalar minimization, 0.1 < am < 3.0 |
| Blocks / bootstrap | Contiguous blocks of 5; 1000 bootstrap draws |
| Score uncertainty | 200 paired outer draws, each with 128 inner draws |
| Seeds | 20261109 development; +1 validation; +2 nested intervals |
| Amplitude gate | Absolute amplitude / bootstrap SE at least 3 |
| Reference gates | Relative mass SE <= 5%; approximate fit p >= .01; paired window shifts <= 2.5 SE; no mass-bound hits |
| Reflection gate | Maximum absolute forward/backward block-mean z <= 4 over basis channels and t = 1–7 |
| Per-configuration budget | 300 seconds, checked after completion; CuPy pool limit 10 GiB |

The covariance is computed from block means. Reference fitting fixes its metric
from development; each candidate's late-amplitude metric is also fixed from its
development data before validation. Bootstrap samples vary means, reference mass,
and fitted amplitudes using these fixed metrics. Thus the intervals are conditional
on the declared fitting metrics, not a marginalization over all covariance
estimators. Shrinkage chi-square p-values are diagnostic approximations, not exact
finite-sample probabilities. The paired mass-window check uses the uncertainty of
the *difference* on common bootstrap draws.

The five-configuration block length was conservatively fixed; SEs for lengths
1, 2, 3, and 5 are reported on development. Trajectory spacing is not known and
decorrelation has not been established. Only six development blocks and three
validation blocks remain. Nested intervals include the statistical-variance term
in each outer loss; no replica is discarded. Replicas with repeated identical
blocks remain evaluable because fit metrics are fixed. These intervals are
necessarily coarse with three validation blocks.

The reference failure is a gate on scores, not a reason to search for more
favorable windows after looking at validation. Candidate failures produce an
explicit error or invalid record; incomplete measurement checkpoints cannot be
loaded as complete splits.

## Agent search and actual operator

The baseline search evaluates all seven diffusion powers on development and
selects the lowest loss. `H^(2k)` is the sink bilinear obtained by smearing both
quark legs with `H^k`; its contraction equals `||H^k S||^2`. QUDA evaluates these
norms incrementally, avoiding redundant smearing and inversions.

For the agent attempt, let `A_n` be the fitted development amplitude of basis n.
Optimize the actual linear kernel

\[
K=\sum_n \frac{u_n}{A_n}H^n,\qquad \sum_n u_n=1,\quad \sum_n |u_n|\le 8.
\]

The search minimizes development loss plus `0.001 * sum(u_n^2)` using SLSQP,
at most 500 iterations per start. The eight deterministic starts are the seven
basis vectors and their uniform average. The regularizer is a search choice and
is **not** added to the reported task loss. All starts, hyperparameters, and
weights are frozen before validation. An overall normalization has no effect on
the task score.

`submission.json` records raw coefficients, normalized coefficients, development
amplitudes, selected baseline, fitting metrics, protocol/data/source hashes, and
reference diagnostics. Applying the same coefficients to configuration-level
correlators is exact by linearity; it is not a time-dependent alteration of the
estimator. The generic kernel route below independently applies the actual
operator to propagators.

## Submit an arbitrary spatial kernel

The implemented v1 channel requires a **real Hermitian, spin-scalar** observable.
Signed kernels and negative correlators are permitted. The broader complex
non-Hermitian scoring extension in the scientific specification is not implemented
here and is rejected, rather than silently discarding a physical imaginary part.

A Python module exposes `build_kernel(spatial_links, parameters)` returning an
object with `apply(field)`. Spatial links are CuPy arrays `(3,z,y,x,3,3)`, ordered
x, y, z; a field is `(z,y,x,3,nrhs)`. The returned field has the same shape.
The evaluator provides only the current time slice's spatial links, freezes
parameters, and owns propagator loading, contraction, folding, and scoring.
The same constructor is used at every time and configuration.

The generic runner takes this JSON definition (use actual module/parameters):

```json
{
  "builder": "tasks.pion_projection.kernels:build_kernel",
  "parameters": {
    "alpha": 0.2,
    "powers": [0, 8, 32],
    "weights": [-0.2, 0.5, 0.7]
  }
}
```

```bash
bash scripts/run_pion.sh candidate --definition my_candidate.json --split development
bash scripts/run_pion.sh candidate --definition my_candidate.json --split validation
```

These commands use the complete cached propagator manifests from the main run.
Development application freezes the definition, builder hash, correlators, and
late-fit metric under `results/.../candidates/<hash>/`; validation verifies that
freeze and evaluates against the same common reference and local baseline,
including nested uncertainty estimates. User modules run as trusted local code:
kernel covariance, linearity, translation/cubic symmetry, and absence of hidden
data access remain required and subject to review/testing. This is not a Python
sandbox or an automatic proof of those properties. Time limits are cooperative
checks; the CuPy limit does not cap QUDA's separately allocated memory.

The real submitted operator for this run is in
[`agent_candidate.json`](../../results/pion_projection/s16t16_v1/agent_candidate.json).
For the diffusion family, a quick independent operator check uses the first
configuration of each split and compares the generic contraction against the
previously measured basis combination:

```bash
bash scripts/run_pion.sh candidate --definition results/pion_projection/s16t16_v1/agent_candidate.json --split development --verify-only
bash scripts/run_pion.sh candidate --definition results/pion_projection/s16t16_v1/agent_candidate.json --split validation --verify-only
```

The generic matrix-free Python route is slower than the QUDA diffusion fast path;
these checks do not retrain the proposal. Reusing the now-public validation set
for subsequent proposals must be disclosed and cannot support fresh held-out claims.
