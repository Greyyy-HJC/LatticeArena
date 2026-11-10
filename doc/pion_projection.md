# Pion projection: optimize a smeared sink with a fixed point source

**Status:** scientific task definition with an [executable v1 pilot](../tasks/pion_projection/README.md)
and a [first real-ensemble experiment](pion_projection_results.md). The pilot
implements fixed measurement, tuned baselines, a frozen agent attempt, and
bootstrap diagnostics. Its reference checks failed, so it has no valid official
scores or validated leaderboard. [environment.md](environment.md) records setup.
Where this document describes the broader task, the linked v1 protocol gives
the exact currently implemented restrictions and numerical settings.

## 1. Scientific objective

Hadron interpolating operators couple to the desired state and to excited states.
At large Euclidean time, unwanted states are suppressed, but statistical precision
and the available temporal extent limit useful measurements. A better operator
can make ground-state information accessible earlier and more precisely.

The task is to optimize a **sink operator**, keeping the source and measurement
procedure fixed, to improve both signal-to-noise and ground-state isolation in
the pion two-point function. These objectives need not select the same operator;
optimizing overlap alone is not a complete noise optimization strategy. See
[Detmold and Endres, Phys. Rev. D 90, 034503 (2014)](https://arxiv.org/abs/1404.6816).

Pure-gauge configurations have no sea-quark determinant. Here “pion” means the
flavor-nonsinglet pseudoscalar formed from two degenerate **valence** quarks on
that ensemble: a quenched pion, not a physical-mass, dynamical-QCD prediction.
The flavor choice avoids disconnected contractions.

## 2. Fixed problem and pilot data

| Item | Initial specification |
| --- | --- |
| Gauge ensemble | Unfixed Wilson pure-gauge `S16T16`, nominal beta = 6.0 |
| Local location | `/data/jinchen/ensembles/S16T16` |
| Geometry and format | `16^3 x 16`, NERSC, periodic gauge field |
| Configurations | `wilson_b6.0` through `wilson_b6.49`, numerically ordered |
| Valence action | Isotropic Wilson, two degenerate flavors |
| Pilot hopping parameter | `kappa = 0.14`, hence `am = 1/(2*kappa) - 4` |
| Fermion boundaries | Periodic space, antiperiodic time |
| Source | Local pseudoscalar at `(x,y,z,t) = (0,0,0,0)` |
| Spin structure and momentum | `gamma5`, zero spatial momentum |
| Inversion target | Relative residual `1e-10`; failed solves invalidate a run |

The action, mass, solver policy, configuration list, and analysis settings are
organizer-controlled. The numerical values above are starting choices for the
pilot, not a demonstrated optimum or a physical mass tuning. A single-configuration
smoke test establishes software feasibility, not ensemble-level validity.

Exclude `wilson_b6.balance`, the post-thermalization starting field. Do not use
directories with gauge-fixing suffixes such as `_cg`, `_landau`, `_itpg`, or
`_cg_ipg`. Keep PyQUDA's default NERSC checksum, plaquette, and link-trace checks
enabled. Record configuration checksums and provenance in the future manifest.
The headers inspected have repeated `SEQUENCE_NUMBER = 1`; use numeric filename
order rather than this field, and verify trajectory spacing/autocorrelation before
treating samples as independent.

## 3. What the agent may optimize

With a fixed local source `P(0)`, define

\[
O_\theta(t)=\sum_{\mathbf x,\mathbf y}
\bar d(\mathbf x,t)\gamma_5
K_\theta[U_t](\mathbf x,\mathbf y)u(\mathbf y,t),
\qquad C_\theta(t)=\langle O_\theta(t)P^\dagger(0)\rangle.
\]

`K` is a color-space kernel, scalar in spin, acting within one time slice. Its
allowed design includes spatial support, shape, paths, signed or complex weights,
nodes, multiple length scales, and linear combinations of different kernels.
Complex outputs use the complex-valued scoring extension below. In particular,
the kernel is not restricted to a Gaussian or to the positive form `H†H` obtained
by applying the same quark smearing on both legs.

The defining gauge-covariance condition is

\[
K_\theta[U^g_t](\mathbf x,\mathbf y)
=g(\mathbf x,t)K_\theta[U_t](\mathbf x,\mathbf y)g^\dagger(\mathbf y,t).
\]

Displaced quarks therefore need parallel transport; an ordinary weighted sum of
quark fields at different sites is not sufficient on an unfixed ensemble. The
kernel must be covariant under spatial translations and have the even scalar
spatial symmetry (`A1+`) so that multiplication by `gamma5` gives the pion channel.
One may start with an arbitrary seed shape and project over cubic rotations and
inversion. Zero momentum refers to summing the operator's center over space, not
to placing the sink only at the spatial origin.

The rule constructing `K` may adapt to the spatial gauge links on that time slice,
including gauge-covariant spatial link smearing, Laplacian filters, or learned
constructions. It must be the same rule with the same frozen parameters on every
configuration and time slice. It may not use temporal links, the source location,
absolute site coordinates, time separation, configuration identifiers, or other
time slices to change the rule. Translation-covariant dependence on relative
displacements is allowed.

At application time the resulting operator must be **linear in its input field**.
The candidate may not inspect the propagator to choose its weights, normalize
each configuration by its measured signal, or modify the correlator after the
fixed contraction. Training on development measurements is allowed; adaptation
to held-out measurements is not. Fixed trained parameters and random seeds form
part of the submission.

## 4. Submission and evaluator boundary

The interface has two stages (implemented by the v1 local kernel runner):

1. `build_kernel(spatial_links, parameters)` constructs a time-slice operator.
2. The evaluator invokes that operator on field columns with a common linear
   `apply(field)` operation, producing fields with the same layout and dtype.

A candidate returns neither an inversion nor a two-point estimator. The evaluator
owns the lattice layout, boundary conditions, gamma convention, and contraction.
Dense kernels need not be materialized; sparse paths and matrix-free operations
are valid. Resource limits must be fixed before a scored benchmark is released.

The trusted PyQUDA pipeline:

1. Validate and read a configuration using `pyquda_utils.io.readNERSCGauge`.
2. Construct the fixed Wilson operator with `core.getWilson` and solve all 12
   spin-color columns with `core.invert(dirac, "point", [0, 0, 0, 0])`.
3. Build the candidate kernel from spatial links and apply it at the sink.
4. Contract the flavor-nonsinglet two-point function and sum its spatial center.
5. Store configuration-level correlators and solver diagnostics, then perform
   the fixed ensemble analysis and scoring.

For degenerate Wilson quarks, gamma5 hermiticity gives the convention

\[
C_\theta[U](t)=\sum_{\mathbf x}
\operatorname{tr}_{s,c}\left[(K_\theta S)(\mathbf x,t;0)
S(\mathbf x,t;0)^\dagger\right],
\]

with the source phase chosen so that the identity kernel yields the positive
local pion correlator `sum |S|^2`. This identity is an essential normalization and
index-order check. The mixed source/sink spectral coefficients need not be
nonnegative even when the local correlator is positive.

Cache propagators by gauge checksum, action, mass, boundaries, source, solver
precision/tolerance, and relevant software versions. A sink change reuses the
point-source propagator; it does not change the Dirac operator or require another
inversion. Only links used to construct the sink may be smeared.

## 5. Score: statistical uncertainty and ground-state departure

### Reference and windows

Use lattice units throughout. Before candidate ranking, the organizer determines
a common reference mass `m0` from development data and a predeclared baseline
set, checking fit-window stability and model dependence. Candidates may not fit
their own reference mass to improve their score. Retain the reference uncertainty
and its correlations; it is an estimated quantity, not exact ground truth.

The finite-time ground-state shape is

\[
f_0(t)=e^{-m_0t}+e^{-m_0(T-t)}.
\]

The pion correlator is periodic even though individual valence quarks have
antiperiodic temporal boundaries. Fold the forward and backward measurements
per configuration, `C_fold(t) = [C(t) + C(T-t)]/2`, for `1 <= t < T/2`;
keep the midpoint once. Do not count the two directions as independent samples.
Require time-reflection symmetry of the ensemble and operator prescription for
this folding; verify it in the pilot rather than silently enforcing it on a
violating observable. Use the same operation in every resample.

For the `T = 16` pilot, propose the early scoring window `W = {2,3,4,5}` and the
disjoint late amplitude window `L = {6,7,8}`. Exclude the contact point `t = 0`.
Fit only the candidate amplitude `A_theta` on `L`, with `m0` fixed to the common
reference. Use the same prescribed correlated least-squares procedure and
covariance regularization for every candidate. Neither window is agent-tunable.

These windows require calibration: three late time points on a small ensemble
do not guarantee a controlled ground-state reference. Thermal effects or an
unresolved excited contribution can invalidate the single-cosh interpretation.
If reference stability fails, report diagnostics without a formal score; expand
statistics or temporal extent in a new benchmark version instead of changing
windows separately for individual submissions.

### Loss and baseline-normalized score

Let `Cbar_theta` be the mean folded correlator and define

\[
q_\theta(t)=\frac{\bar C_\theta(t)}{A_\theta f_0(t)},\qquad
R_\theta=\frac{1}{|W|}\sum_{t\in W}
\left[\operatorname{Var}_{\rm boot}(q_\theta(t))
+(q_\theta(t)-1)^2\right].
\]

For complex kernels, replace the squared departure by `|q - 1|^2` and use
`Var_boot(q) = mean |q* - mean(q*)|^2` (with the prescribed finite-sample
normalization). Fit a complex amplitude using the joint real/imaginary
covariance. This reduces to the displayed expression for real correlators;
discarding an imaginary component or taking `|C|` before averaging is not allowed.

The two terms have equal weight. The first penalizes uncertainty relative to the
estimated ground-state contribution, including uncertainty in its normalization.
The second measures departure from that reference shape. Using ground-state
normalization prevents a large excited-state signal from automatically looking
good simply because its relative error is small.

Given one organizer-selected reference baseline, define

\[
\mathrm{Score}_\theta=100\,
\frac{R_{\rm baseline}+10^{-12}}{R_\theta+10^{-12}}.
\]

Higher is better and the baseline scores 100. The additive floor is dimensionless
and only stabilizes a near-zero loss; it does not make an undefined `q` valid.
Use the local identity sink as the initial pilot reference. A research benchmark
must also compare with tuned strong baselines; merely scoring above 100 is not
a SOTA claim.

Multiplying an operator by a fixed nonzero scalar also multiplies its fitted
amplitude, leaving `q` and the score unchanged. A large bootstrap uncertainty
increases the loss rather than making an apparent plateau easier to accept.
The departure term is a **finite-window contamination proxy**, not a measured
excited-state probability. It includes reference-model error and statistical
fluctuations; adding its square to the variance is a conservative operational
loss, not an unbiased decomposition of physical mean-squared error. Signed
excited-state contributions can cancel, so a low loss does not prove that all
excited overlaps vanish.

### Resampling, reporting, and invalid results

Use bootstrap blocks along configuration order, choosing block length from
development-set autocorrelation checks before ranking. Preserve whole time
series and use identical resampled configuration indices for each candidate and
the baseline. Refit the late amplitude in each resample and propagate reference
mass uncertainty. If reference and evaluation data overlap in the pilot, their
resampling must be joint; with independent datasets, resample them independently.

Report both loss terms, the baseline-normalized score and its uncertainty, raw
and folded correlators, effective masses where defined, and
`SNR(t) = |Cbar(t)| / SE(Cbar(t))`. State the number of configurations and blocks:
this SNR uses the error of the mean, not the single-configuration standard
deviation. Keep invalid effective-mass points visible rather than clipping them.
Record build/application time, peak memory, inversion costs, and software versions.
The eventual score interval must resample the complete estimator, including its
variance term, rather than treating a bootstrap replica as an exact correlator.

Zero signal, a late amplitude statistically compatible with zero, unstable
normalization, nonfinite output, invalid kernel behavior, or any failed inversion
produces **no valid score**, with an explicit reason. Do not silently drop
configurations, times, or failed resamples. Amplitude-identifiability criteria,
bootstrap count, block length, fit regularization, and confidence-interval method
must be calibrated and frozen in a benchmark manifest before official evaluation.

## 6. Baselines, data separation, and research claims

Start with the identity sink, then tune Gaussian/Wuppertal smearings, sums of
smearings with signed weights, and Laplacian-based kernels on development data.
PyQUDA provides `source.gaussianSmear` and spatial Laplacian operations.
[Peardon et al., Phys. Rev. D 80, 054506 (2009)](https://arxiv.org/abs/0905.2160)
provides the distillation reference. A Laplacian kernel applied to these fixed
point-source propagators is a sink-only construction, not a requirement to build
the full distillation/perambulator pipeline.

The current 50 configurations support protocol development and smoke tests.
They are not an untouched hidden test set. Before a research leaderboard:

- Freeze the action, data manifest and independent development/validation/test
  protocol, accounting for Markov-chain correlations and any separation gaps.
- Freeze reference extraction, windows, resampling, failure rules, baseline
  tuning budgets, and per-submission runtime/memory limits.
- Freeze submissions before independent held-out evaluation; do not repeatedly
  tune against test scores. Keep training data use and evaluator data use distinct.
- Compare with tuned baselines using paired uncertainty estimates and report the
  two-objective tradeoff, reproducibility, and relevant literature comparisons.

A better pilot score is evidence for a promising design. An improvement over
SOTA additionally needs statistically supported independent validation and fair
comparisons with the applicable published methods. No such result is claimed here.

## 7. Acceptance checks for the future evaluator

- The identity kernel reproduces the local `sum |S|^2` pion contraction.
- A local gauge transformation, including at the source, leaves the correlator
  unchanged; the candidate kernel obeys the covariance identity.
- Kernel application is linear, time-local, translation-covariant, and in the
  required cubic symmetry channel; applying an overall scalar preserves score.
- Direct and cached propagation agree, and every source column satisfies the
  fixed residual tolerance and boundary conditions.
- Synthetic finite-time single-state data with increased noise worsens the
  statistical term; fixed-noise two-state data tests contamination sensitivity,
  including signed amplitudes and cancellations.
- Synthetic correlated configurations exercise blocking and paired resampling;
  reference uncertainty is propagated rather than ignored.
- Zero or unstable amplitudes, NaNs, failed solves, invalid symmetry, and
  unsupported reference fits cannot yield leaderboard entries.

The v1 tests exercise gauge covariance, linearity, Hermiticity, translations,
cubic rotations/parity, normalization invariance, synthetic statistical and
excited-state sensitivity, missing/invalid data, and a synthetic common-mass fit.
Actual S16T16 runs check all source residuals and compare the generic kernel
application with QUDA contractions. General complex non-Hermitian submissions
and a secure remote competition service remain outside the executable pilot.
