# RSI Problem: Optimization of Hadron Interpolating Operators in Lattice QCD

## 1. Scientific Motivation

The construction of hadron interpolating operators in lattice quantum chromodynamics (lattice QCD) directly affects the statistical precision and excited-state contamination of hadronic correlation functions.

Conventional approaches typically employ predefined spatial structures, such as Gaussian smearing and momentum smearing. However, these constructions are not necessarily optimal for projecting onto hadronic ground states across different hadron species and momentum regimes.

The objective of this project is to enable an AI agent to autonomously discover and construct improved hadron interpolating operators that simultaneously achieve:

- Improved relative overlap with the target hadronic ground state;
- Reduced excited-state contamination;
- Improved signal-to-noise ratio (SNR);
- Earlier and more reliable ground-state dominance.

The evaluation will rely exclusively on hadron two-point correlation functions. Calculations of three-point correlation functions or hadronic matrix elements are not required.

## 2. Physical Problems

Two representative hadrons, the pion and the proton, will be considered in three momentum regimes:

| Hadron | Zero momentum | Low momentum | High momentum |
|---|---|---|---|
| Pion | $P_z=0$ | $P_z=P_{\rm low}$ | $P_z=P_{\rm high}$ |
| Proton | $P_z=0$ | $P_z=P_{\rm low}$ | $P_z=P_{\rm high}$ |

The specific momentum values will be determined by the spatial extent and lattice spacing of the provided ensemble:

$$
P_z=\frac{2\pi n_z}{aL_s}.
$$

For example, $n_z=0,1,4$ may be selected for the initial benchmark.

The agent may construct independently optimized operators for different hadrons and momenta or propose a unified operator construction applicable across multiple physical conditions.

## 3. Operator Construction and Physical Constraints

The agent is allowed to explore a general space of interpolating operators:

$$
\mathcal O_h(\mathbf P,t)
=
\sum_{\mathbf x}e^{-i\mathbf P\cdot\mathbf x}
\sum_{\alpha,\{\mathbf r_i\}}
W_{h,\alpha}(\{\mathbf r_i\},\mathbf P;U_t)
\mathcal B_{h,\alpha}
(\mathbf x,\{\mathbf r_i\},t).
$$

Here, $\mathcal B_{h,\alpha}$ denotes an operator building block containing the relevant quark fields, color contractions, Dirac structures, and spatial displacements. The function $W_{h,\alpha}$ represents the corresponding weights, which may depend on the gauge field on the same time slice.

The operator construction is not restricted to Gaussian smearing, spherical symmetry, separable kernels, or any predefined analytical ansatz.

Permitted constructions include:

- Arbitrary spatial shapes and complex-valued weights;
- Correlated spatial distributions among constituent quarks;
- Anisotropic smearing kernels;
- Momentum-dependent operator structures;
- Linear combinations of symmetry-compatible Dirac structures;
- Gauge-equivariant neural networks and other machine-learning-based constructions.

All proposed operators must satisfy the following physical constraints:

1. Preserve the appropriate flavor, baryon number, color-singlet structure, and relevant lattice symmetries.
2. Ensure that the ground state of interest corresponds to the target pion or proton state.
3. For gauge-unfixed ensembles, spatially separated quark fields must be combined using appropriate gauge transport or gauge-covariant constructions.
4. At nonzero momentum, account for the reduced lattice symmetry (little group) and the allowed mixing of different Dirac structures. For the proton, possible opposite-parity contamination must also be considered.
5. Define operators on individual Euclidean time slices without introducing weights that explicitly depend on the source-sink separation.

The optimization must not modify the underlying QCD action, gauge ensemble, or physical definition of the quark propagators.

## 4. Data and Computational Resources

The task provider should supply one of the following two types of input data.

### Option A: Dynamical-fermion gauge ensembles

A fixed lattice QCD gauge ensemble generated with dynamical fermions, together with:

- Lattice dimensions, lattice spacing, quark masses, and fermion action;
- Boundary conditions and configuration identifiers;
- Software and parameters required for quark propagator calculations;
- A predefined computational budget and Dirac inversion budget.

The agent may construct source and sink operators and calculate the corresponding propagators and two-point functions within the allocated computational budget.

### Option B: Precomputed point-to-all propagators

A fixed set of point-to-all quark propagators computed on a specified dynamical-fermion gauge ensemble, including the required source positions, spin/color components, quark flavors, and gauge links.

The agent will construct and optimize hadron operators using these precomputed propagators.

If the available propagators do not support arbitrary source kernels, the task will be restricted to sink operator optimization with a fixed source, or linear combinations of the available source basis, with a corresponding matched baseline.

Multiple gauge configurations are required for formal evaluation of ensemble-level statistical uncertainties and SNR. A single gauge configuration may be used for prototyping and implementation tests only.

## 5. Evaluation and Quantitative Score

The momentum-projected two-point correlation function is defined as

$$
C_{\mathcal O}(t,\mathbf P)
=
\left\langle
\mathcal O_h(\mathbf P,t)
\mathcal O_h^\dagger(\mathbf P,0)
\right\rangle.
$$

Throughout the evaluation, time and energy are expressed in lattice units. The physically relevant range of Euclidean time separations is

$$
0<t=t_{\rm sep}<\frac{L_t}{2},
$$

where $L_t$ is the temporal extent of the lattice.

In practice, the useful time window may be smaller because of statistical noise and finite-temporal-extent effects:

$$
\mathcal T=
\{t_{\min},\ldots,t_{\max}\}.
$$

For each physical condition, the evaluation window and its associated weights will be determined in advance using the development data. All candidate operators and baseline operators must be evaluated over the same time window.

### 5.1 Excited-State Contamination

The evaluation of excited-state contamination is inspired by the normalized-correlator optimization principle introduced in Section III.C of [arXiv:2602.02436](https://arxiv.org/abs/2602.02436).

For matching source and sink operators with nonnegative forward spectral weights, after accounting for backward-propagating contributions, the forward correlator has the spectral representation

$$
C_{\mathcal O}^{+}(t,\mathbf P)
=
\sum_{n=0}^{\infty}
|Z_n|^2 e^{-E_n t}.
$$

An ideal ground-state-only correlator satisfies

$$
\frac{C_{\mathcal O}^{+}(t)}
{C_{\mathcal O}^{+}(t_0)}
=
e^{-E_0(t-t_0)}.
$$

We define a dimensionless excited-state contamination indicator:

$$
D_{\mathcal O}(t)
=
1-
e^{E_0^{\rm ref}(t-t_0)}
\frac{C_{\mathcal O}^{+}(t)}
{C_{\mathcal O}^{+}(t_0)},
$$

where $E_0^{\rm ref}$ is an independently determined reference energy of the target hadronic ground state, and $t_0$ is a predefined reference time satisfying $0<t_0<t_{\min}$.

The aggregate excited-state contamination metric is

$$
D_{\rm ES}(\mathcal O)
=
\sum_{t\in\mathcal T}
w_{\rm ES}(t)D_{\mathcal O}(t),
$$

with

$$
\sum_{t\in\mathcal T}w_{\rm ES}(t)=1.
$$

Uniform weights are used by default.

Under the assumptions of a nonnegative spectral representation and a correct reference ground-state energy, a smaller value of $D_{\rm ES}$ indicates improved ground-state projection.

This metric should be estimated using a common spectral analysis procedure that accounts for statistical uncertainties, finite-temporal-extent effects, and uncertainty in the reference energy.

Asymmetric source-sink constructions should be evaluated in a separate track using the fraction of absolute excited-state contributions extracted from multi-state fits, avoiding artificial improvements caused by cancellations among excited-state contributions.

### 5.2 Signal-to-Noise Ratio

The signal-to-noise ratio at each Euclidean time separation is defined as

$$
\mathrm{SNR}_{\mathcal O}(t)
=
\frac{|\langle C_{\mathcal O}(t)\rangle|}
{\sigma_{\langle C_{\mathcal O}(t)\rangle}},
$$

where $\sigma_{\langle C_{\mathcal O}(t)\rangle}$ denotes the statistical uncertainty of the ensemble mean.

Under matched statistics and computational budgets, the time-weighted SNR improvement relative to a baseline operator $\mathcal O_B$ is defined as

$$
R_{\rm SNR}(\mathcal O)
=
\sum_{t\in\mathcal T}
w_{\rm SNR}(t)
\frac{1+\mathrm{SNR}_{\mathcal O}(t)}
{1+\mathrm{SNR}_{\mathcal O_B}(t)}.
$$

The default weighting scheme is

$$
w_{\rm SNR}(t)
=
\frac{t}{\sum_{t'\in\mathcal T}t'}.
$$

This scheme assigns greater importance to signal quality at intermediate and large Euclidean time separations, while preventing the intrinsically large SNR at short time separations from dominating the overall evaluation.

### 5.3 Final RSI Score

The combined quantitative score for each physical condition is defined as

$$
\boxed{
S(\mathcal O)
=
100
\left[R_{\rm SNR}(\mathcal O)\right]^\alpha
\left[
\frac{\epsilon+D_{\rm ES}(\mathcal O_B)}
{\epsilon+D_{\rm ES}(\mathcal O)}
\right]^{1-\alpha}
}
$$

The proposed initial parameters are

$$
\alpha=0.5,\qquad \epsilon=0.01.
$$

The regularization parameter $\epsilon$ prevents the score from diverging when excited-state contamination approaches zero.

- Baseline score = 100.
- A score greater than 100 indicates improved combined performance relative to the baseline.
- A higher score corresponds to better overall operator performance.
- Operators that fail quantum-number, ground-state energy, or statistical reliability checks receive a score of zero for the corresponding task.

The final aggregate score is the equally weighted arithmetic mean of the scores across the six physical conditions. Individual SNR and excited-state contamination metrics must also be reported separately.

All scoring parameters, spectral analysis procedures, and evaluation windows must be frozen before formal testing.

## 6. Baseline Reference

The following conventional operator constructions will serve as baseline references:

| Momentum regime | Baseline operator |
|---|---|
| Zero momentum | Gauge-covariant Gaussian smearing |
| Low momentum | Gaussian or momentum smearing |
| High momentum | Momentum smearing |
| Coulomb gauge | Coulomb-gauge Gaussian or momentum smearing |

Baseline parameters should be reasonably optimized using the development data. Baseline and candidate operators must be evaluated using matched datasets, statistical samples, and computational budgets.

References:

- G. S. Bali et al., *Novel quark smearing for hadrons with high momenta in lattice QCD*, [arXiv:1602.05525](https://arxiv.org/abs/1602.05525).
- V. Bellscheidt et al., *Wilson loops with neural networks*, [arXiv:2602.02436](https://arxiv.org/abs/2602.02436).

## 7. Optional Extension: Coulomb-Gauge Operator Optimization

The benchmark may be extended to operator optimization on gauge ensembles fixed to Coulomb gauge.

In this variant, the agent can directly explore spatially separated quark-field combinations without explicitly introducing spatial Wilson lines, provided that the construction respects residual gauge symmetry, the color-singlet condition, and the appropriate hadronic quantum numbers.

Specific questions include:

- Can Coulomb gauge fixing simplify the spatial structure of optimized hadron operators?
- Can non-Gaussian or anisotropic operator constructions improve ground-state projection at high momentum?
- How do optimized Coulomb-gauge operators compare with explicitly gauge-invariant operators?

A consistent gauge-fixing prescription must be employed throughout the ensemble. The Coulomb-gauge variant will be evaluated using a separate matched baseline and quantitative score.

## 8. Pilot Deliverables

The agent is expected to provide:

1. Explicit optimized operator constructions for each hadron and momentum condition, including spatial weights, Dirac structures, and implementation details.
2. Two-point correlation functions obtained using both optimized and baseline operators.
3. Effective energies, signal-to-noise ratios, and excited-state contamination metrics as functions of Euclidean time separation.
4. Individual scores for all six physical conditions and the aggregate RSI score.
5. A scientific assessment of the resulting operator structures, momentum dependence, and physical properties.
6. Executable source code, optimized parameters or model checkpoints, and computational resource usage.

## 9. Reproducibility and Contributor Credits

- Gauge configurations must be divided into training, validation, and held-out test sets at the configuration level.
- Operator optimization, hyperparameter selection, and baseline tuning must not use the test set.
- Final scores must be computed using frozen operators evaluated on independent test configurations.
- Statistical uncertainties must be estimated using a consistent bootstrap or jackknife procedure, accounting for gauge-configuration autocorrelations.
- Source code, execution parameters, random seeds, and dataset documentation must be made available to the task provider.
- Contributions and appropriate credits must be documented for ensemble providers, propagator providers, task designers, baseline developers, and other contributors.