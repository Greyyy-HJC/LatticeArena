# RSI Problem: Optimization of Hadron Interpolating Operators in Lattice QCD

## 1. Scientific Objective

This project aims to enable AI agents to autonomously construct improved hadron interpolating operators in lattice QCD, achieving reduced excited-state contamination and improved signal-to-noise ratios (SNR) in hadron two-point correlation functions.

Two representative hadrons, the pion and proton, are considered across three momentum regimes:

| Hadron | Zero momentum | Low momentum | High momentum |
|---|---|---|---|
| Pion | $P_z=0$ | $P_{\rm low}$ | $P_{\rm high}$ |
| Proton | $P_z=0$ | $P_{\rm low}$ | $P_{\rm high}$ |

The specific momenta will be determined by the provided lattice ensemble.

## 2. Operator Construction

The agent may explore a general operator space:

$$
\mathcal O_h(\mathbf P,t)=
\sum_{\mathbf x,\{\mathbf r_i\},\alpha}
e^{-i\mathbf P\cdot\mathbf x}
W_\alpha(\{\mathbf r_i\},\mathbf P;U_t)
\mathcal B_{h,\alpha}(\mathbf x,\{\mathbf r_i\},t),
$$

where $\mathcal B_{h,\alpha}$ denotes operator building blocks with different spatial displacements and Dirac structures, and $W_\alpha$ represents the weights to be optimized.

Arbitrary spatial shapes, complex-valued weights, anisotropic structures, correlated quark distributions, and momentum-dependent constructions are permitted. Neither Gaussian smearing nor any particular optimization algorithm is imposed.

Physical constraints include:

- Preserving the appropriate quantum numbers and ensuring that the ground state corresponds to the target hadron.
- Accounting for reduced lattice symmetry, allowed mixing of Dirac structures at nonzero momentum, and possible opposite-parity contamination for the proton.
- Respecting the relevant gauge symmetries.
- Defining operators on individual time slices without explicit dependence on the source-sink separation.

## 3. Data Requirements

The task provider should supply one of the following:

- **Dynamical-fermion ensembles:** A fixed gauge ensemble generated with dynamical fermions, together with propagator calculation tools and computational resources.
- **Precomputed propagators:** A fixed set of point-to-all propagators computed on a specified dynamical-fermion ensemble, together with the necessary gauge links and ensemble parameters.

If arbitrary source operators cannot be constructed from the provided propagators, optimization will be restricted to sink operators or the available source basis.

All candidates must be evaluated using matched statistics and computational budgets. Multiple gauge configurations are required for reliable statistical uncertainty estimates.

## 4. Quantitative Evaluation

The hadron two-point correlation function is defined as

$$
C_{\mathcal O}(t,\mathbf P)=
\langle\mathcal O_h(\mathbf P,t)
\mathcal O_h^\dagger(\mathbf P,0)\rangle.
$$

The evaluation window $\mathcal T$ satisfies

$$
0<t_{\rm sep}<L_t/2,
$$

where $L_t$ is the temporal lattice extent. The practical evaluation window may be narrower depending on data quality and must be fixed before formal evaluation.

### Excited-State Contamination

Inspired by [arXiv:2602.02436](https://arxiv.org/abs/2602.02436), excited-state contamination is quantified through deviations of the normalized correlator from pure ground-state exponential decay:

$$
D_{\rm ES}(\mathcal O)=
\sum_{t\in\mathcal T}w_{\rm ES}(t)
\left[
1-e^{E_0^{\rm ref}(t-t_0)}
\frac{C_{\mathcal O}^{+}(t)}
{C_{\mathcal O}^{+}(t_0)}
\right],
$$

where $E_0^{\rm ref}$ is an independently determined ground-state energy, $C^+$ is the forward correlator after accounting for finite-temporal-extent effects, and the weights are normalized to unity.

For nonnegative spectral weights, smaller $D_{\rm ES}$ indicates improved ground-state projection. For asymmetric source-sink constructions, contamination can instead be quantified through the absolute excited-state contributions extracted from multi-state fits.

### Signal-to-Noise Ratio

$$
\mathrm{SNR}_{\mathcal O}(t)=
\frac{|C_{\mathcal O}(t)|}
{\sigma_{C_{\mathcal O}(t)}}.
$$

The time-weighted SNR improvement relative to the baseline is defined as

$$
R_{\rm SNR}(\mathcal O)=
\sum_{t\in\mathcal T}w_{\rm SNR}(t)
\frac{1+\mathrm{SNR}_{\mathcal O}(t)}
{1+\mathrm{SNR}_{B}(t)}.
$$

Normalized weights proportional to $t$ are proposed to emphasize signal quality at larger Euclidean time separations.

### RSI Score

$$
S(\mathcal O)=100
\left(R_{\rm SNR}\right)^\alpha
\left[
\frac{\epsilon+D_{\rm ES}(B)}
{\epsilon+D_{\rm ES}(\mathcal O)}
\right]^{1-\alpha}.
$$

The proposed initial parameters are $\alpha=0.5$ and $\epsilon=0.01$. Physically constrained spectral analysis should ensure a nonnegative contamination metric.

The baseline score is 100, and higher scores indicate better combined performance. Operators violating the physical constraints receive a score of zero. The aggregate score is the average across the six hadron-momentum conditions.

## 5. Baseline and Coulomb-Gauge Extension

Conventionally optimized Gaussian smearing and [momentum smearing](https://arxiv.org/abs/1602.05525) will serve as baseline references.

An optional extension considers the same optimization problem on Coulomb-gauge-fixed ensembles. The agent may directly explore spatially separated quark-field combinations without explicit spatial Wilson lines, while respecting residual gauge symmetry and the appropriate hadronic quantum numbers.

This variant will use a separate matched baseline and quantitative score.

## 6. Deliverables and Reproducibility

The agent should provide:

- Explicit optimized hadron operators, including spatial weights, Dirac structures, and implementation details.
- Two-point correlators, effective energies, SNR, excited-state contamination metrics, and RSI scores for both candidate and baseline operators.
- Scientific analysis of the optimized operator structures and their momentum dependence.
- Reproducible source code, parameters, random seeds, and computational resource usage.

Gauge configurations should be divided into independent training, validation, and test sets. Final scores must be evaluated on held-out configurations using consistent bootstrap or jackknife uncertainty estimation.

Appropriate contributor credits should be provided for ensemble and propagator providers, task designers, baseline developers, and other contributors.