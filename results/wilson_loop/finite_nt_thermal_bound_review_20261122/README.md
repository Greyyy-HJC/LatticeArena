# Wilson loop finite-`N_t` thermal contribution review (2026-11-22)

## Decision and scope

The user selected two separate estimands: the finite-loop-time Creutz observable and the ground-state discrete static force. The R/T/kernel choice, precision and multiplicity rules, and accepted spectral/thermal contamination limits remain open. This note derives the finite-`N_t` term needed by the ground-state lane; it does not read or refit Wilson-loop measurements, run Monte Carlo, choose numerical thresholds, or access D gap, validation, or shared holdout data. Current status remains `NOT_READY/UNCERTIFIED`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `physical_score=null`.

**Denominator/status:** no new configurations, fits, or synthetic datasets were run (`new planned/valid/failed/abstained = 0/0/0/0`; no execution, not an error rate). The table below is symbolic theory, not a sample-size or calibration result.

The finite-temperature transfer-matrix expression below is the standard thermal trace for a static-meson correlator. It is applied here to the rectangular Wilson loop viewed as propagation of a static pair created by a spatial transporter. The exact overlap coefficients depend on the operator and action. The cited static-force paper gives the zero-temperature Wilson-loop spectral sum; Bala et al. give the finite-temperature trace and spectral decomposition with separate no-source and static-pair Hilbert sectors in Appendix A, Eqs. (35)–(36). ([static force spectral sum](https://doi.org/10.1103/PhysRevD.105.054514); [finite-temperature spectral trace](https://arxiv.org/abs/2110.11659))

## Transfer-matrix decomposition

Let `t` be the integer temporal extent of the rectangle, `N_t` the lattice temporal extent, `a_t` the temporal lattice spacing, and `𝒯=N_t a_t`. To avoid confusing this with the bare gauge coupling commonly also named `β`, write the thermal extent as `𝒯`:

\[
W_k(R,t;N_t)=\frac{1}{Z_0(\mathcal T)}
\sum_{n\in\mathcal H_0}\sum_{m\in\mathcal H_{Q\bar Q}(R)}
e^{-(\mathcal T-ta_t)E_n^{(0)}}e^{-ta_t E_m^{(Q\bar Q,R)}}
\left|\langle n,0|O_{k,R}|m,Q\bar Q\rangle\right|^2.
\]

`n` labels states in the thermal ensemble without the static pair; `m` labels states in the sector with a pair at separation `R`; `O_{k,R}` is the spatial transporter/operator selected by kernel `k`. Set `E_0^(0)=0` and `E_0^(Q Qbar,R)=V_0(R)`. Relative to the vacuum/static-pair ground term, the leading structures have the form

\[
\begin{aligned}
W_k(R,t;N_t) &\propto e^{-a_tV_0(R)t}
\left[1+q_k(R,t)+h_k(R,N_t-t)+x_k(R,t,N_t-t)\right],\\
q_k &\sim \sum_{m>0} Q_{m,k}(R)e^{-a_t\Delta_m^{Q\bar Q}(R)t},\\
h_k &\sim \sum_{n>0} H_{n,k}(R)e^{-a_t\Delta_n^{(0)}(N_t-t)},\\
x_k &\sim \sum_{n>0,m>0} X_{nm,k}(R)
e^{-a_t\Delta_n^{(0)}(N_t-t)}e^{-a_t\Delta_m^{Q\bar Q}(R)t}.
\end{aligned}
\]

The coefficients denote overlap ratios to the leading term; they are operator dependent and are not assumed to be positive bounds unless positivity and the chosen action/operator justify that step. `q` is the familiar static-pair excited-state contamination (`Q/gap`). `h` is the thermal/winding contribution from excited no-source states propagating over the *complementary* temporal interval. `x` contains mixed terms. The partition-function normalization is independent of `R,t` and cancels from the double difference, but the numerator's thermal terms do not.

At zero temperature (`𝒯→∞` with fixed `t`), `h` and `x` vanish and the usual static-pair spectral sum remains. At fixed finite `N_t`, raising `t` suppresses `q` but shortens `N_t−t`, so it weakens the exponential suppression of `h`. A one-sided statement that “large `t` removes excited states” is therefore insufficient at finite temporal extent. Bala et al. explicitly identify Boltzmann-weighted thermal terms and terms that become important near `τ≈β`; the zero-temperature Wilson-loop spectrum and overlap interpretation are stated by Brambilla et al. in the static-force study. ([Bala et al., Appendix A](https://arxiv.org/abs/2110.11659); [Brambilla et al., Eq. (3)](https://doi.org/10.1103/PhysRevD.105.054514))

Spatial-link smearing changes the overlap coefficients `Q,H,X`, not the transfer-matrix energy levels, when the action and temporal transport are held fixed. Thus different kernels may have different finite-`t` Creutz values even though they project onto the same `V_0` in a controlled ground-state limit. This is why a paired kernel difference in Lane C is a difference of finite-time observables, not automatically a difference in error against a common ground-state truth. ([Bala et al., operator dependence discussion](https://arxiv.org/abs/2110.11659))

## How the two estimands use the same loops

For positive ensemble means, define the lattice Creutz observable with the task's sign convention:

\[
\chi_k(R,t)=-\log\frac{\bar W_k(R+1,t+1)\,\bar W_k(R,t)}
{\bar W_k(R+1,t)\,\bar W_k(R,t+1)}.
\]

If `log W = log A_k(R) − a_t V_0(R)t + δ_k(R,t;N_t)`, then exactly

\[
\chi_k(R,t)=a_t\big[V_0(R+1)-V_0(R)\big]
-\Delta_R\Delta_t\,\delta_k(R,t;N_t),
\]

where

\[
\Delta_R\Delta_t\delta=
\delta(R+1,t+1)+\delta(R,t)-\delta(R+1,t)-\delta(R,t+1).
\]

The correction `δ` can be decomposed exactly into a `Q/gap` term, a thermal term, and a mixed remainder by taking logarithms of `1+q`, `1+h`, and the residual difference. For small corrections the leading terms add, but the contract should propagate the exact log ratio through all four loop means. The constant overlap `log A_k(R)` cancels in `Δ_t`; the finite-time contamination generally does not.

This yields a clean lane split:

- **Lane C, finite-`t` Creutz:** `χ_k(R,t)` is a valid observable at the declared finite time and finite temporal extent. It does not need a ground-state interpretation to be reported as that observable. Its kernel, `R,t`, positivity/log rule, comparison family, and confidence procedure still need a predeclared contract and independent confirmation.
- **Lane F, ground-state force:** the estimator targets `a_t[V_0(R+1)−V_0(R)]`. It needs a force-level bound on the double-difference correction, with `Q/gap`, finite-`N_t` thermal, mixed, and other systematic terms kept separate. Statistical precision of `χ` alone does not supply that bound.

## `N_t=16` design implications

The archived D-chain candidate protocol declares periodic temporal boundary conditions, `N_t=16`, `R=1…4`, and `t=4…8`; that file is a preflight design artifact and does not freeze the newly selected lane's primary point. For those candidate times, the complementary extent is:

| Loop time `t` | Complement `N_t−t` | Leading excited-pair factor | Leading thermal-state factor |
|---:|---:|---|---|
| 4 | 12 | `Q_k(R) exp[−4 a_t Δ_Q(R)]` | `H_k(R) exp[−12 a_t Δ_0]` |
| 5 | 11 | `Q_k(R) exp[−5 a_t Δ_Q(R)]` | `H_k(R) exp[−11 a_t Δ_0]` |
| 6 | 10 | `Q_k(R) exp[−6 a_t Δ_Q(R)]` | `H_k(R) exp[−10 a_t Δ_0]` |
| 7 | 9 | `Q_k(R) exp[−7 a_t Δ_Q(R)]` | `H_k(R) exp[−9 a_t Δ_0]` |
| 8 | 8 | `Q_k(R) exp[−8 a_t Δ_Q(R)]` | `H_k(R) exp[−8 a_t Δ_0]` |

This table is an algebraic sensitivity map, not a numerical estimate: `Δ_Q`, `Δ_0`, and the overlaps are not supplied here. The candidate window has no point at which both exponents can be declared small without bounds on both sectors. Moving to larger `t` cannot be justified by the static-pair gap alone. Physical temperature `1/(N_t a_t)` also cannot be assigned from `N_t` alone; a scale for `a_t` is required.

## Required inputs and stop rule

| Contribution | Quantity needed for a defensible bound | What D0–99 can constrain without opening sealed data | What it cannot identify by itself |
|---|---|---|---|
| Static-pair excited states `q` | `Δ_m^(Q Qbar)(R)` and kernel-dependent overlap ratios, or a validated multi-state/variational fit with truncation control | The measured finite-`t` loop curve for the already observed operators and their covariance | A unique gap/overlap spectrum from a short, single-`N_t` time range; the reference curve cannot be treated as a certified spectrum |
| Thermal/winding states `h` | Vacuum-sector gaps `Δ_n^(0)`, degeneracies/thermal weights, transition overlaps, and boundary conditions | The combined finite-`N_t` observable and its sampling uncertainty | Separation of `h` from `q` and `x` without independent spectral or temporal-extent information |
| Mixed terms `x` and partition | Transition matrix elements in both sectors and the normalization convention | The full observed loop values | A bound from a `Q/gap`-only analysis; `Z` cancellation does not remove numerator wrap contributions |
| Force/Creutz propagation | A bound on `|Δ_RΔ_t δ|` in the same units as `a_t f_0`, with planned R/T and kernel family | The algebraic transform once the four loop means are available and valid | A tolerance value, multiplicity rule, or physical precision chosen after looking at D0–99 |

## Cross-check against the existing zero-temperature `Q/gap` bound

The earlier positive-forward-spectrum result is algebraically valid under its stated **zero-temperature** assumption. Write

\[
\rho_k(R,t)=\sum_{m>0}Q_{m,k}(R)e^{-t\widehat\Delta_m^Q(R)},\qquad
g_k(R,t)=\log\frac{1+\rho_k(R,t)}{1+\rho_k(R,t+1)},
\]

with `Q_m≥0`, `\widehat\Delta_m^Q>0`, and dimensionless lattice energies `\widehat\Delta=a_t\Delta`. Then `0≤g_k(R,t)≤log[1+Q_k(R)e^{−t\widehat\Delta_{min}(R)}]`, and the finite-time Creutz difference obeys

\[
\chi_k(R,t)-f_0(R+1/2)=g_k(R+1,t)-g_k(R,t).
\]

Because both `g` terms are nonnegative, the absolute difference is bounded by the larger of their two per-R upper bounds. This validates the older `Q/gap` expression for the zero-temperature forward spectrum; it does **not** give a finite-`N_t` bound. The added thermal `h` and mixed `x` terms can make the log correction non-monotone in `t`, so the one-sided `g≥0` argument no longer covers them.

A conservative finite-extent force-level envelope follows directly from the positive thermal trace above. Suppose independent inputs give, for each kernel and separation,

\[
q_k(R,t)\le Q_k(R)e^{-t\widehat\Delta_Q(R)},\quad
h_k(R,N_t-t)\le H_k(R)e^{-(N_t-t)\widehat\Delta_0},\quad
x_k(R,t,N_t-t)\le X_k(R)e^{-t\widehat\Delta_Q(R)-(N_t-t)\widehat\Delta_0}.
\]

Define

\[
B_k(R,t)=\log\!\left[1+Q_k(R)e^{-t\widehat\Delta_Q(R)}
+H_k(R)e^{-(N_t-t)\widehat\Delta_0}
+X_k(R)e^{-t\widehat\Delta_Q(R)-(N_t-t)\widehat\Delta_0}\right].
\]

Since `0≤δ_k(R,t)≤B_k(R,t)`, the triangle inequality gives the sufficient (conservative) bound

\[
|\chi_k(R,t)-f_0(R+1/2)|\le
B_k(R+1,t+1)+B_k(R,t)+B_k(R+1,t)+B_k(R,t+1).
\]

This expression is only conditional on those spectrum/overlap envelopes and positivity. It can be tightened if correlated differences of `δ` are independently bounded, but a bound on each loop's relative contamination alone does not automatically imply a comparably small force bias. For candidate-minus-identity, the conservative bound is the sum of the two kernels' force-level bounds. No numerical `Q,H,X,Δ` values are available in this task, so this derivation supplies a future acceptance contract, not a current pass.

**Stop rule:** if either `Q/gap` or thermal/mixed force-level contamination lacks an independent upper bound at the predeclared target tolerance, report Lane C only as the chosen finite-`t` observable (after its own protocol is frozen) and leave Lane F as `GROUND_STATE_FORCE_UNCERTIFIED`. Do not replace the missing bound with a narrow block-bootstrap interval, a stable-looking window, a high-precision finite-`t` Creutz ratio, or an assumed `exp(−gap × time)` with an unmeasured gap/overlap.

## Textual error-flow diagram

```text
periodic N_t=16 loop W_k(R,t)
  ├─ static-pair excited sector: Q_m,k(R) exp[−Δ_m^Q(R) t]
  ├─ thermal no-pair sector:     H_n,k(R) exp[−Δ_n^0 (16−t)]
  └─ mixed sector:               X_nm,k(R) exp[−Δ_n^0(16−t)−Δ_m^Q(R)t]
             │
             ├─ finite-t Creutz χ_k(R,t): retain full value as Lane C observable
             │
             └─ subtract target a_t f_0(R+1/2): propagate log terms through Δ_R Δ_t
                    ├─ independent Q/gap bound available?
                    ├─ independent thermal/mixed bound available?
                    └─ total bound ≤ prospectively frozen tolerance?
                         yes → candidate ground-state force certification gate
                         no  → GROUND_STATE_FORCE_UNCERTIFIED; physical_score=null
```

## Next actionable work under the current data seal

There is no further CPU fit or synthetic grid that can identify the missing `Δ`/overlap inputs from this one `N_t=16` design without crossing the agreed boundary. Keep the protocol open and ask for no production run. The next reviewable plan, once a physical error tolerance is selected, is to compare:

1. an independent spectrum route (a static-source operator basis with rank/mode/overlap stability, or a validated multi-state fit with complete failure counts); and
2. a matched temporal-extent route (same action, spacing, spatial volume, `R`, and operator family at larger `N_t`, with paired analysis) to quantify finite-`N_t` sensitivity.

Neither route alone automatically proves a bound; both need frozen operators, uncertainty propagation, and a failure/abstention rule. No numerical threshold is filled in this note.

## Provenance inspected

- [`ground_state_bound.py`](../../../tasks/wilson_loop/ground_state_bound.py), positive-spectrum `Q/gap` sensitivity implementation
- [`ground_state_bound_20261121_v1`](../ground_state_bound_20261121_v1/README.md), historical illustrative sensitivity report
- Research gates remain open; this is a derivation and design aid, not a task closure.
- [Static-force spectral decomposition, Phys. Rev. D 105, 054514](https://doi.org/10.1103/PhysRevD.105.054514)
- [Static quark-antiquark correlators at finite temperature, Phys. Rev. D 105, 054513](https://arxiv.org/abs/2110.11659), Appendix A, thermal trace and spectral decomposition, Eqs. (35)–(36)
