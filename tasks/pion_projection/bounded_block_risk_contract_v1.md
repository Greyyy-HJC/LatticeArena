# Pion bounded block-risk score contract v1

**Status: `DRAFT_NOT_READY`.** Method defaults have been selected and are recorded below and in the machine contract. The physical estimate, chain/anchor receipts, independent review and confirmation authorization remain unset. This is not a physical result or release; it reads no correlator, anchor, gap, validation or holdout data and never emits `physical_score`.

## Estimand and score map

For a predeclared candidate family and planned blocks, let `A` be the E-anchor estimate and `m*` the corresponding target value. The current candidate score component is the paired squared-error contrast

```text
D_i(A) = (A - y_reference,i)^2 - (A - y_candidate,i)^2
Z_i    = clip(D_i(A) / R^2, -1, 1)
theta_N = (1/N) sum_i E[Z_i | frozen protocol, shared anchor]
```

Positive `theta_N` favors the candidate on this **finite planned-block bounded estimand**. It is distinct from the original ensemble-risk and is not itself a physical score. The selected protocol uses `R=2.9`, clipping `[-1,1]`, and failure score `-1`; these method choices do not fill the still-missing physical or chain evidence.

Every planned block remains in `N`. A valid block uses its clipped `Z_i`; a failed block is assigned `-1`, the worst normalized improvement. With failure fraction `phi_hat`, the coded planned mean is `(1-phi_hat)*Zbar_valid - phi_hat`. An all-failed panel returns `ABSTAIN_NO_VALID_BLOCKS`; it has no lower bound.

Report two denominator levels: the block counts `(planned, valid, failed, abstained)` retain every planned block; the single comparison-result counts `(1, 1, 0, 0)` for a returned bound, or `(1, 0, 0, 1)` when all blocks fail. The latter prevents an abstained panel from disappearing inside a block-only denominator.

## Conditional calibration

Let `F` be the fixed comparison family, `chi` a proven upper bound on the chromatic number of a score-specific dependency graph, `alpha_s` the score error budget, and `alpha_A` the anchor-coverage error budget. The graph's color classes must be jointly independent conditional on the frozen protocol and shared anchor. Under that condition, for `Z_i` in `[-1,1]`:

```text
epsilon = sqrt(2 * chi * log(F / alpha_s) / N)
```

For the squared-error component,

```text
D_i(A) - D_i(m*) = 2 * (A - m*) * (y_candidate,i - y_reference,i)
```

so an anchor bound `|A-m*| <= B` yields normalized correction `2B/R` only when the predeclared prediction difference also satisfies `|y_candidate-y_reference| <= R`. If the score adds variance, fit, or other terms, each needs its own valid range and Lipschitz bound before it enters this contract.

The alternative Markov pseudo-gap route uses Paulin's corrected non-reversible Bernstein inequality (arXiv v5, Theorem 3.11 / Eq. (3.25)). It requires the same fixed bounded score function `f` on each state of a homogeneous block chain, with a stationary law and an auditable `gamma_ps` lower bound for that block chain. The structural contract therefore requires `score_function_time_homogeneous=true`; a score map that changes with block index is not covered by this helper.

For a nonstationary start, the accepted input mode is either (a) a stationary initial block law with `burn_in_blocks=0`, `delta=0`, and an initial-law receipt, or (b) an audited bound `delta >= d_TV(q P^t0, pi)` after the declared burn-in, also with a receipt. Paulin Proposition 3.15 / Eq. (3.29) adds at most `delta` to each comparison's event probability. The inspector checks that `F*delta < alpha_s`, leaving `alpha_s-F*delta` for the stationary two-sided Bernstein radius; for the operational finite-plan target it also subtracts the conservative `2*delta` mean shift allowed by the score range `[-1,1]`. It cannot verify the receipt itself.

### Selected Markov-aware anchor coverage

The selected anchor-coverage proposal applies the corrected Bernstein theorem to a **fixed bounded scalar emitted by each state of an independent anchor chain**, with `n_A=4000`, `gamma_ps>=0.20`, burn-in 8, post-burn-in `TV_A<=0.001`, `alpha_A=0.01`, and support `[0.1,3.0]`. For support `[L,U]`, use `Var(f)<= (U-L)^2/4` and centered bound `U-L`; the anchor chain needs its own independently reviewed `gamma_ps` lower bound. Proposition 3.15 / Eq. (3.29) spends `TV_A` from the one shared anchor alpha, so the stationary radius uses `alpha_A-TV_A=0.009`. The anchor interval is shared across the three candidate comparisons and therefore has family size one. With the proposed `C<=0.029`, the existing helper gives a conditional planning radius `0.585915`, total `B=0.614915`, and score correction `2B/R=0.424079`. These are formula values under the selected assumptions, not measurements or receipts. The bounded-IID Hoeffding design (`n_A=1000`) is retained only if true IID stationary units and the same support/map/bias bridge are proved.

This route does **not** make the existing global joint PP/A4P fit into a scalar per-configuration observable. The selected state-map field is therefore explicitly unresolved, and the anchor evaluator must abstain. Before implementation, a fixed state-to-scalar mapping must be defined, shown to have support `[0.1,3.0]`, and shown to estimate the selected `m*` within `C<=0.029`; otherwise the Markov anchor route is invalid. Score-chain and anchor-chain gap/TV receipts must remain separate.

The optional time-varying route follows Paulin Eq. (3.26) for `S'=sum_i f_i(X_i)` and the finite-plan estimand `theta_N=N^{-1}sum_i E_pi[f_i(X_i)]`. It does not automatically equal the fixed-score estimand above: the full `f_i` schedule and its target equivalence must be frozen and human-reviewed. It requires a receipt for the smallest maximizing `k_ps`, a strictly positive upper bound for every residue-class variance sum `V'_r>=V_r`, a variance receipt, a schedule receipt, and an externally supported normalized-score anchor-shift bound with receipt. For the existing paired-squared-error component, the inspector requires that shift bound to be at least `2B/R`; a larger schedule-specific Lipschitz correction may be needed. The implementation computes `V'_sum=sum_r V'_r`, `M'=(sum_r sqrt(V'_r))/min_r sqrt(V'_r)`, and retains `M'/k_ps`; Paulin's Remark 3.12 says it is unknown whether that factor can be replaced by 1. Receipts are checked for shape only, and a complete fixture still requires human review.

The anchor input `B` in the lower bound is the **total** bound on `|A-m*|`, including any common-mode bias. The receipt must state the random-coverage component, an explicit common-bias bound `C`, and a justified rule that combines them into `B`; the v1 conservative rule checked here requires `B >= random_coverage_radius + C`, with `B` and `C` in target units; merely hashing the same shared anchor or listing its standard error is insufficient. The selected coverage route must also expose its assumptions: known variance and a justified normal model for a normal interval; predeclared finite support and iid units for Hoeffding; or a fixed scalar map, support, and separately justified chain/gap/TV conditions for Markov Bernstein. The structural inspector checks that these fields are present and numerically consistent; it does not verify the receipts or assumptions.

The candidate lower bound is

```text
LCB = (1-phi_hat)*Zbar_valid - phi_hat - epsilon - 2B/R
```

If `A` is covered only with probability at least `1-alpha_A`, require `alpha_s + alpha_A <= alpha_total`; multiple anchor targets need a predeclared allocation. A statistical certificate would require `LCB > delta_star/R^2` and a separately passing physical reference/model-identification gate. Missing assumptions or gates yield `ABSTAIN/NOT_READY`; `physical_score` remains null in this helper.

## Inputs that must be frozen or evidenced

|Field|Current value|Required decision/evidence|
|---|---|---|
|Target `m*`, observable, physical units and interpretation|`a_t m_pi` for the lowest zero-momentum valence isovector pseudoscalar; measured value `null`|Provide the candidate-blind E-chain joint-fit receipt and independently reviewed reference/model gate|
|Scale `R` and prediction-difference bound|`R=2.9 a_t m_pi`; selected domain `[0.1,3.0]`|Provide the receipt that all included prediction pairs remain in-domain and satisfy `|y_candidate-y_reference|<=R`|
|Clipping interval and score components|selected `[-1,1]`; paired squared-error contrast only|Preserve this estimand or version any added score terms with their own range/Lipschitz bounds|
|Failure coding|selected failed-block score `-1`; all-failed comparison abstains|Receipt the failure reasons and preserve planned, valid, failed and abstained denominators|
|Planned block count `N` and block identities|selected `N=4000`, length-5 blocks; IDs/receipt `null`|Provide score-specific block identities and complete per-comparison denominators|
|Result unit and panel-level denominator|one candidate-reference comparison|Count an all-failed comparison as abstained while retaining its failed blocks in the block denominator|
|Dependence route|selected fixed-score Markov pseudo-gap Bernstein; required `gamma_ps>=0.20`, burn-in 8 blocks, post-burn-in `TV<=0.001`; receipts `null`|Supply score-block chain assumptions, independent gamma lower-bound and TV receipts; for nonstationary starts ensure `F*delta < alpha_s`. Eq. (3.26) remains an optional distinct route requiring schedule/variance/`k_ps` receipts|
|Candidate family `F`|selected three comparisons: candidate vs local identity, tuned `H^32`, and frozen signed-SLSQP|Preserve all three comparison IDs in simultaneous inference|
|Anchor estimate `A`, bound `B`, coverage and independence|primary selected route: Markov Bernstein, `n_A=4000`, `gamma_A>=0.20`, burn-in 8, `TV_A<=0.001`, support `[0.1,3.0]`, `alpha_A=0.01`, `C/R<=0.01`; receipts `null`. IID Hoeffding `n_A=1000` is conditional fallback only|Provide matched E-correlator receipt, total `B`, explicit common-bias `C<=B`, coverage and conditional-independence receipts. Primary route additionally needs one fixed bounded scalar per anchor-chain state, its target-map/support receipt, separate anchor-chain/gamma/TV receipts, and reviewed bias bridge to the global fit|
|`alpha_total`, `alpha_s`, `alpha_A`|selected `0.05`, `0.04`, `0.01`|Preserve `alpha_s+alpha_A<=alpha_total` and the shared-anchor interval allocation|
|Minimum effect `delta_star` and reference gate|selected `delta_star=0.05 R^2=0.4205 (a_t m_pi)^2`; reference gate receipts `null`|Maintain the independent reference/model-identification gate; do not treat threshold selection as observed effect|

## Text decision flow

```text
freeze score and planned blocks
  -> verify graph-color evidence, fixed-score Markov assumptions, or the optional Eq. (3.26) schedule/variance/k_ps contract
  -> for Markov TV mode, require F*delta < alpha_s before computing the radius
  -> verify the selected Markov-anchor scalar-map/support/gap/TV/bias contract (use IID only with proof of true independent stationary units)
  -> compute coded planned mean - graph penalty - anchor correction
  -> compare LCB with delta_star / R^2
  -> separately require the physical reference/model-identification gate
  -> otherwise ABSTAIN / NOT_READY; physical_score remains null
```

Current analytic, Monte Carlo and exact-enumeration reports test formula behavior under declared synthetic assumptions only. They do not fill any `null` field above. In particular, trajectory spacing is not an independence proof, a bootstrap interval is not a finite-sample dependence theorem, and an assumed anchor support is not an E-anchor receipt.
