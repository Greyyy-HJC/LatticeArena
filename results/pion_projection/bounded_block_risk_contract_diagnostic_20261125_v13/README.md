# Pion bounded block-risk contract diagnostic

**STRUCTURE CHECK ONLY. `physical_score=null`; no correlator, anchor, gap, validation, holdout, GPU, or QUDA inputs were read.**

The selected method defaults are recorded in `bounded_block_risk_contract_v1.json`: target `a_t m_pi`, `R=2.9`, `delta_star/R^2=0.05`, three comparisons with `alpha_score=0.04` and `alpha_anchor=0.01`, length-5 blocks with `N=4000`, required score-chain `gamma_ps>=0.20`, burn-in 8 and per-comparison TV `<=0.001`; the primary anchor route requires 1000 bounded iid stationary units and a separately evidenced common-bias limit. The contract also defines a conditional Markov-anchor fallback with separate chain/gap/TV and bounded scalar-map receipts. All are planning choices, not measurements or receipts.

## Selected protocol flow

```text
independent candidate-blind E-chain joint fit -> pass mass/model/folding gates -> freeze bounded score and F=3 family
  -> review score-specific length-5 block-chain, gamma_ps lower bound, and burn-in/TV receipts
  -> review independent 1000-unit bounded-IID anchor coverage and common-bias C receipt
  -> if IID is unproven, require an independent Markov anchor scalar map, its own gamma_ps/TV receipts, and target-bias bridge
  -> if IID is not evidenced, use only a separately receipted Markov anchor scalar-map/gap/TV route
  -> only then evaluate the full N=4000 planned-block LCB; retain failed slots as -1
  -> require normalized LCB > 0.05; otherwise ABSTAIN; physical_score remains null until all gates are separately authorized
```

The inspector checks whether a future contract contains the fields required by the v1 estimand: bounded paired squared-error scores, a score-specific dependence route, anchor coverage/common-bias evidence, preallocated alpha, and a separate physical reference gate.

The fixed-score Markov route follows Paulin arXiv v5 Theorem 3.11 / Eq. (3.25). The optional time-varying routes follow Eq. (3.26) for predeclared `f_i(X_i)` and `theta_N=N^-1 sum_i E_pi[f_i(X_i)]`; a human must review whether this schedule preserves the accepted target. The exact-`k_ps` route requires the smallest maximizing `k_ps`, a receipt for that choice, positive upper bounds for every residue-class variance sum, a variance receipt, and a score-schedule receipt. A second support-only route uses `Var_pi(f_i)<=1` for scores in `[-1,1]` plus a separately receipted positive pseudo-gap lower bound; it removes the exact-`k_ps` and per-class variance receipts by using `V_sum<=N+ceil(1/gamma_ps_lower)-1` and `M/k_ps=1` after uniform residue-bound inflation. This may be conservative and does not validate the gap or chain assumptions. Both routes require review of the schedule-specific finite-plan estimand. See [Paulin, arXiv:1212.2015v5](https://arxiv.org/html/1212.2015v5).

For a planned family with `F` comparisons, a post-burn-in TV bound `delta` spends `F*delta` from `alpha_score`; the remaining stationary alpha enters the Eq. (3.26) radius. The LCB subtracts `2*delta` for the bounded-score mean shift and an externally justified anchor shift in normalized score units; for the existing paired-squared-error component the inspector requires this shift to cover at least `2B/R`. Zero class-variance bounds are not accepted: a strictly positive conservative upper bound is needed to keep `M` finite.

## Text decision flow

```text
load the null v1 contract; do not load measurements
  -> check estimand, R/prediction bound, clipping, failure policy, and planned-block receipt
  -> require a score-specific block map and dependence-evidence receipt
  -> graph-color route: prove color-class independence and provide its receipt
  -> every Markov route: provide block-chain and gamma_ps-lower-bound receipts
  -> fixed-score Markov: freeze one score function; time-varying Markov: receipt the full score schedule
  -> exact-k Eq. (3.26): check k_ps/argmax receipt, positive variance upper bound per residue, variance receipt, and score-schedule receipt
  -> support-only Eq. (3.26): check [-1,1] score support, block-chain/gamma lower-bound receipts, and score-schedule receipt; exact k_ps and class-variance receipts are unnecessary
  -> for the Markov route, require stationary-start evidence or post-burn-in TV evidence and F*delta < alpha_score
  -> require matched anchor, bias/coverage/common-bias, and conditional-independence receipts
  -> check comparison family F and alpha_score + alpha_anchor <= alpha_total
  -> check independent physical reference/model-identification gate
  -> missing/malformed field: NOT_READY with exact paths
  -> complete synthetic shape: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only
  -> physical_score remains null throughout
```

For the two-sided known-sigma normal route, the random radius is `z_(1-alpha_A/2) * sigma / sqrt(n_A)`; for bounded iid Hoeffding it is `(U-L) * sqrt(log(2/alpha_A) / (2*n_A))`. The Markov anchor route applies corrected Theorem 3.11 to one fixed scalar per anchor-chain state, with worst-case variance `(U-L)^2/4`, centered bound `U-L`, and that anchor chain's own pseudo-gap. A post-burn-in TV bound spends `delta_A` from the single shared anchor alpha, so the stationary Bernstein radius uses `alpha_A-delta_A`. Each route requires total `B >= random_coverage_radius + C`, where `C` bounds common bias.

Receipts are checked for presence and SHA-256 syntax only; their scientific contents and independence are not verified. `valid` below means structurally complete, not evidence-valid, measured, or certified. Each row counts one contract check; it is not a score block or physical sample.

`run_receipt.json` records these 19 contract checks as planned/valid/failed/abstained/not_run = 19/6/13/0/0, with one blocker reason for every fixture that is not structurally complete. Its `complete` status means every planned schema check was accounted for, not that any dependence, anchor, reference, coverage or physical claim succeeded. `artifact_hashes.json` covers the run receipt and all report payloads; source hashes bind the inspector, null contract, score helpers and shared receipt implementation.

|fixture|planned|valid|failed|abstained|status|blockers|
|---|---:|---:|---:|---:|---|---:|
|current_null_contract|1|0|1|0|NOT_READY|40|
|synthetic_graph_color_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_markov_gap_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_markov_time_varying_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_markov_time_varying_support_gap_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_markov_anchor_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_markov_anchor_missing_scalar_map_receipt|1|0|1|0|NOT_READY|1|
|synthetic_markov_anchor_tv_spends_alpha|1|0|1|0|NOT_READY|1|
|synthetic_markov_post_burn_in_tv_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_alpha_over_budget|1|0|1|0|NOT_READY|1|
|synthetic_prediction_bound_over_R|1|0|1|0|NOT_READY|1|
|synthetic_missing_anchor_coverage|1|0|1|0|NOT_READY|1|
|synthetic_missing_initial_law_receipt|1|0|1|0|NOT_READY|1|
|synthetic_initial_tv_spends_score_alpha|1|0|1|0|NOT_READY|1|
|synthetic_time_varying_score_function|1|0|1|0|NOT_READY|1|
|synthetic_time_varying_missing_schedule_receipt|1|0|1|0|NOT_READY|1|
|synthetic_time_varying_zero_class_variance|1|0|1|0|NOT_READY|1|
|synthetic_time_varying_anchor_shift_under_2B_over_R|1|0|1|0|NOT_READY|1|
|synthetic_markov_missing_gamma_receipt|1|0|1|0|NOT_READY|1|

## Remaining researcher-supplied inputs

- Target `m*`, observable, interpretation and units; positive `R`, the all-included prediction-difference bound and its receipt; clipping and failure-policy approval.
- Planned block identities and score-specific dependence evidence. The graph route needs a coloring/color-class-independence receipt. Every Markov route needs block-chain assumptions, an independently reviewed `gamma_ps` lower-bound receipt, and stationary-start or post-burn-in TV evidence. Fixed-score Markov must freeze one score; Eq. (3.26) must receipt the full time-varying schedule. The exact-`k_ps` variant additionally needs the maximizing-index and residue-variance receipts; the support-only variant avoids those two inputs but not the chain/gap evidence. Freeze candidate comparison family `F`.
- For time-varying scores, predeclare and receipt the full `f_i` schedule and an anchor-induced score-shift bound in normalized units. The exact-`k_ps` route needs class-variance receipts and keeps its actual `M/k_ps`; the support-only route uses the coded `[-1,1]` range and the conservative `N+ceil(1/gamma_ps_lower)-1` variance-sum bound. Neither route proves its chain/gap assumptions.
- For the Markov route, document a stationary initial block law (`burn_in_blocks=0`, `delta=0`) or an audited post-burn-in TV bound and receipt; require `F*delta < alpha_score`.
- Matched E-anchor receipt, total anchor-error bound `B`, explicit common-bias sub-bound `C <= B`, a justified combination rule, method-specific coverage assumptions, score/anchor conditional-independence evidence, and the preallocated alpha split. For the v1 conservative combination, the inspector also checks `B >= random_coverage_radius + C`.
- Markov-anchor fallback: one fixed measurable bounded scalar per anchor-chain state, a receipt bridging its stationary mean to the chosen `m_star`, separate anchor-chain and `gamma_ps` receipts, stationary-start or post-burn-in TV receipt, and the same `B >= random_coverage_radius + C` check. The selected global multichannel fit is not automatically a per-state scalar map.
- Minimum effect `delta_star` with units and rationale, plus an independently reviewed physical reference/model-identification gate.
- Even a structurally complete contract remains subject to human evidence review and separate data/GPU authorization. No fixture changes readiness or emits a score.
