# Pion bounded block-risk contract diagnostic

**STRUCTURE CHECK ONLY. `physical_score=null`; no correlator, anchor, gap, validation, holdout, GPU, or QUDA inputs were read.**

The inspector checks whether a future contract contains the fields required by the v1 estimand: bounded paired squared-error scores, a score-specific dependence route, anchor coverage/common-bias evidence, preallocated alpha, and a separate physical reference gate.

The fixed-score Markov route follows Paulin arXiv v5 Theorem 3.11 / Eq. (3.25). The optional time-varying route follows Eq. (3.26) for predeclared `f_i(X_i)` and `theta_N=N^-1 sum_i E_pi[f_i(X_i)]`; a human must review whether this schedule preserves the accepted target. It requires the smallest maximizing `k_ps`, a receipt for that choice, positive upper bounds for every residue-class variance sum, a variance receipt, and a score-schedule receipt. Its range term retains the published `M/k_ps`; the paper says it is not known whether that factor can be replaced by 1. The inspector checks receipt presence and field arithmetic, not scientific validity. See [Paulin, arXiv:1212.2015v5](https://arxiv.org/html/1212.2015v5).

For a planned family with `F` comparisons, a post-burn-in TV bound `delta` spends `F*delta` from `alpha_score`; the remaining stationary alpha enters the Eq. (3.26) radius. The LCB subtracts `2*delta` for the bounded-score mean shift and an externally justified anchor shift in normalized score units; for the existing paired-squared-error component the inspector requires this shift to cover at least `2B/R`. Zero class-variance bounds are not accepted: a strictly positive conservative upper bound is needed to keep `M` finite.

## Text decision flow

```text
load the null v1 contract; do not load measurements
  -> check estimand, R/prediction bound, clipping, failure policy, and planned-block receipt
  -> require a score-specific graph/color receipt, fixed-score gamma_ps receipt, or Eq. (3.26) time-varying Markov contract
  -> for Eq. (3.26), check k_ps/argmax receipt, positive variance upper bound per residue, variance receipt, and score-schedule receipt
  -> for the Markov route, require stationary-start evidence or post-burn-in TV evidence and F*delta < alpha_score
  -> require matched anchor, bias/coverage/common-bias, and conditional-independence receipts
  -> check comparison family F and alpha_score + alpha_anchor <= alpha_total
  -> check independent physical reference/model-identification gate
  -> missing/malformed field: NOT_READY with exact paths
  -> complete synthetic shape: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only
  -> physical_score remains null throughout
```

For the two-sided known-sigma normal route, the random radius is `z_(1-alpha_A/2) * sigma / sqrt(n_A)`; for bounded iid Hoeffding it is `(U-L) * sqrt(log(2/alpha_A) / (2*n_A))`. The v1 conservative combination requires total `B >= random_coverage_radius + C`, where `C` bounds common bias.

Receipts are checked for presence and SHA-256 syntax only; their scientific contents and independence are not verified. `valid` below means structurally complete, not evidence-valid, measured, or certified. Each row counts one contract check; it is not a score block or physical sample.

`run_receipt.json` records these 14 contract checks as planned/valid/failed/abstained/not_run = 14/4/10/0/0, with one blocker reason for every fixture that is not structurally complete. Its `complete` status means every planned schema check was accounted for, not that any dependence, anchor, reference, coverage or physical claim succeeded. `artifact_hashes.json` covers the run receipt and all eight report payloads; source hashes bind the inspector, null contract, score helper and shared receipt implementation.

|fixture|planned|valid|failed|abstained|status|blockers|
|---|---:|---:|---:|---:|---|---:|
|current_null_contract|1|0|1|0|NOT_READY|40|
|synthetic_graph_color_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_markov_gap_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_markov_time_varying_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
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

## Remaining researcher-supplied inputs

- Target `m*`, observable, interpretation and units; positive `R`, the all-included prediction-difference bound and its receipt; clipping and failure-policy approval.
- Planned block identities; score-specific graph/color proof, fixed-score block-chain assumptions, or the time-varying route's `k_ps` maximizer and class-variance receipts; candidate comparison family `F`.
- For time-varying scores, predeclare and receipt the full `f_i` schedule, reviewed `k_ps`, positive residue-class variance-sum upper bounds, and an anchor-induced score-shift bound in normalized units. Do not assume `M/k_ps=1`.
- For the Markov route, document a stationary initial block law (`burn_in_blocks=0`, `delta=0`) or an audited post-burn-in TV bound and receipt; require `F*delta < alpha_score`.
- Matched E-anchor receipt, total anchor-error bound `B`, explicit common-bias sub-bound `C <= B`, a justified combination rule, method-specific coverage assumptions, score/anchor conditional-independence evidence, and the preallocated alpha split. For the v1 conservative combination, the inspector also checks `B >= random_coverage_radius + C`.
- Minimum effect `delta_star` with units and rationale, plus an independently reviewed physical reference/model-identification gate.
- Even a structurally complete contract remains subject to human evidence review and separate data/GPU authorization. No fixture changes readiness or emits a score.
