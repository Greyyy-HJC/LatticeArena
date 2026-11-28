# Pion bounded block-risk contract diagnostic

**STRUCTURE CHECK ONLY. `physical_score=null`; no correlator, anchor, gap, validation, holdout, GPU, or QUDA inputs were read.**

The selected method defaults are recorded in `bounded_block_risk_contract_v1.json`: target `a_t m_pi`, `R=2.9`, `delta_star/R^2=0.05`, three comparisons with `alpha_score=0.04` and `alpha_anchor=0.01`, length-5 score blocks with `N=4000`, score-chain `gamma_ps>=0.20`, burn-in 8 and per-comparison TV `<=0.001`; the primary anchor route uses a separate Markov chain with `n_A=4000`, `gamma_A>=0.20`, burn-in 8, TV `<=0.001`, and a separately evidenced common-bias limit. The fixed bounded anchor scalar map remains unresolved, so no anchor interval can yet be evaluated. The 1000-unit bounded-IID Hoeffding design is conditional fallback only. All are planning choices, not measurements or receipts.

## Selected protocol flow

```text
independent candidate-blind E-chain joint fit -> pass mass/model/folding gates -> freeze bounded score and F=3 family
  -> review score-specific length-5 block-chain, gamma_ps lower bound, and burn-in/TV receipts
  -> review the independent anchor Markov-chain scalar map, gamma_ps/TV receipts, support, common-bias C, and score/anchor independence
  -> use IID Hoeffding only if 1000 truly IID stationary units and the same scalar-map/support/C bridge are independently receipted
  -> only then evaluate the full N=4000 planned-block LCB; retain failed slots as -1
  -> require normalized LCB > 0.05; otherwise ABSTAIN; physical_score remains null until all gates are separately authorized
```

## Current no-data RunState projection

The generic `RunState` projection records all five physical-certification stages as `abstained` with explicit evidence blockers in `contract_diagnostics.json`. It reports `physical_ready=false`, `score=null`, and planned/valid/failed/abstained=`5/0/0/5`. This counts missing evidence and authorization, not failed physical measurements; every future authorized run must start with a fresh state.

|stage|state|current blocker|
|---|---|---|
|operator|abstained|candidate scores are not physically interpretable until an independent m_star reference passes|
|reference|abstained|candidate-blind E-chain reference measurement and independent review receipts are missing|
|numerical|abstained|anchor feature supports and certified PP/A4P fit-image calculation are not available|
|statistical|abstained|score-chain dependence and anchor coverage/common-bias receipts are missing|
|comparison|abstained|authorized fresh confirmation inputs and score-family evaluation are not available|

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

`run_receipt.json` records these 19 synthetic contract checks plus a separate five-stage physical `RunState` denominator of `5/0/0/5` (planned/valid/failed/abstained). Each incomplete fixture and the five abstained evidence gates have explicit reasons. Receipt `complete` means all planned units are accounted for, not that any dependence, anchor, reference, coverage or physical claim succeeded. `artifact_hashes.json` covers the run receipt and all report payloads; source hashes bind the inspector, null contract, score helpers and shared receipt/state implementations.

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

## Remaining evidence receipts after method defaults were selected

- The target observable, units, score transform, `R`, clipping, `delta_star`, `F=3`, and alpha allocation are selected. The fitted numerical `m_star` remains unknown until the independent candidate-blind E-chain fit passes its fixed model/window gates and yields measurement plus review receipts.
- For the selected length-5 score block chain, provide authorized block IDs/provenance, a score-specific pseudo-gap lower-bound receipt, stationary-start or post-burn-in TV evidence, and the complete fixed score schedule. Require `F*delta < alpha_score`; do not infer independence from trajectory spacing or bootstrap count.
- For the selected anchor Markov route, provide 4000 anchor-chain states, a fixed bounded scalar map and support proof, an independently justified anchor-chain `gamma_ps` lower bound, burn-in/post-burn-in TV evidence, common-bias `C` bound, and score/anchor conditional-independence review. The 1000-unit IID Hoeffding route is an alternative only with receipt-proven IID stationary units and the same scalar-map/support/C bridge.
- For the vector route, additionally prove finite support for every PP/A4P feature, fix or simultaneously cover covariance/regularization inputs, and certify a full-box interval image of the two-state PP/A4P fit over every global minimizer. An unbounded, non-unique, boundary-hitting or unclosed fit image abstains; a vector mean box alone does not identify `m_star`.
- Complete the reference gate, anchor/common-bias review and score-chain receipts before any physical LCB. New confirmation inputs, data access and GPU/QUDA each require their corresponding owner authorization. No structural fixture or RunState receipt grants those permissions or emits a score.
