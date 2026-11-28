# Pion bounded block-risk contract diagnostic

**STRUCTURE CHECK ONLY. `physical_score=null`; no correlator, anchor, gap, validation, holdout, GPU, or QUDA inputs were read.**

The inspector checks whether a future contract contains the fields required by the v1 estimand: bounded paired squared-error scores, a score-specific dependence route, anchor coverage/common-bias evidence, preallocated alpha, and a separate physical reference gate.

## Text decision flow

```text
load the null v1 contract; do not load measurements
  -> check estimand, R/prediction bound, clipping, failure policy, and planned-block receipt
  -> require either a score-specific graph/color receipt or block-chain gamma_ps receipt
  -> require matched anchor, bias/coverage/common-bias, and conditional-independence receipts
  -> check comparison family F and alpha_score + alpha_anchor <= alpha_total
  -> check independent physical reference/model-identification gate
  -> missing/malformed field: NOT_READY with exact paths
  -> complete synthetic shape: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only
  -> physical_score remains null throughout
```

For the two-sided known-sigma normal route, the random radius is `z_(1-alpha_A/2) * sigma / sqrt(n_A)`; for bounded iid Hoeffding it is `(U-L) * sqrt(log(2/alpha_A) / (2*n_A))`. The v1 conservative combination requires total `B >= random_coverage_radius + C`, where `C` bounds common bias.

Receipts are checked for presence and SHA-256 syntax only; their scientific contents and independence are not verified. `valid` below means structurally complete, not evidence-valid, measured, or certified. Each row counts one contract check; it is not a score block or physical sample.

|fixture|planned|valid|failed|abstained|status|blockers|
|---|---:|---:|---:|---:|---|---:|
|current_null_contract|1|0|1|0|NOT_READY|40|
|synthetic_graph_color_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_markov_gap_complete|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_alpha_over_budget|1|0|1|0|NOT_READY|1|
|synthetic_prediction_bound_over_R|1|0|1|0|NOT_READY|1|
|synthetic_missing_anchor_coverage|1|0|1|0|NOT_READY|1|

## Remaining researcher-supplied inputs

- Target `m*`, observable, interpretation and units; positive `R`, the all-included prediction-difference bound and its receipt; clipping and failure-policy approval.
- Planned block identities; score-specific graph/color proof or block-chain assumptions and `gamma_ps` lower-bound receipt; candidate comparison family `F`.
- Matched E-anchor receipt, total anchor-error bound `B`, explicit common-bias sub-bound `C <= B`, a justified combination rule, method-specific coverage assumptions, score/anchor conditional-independence evidence, and the preallocated alpha split. For the v1 conservative combination, the inspector also checks `B >= random_coverage_radius + C`.
- Minimum effect `delta_star` with units and rationale, plus an independently reviewed physical reference/model-identification gate.
- Even a structurally complete contract remains subject to human evidence review and separate data/GPU authorization. No fixture changes readiness or emits a score.
