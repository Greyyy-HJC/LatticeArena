# Optimize Gaussian contract diagnostic

**STRUCTURE CHECK ONLY. `physical_score=null`; no measurements, sealed data, GPU or QUDA inputs were read.**

The draft preserves both user-selected comparisons: trainable-W versus same-budget SLSQP, and trainable-W versus the fixed degree-4 Chebyshev spectral baseline. D04 reference, D06 `r_star`, D11 operator details and the production fairness budget remain open.

## Text decision flow

```text
null D04/D06/D11/budget contract
  -> require both selected comparisons and all provenance/units/constraint fields
  -> missing or malformed fields: NOT_READY with exact paths
  -> shape complete: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only
  -> independent reference + predeclared effect + operator review + authorization still required
  -> physical_score remains null throughout this inspector
```

`contract_summary.csv` uses one contract check as its unit; it is not a solver run or measurement denominator. The complete fixture contains only `SYNTHETIC_ONLY` values. It checks shape handling and cannot fill any physical decision.

|fixture|planned|valid|failed|abstained|status|blockers|
|---|---:|---:|---:|---:|---|---:|
|current_null_draft|1|0|1|0|NOT_READY|51|
|synthetic_complete_requires_review|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_missing_comparison|1|0|1|0|NOT_READY|1|
|synthetic_invalid_r_star|1|0|1|0|NOT_READY|1|
|synthetic_incomplete_D11|1|0|1|0|NOT_READY|1|

## What remains unset

- **D04:** independent physical observable/reference, units, data provenance, pass/failure rules and independent review receipt.
- **D06:** minimum meaningful effect `r_star`, units, direction and interpretation, frozen before confirmation.
- **D11:** exact degree-4 Chebyshev `H`, spectral interval and map, identity normalization, coefficient and positivity/projection constraints, failure rule and review receipt.
- **Fair budget:** same objective and split identities, initialization/restarts, per-arm objective/gradient cap, training/setup/solve/inference/I/O/recovery/total-walltime costs, hardware and amortization count.
- Production data and GPU/QUDA authorization are separate gates. No contract-complete fixture passes them.

This diagnostic is a metadata gate. It does not rank the methods, validate a physical reference, confirm `r_star`, or assert that the current Chebyshev candidate is the final baseline contract.
