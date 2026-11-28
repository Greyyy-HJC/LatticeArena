# Optimize Gaussian contract diagnostic

**STRUCTURE CHECK ONLY. `physical_score=null`; no measurements, sealed data, GPU or QUDA inputs were read.**

The method defaults are recorded in the contract: D04 uses the candidate-blind independent E-chain joint spectrum fit and dimensionless Pion risk; D06 uses comparator/trainable-W risk ratios with `r_star=1.111111...`, `F=2`, Bonferroni one-sided `alpha=0.025`, and paired length-5 bootstrap with lengths 2/10 as sensitivity; D11 uses a positive monotone degree-4 Bernstein polynomial mapped by a fixed dyadic rational transform (symbolic identity, float64 evaluation) to Chebyshev form, with a common unnormalized `G^2 P4(H) G^2` sandwich. The two selected contrasts remain trainable-W versus same-budget SLSQP and trainable-W versus this fixed strong spectral baseline. Independent data receipts, measured costs, production review and authorization remain open.

## Text decision flow

```text
selected D04/D06/D11 method defaults -> retain open reference, data, review, split and measured-cost receipts
  -> fit W, same-budget SLSQP and degree-4 positive spectral baseline on common fit IDs
  -> freeze outputs and evaluate both risk ratios on new paired confirmation IDs
  -> require both simultaneous ratio-hull lower bounds > 1.111111...
  -> missing or malformed fields: NOT_READY with exact paths
  -> shape complete: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only
  -> missing receipts/budget or failed reference: NOT_READY; synthetic operator loss is not D04 evidence
  -> independent review + production/GPU authorization remain separate later gates
  -> physical_score remains null throughout this inspector
```

`contract_summary.csv` uses one contract check as its unit; it is not a solver run or measurement denominator. The complete fixture contains only `SYNTHETIC_ONLY` values. It checks shape handling and cannot fill any physical decision.

|fixture|planned|valid|failed|abstained|status|blockers|
|---|---:|---:|---:|---:|---|---:|
|current_selected_method_draft|1|0|1|0|NOT_READY|19|
|synthetic_complete_requires_review|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|0|
|synthetic_missing_comparison|1|0|1|0|NOT_READY|2|
|synthetic_invalid_r_star|1|0|1|0|NOT_READY|2|
|synthetic_incomplete_D11|1|0|1|0|NOT_READY|2|

## Evidence and measured inputs still missing

- **D04:** independent E-chain correlator data receipt, fitted target value, successful reference gates, and independent review receipt.
- **D06:** no validation result is available; confirmation must use the frozen ratio direction, 10,000 paired block draws, Bonferroni allocation and interval hull.
- **D11:** independent operator-contract review receipt and real-link CPU/GPU operator evidence; the selected math constraints do not establish production-link correctness.
- **Fair budget:** fit/heldout/split/seed hashes, measured per-arm training/setup/solve/inference/I/O/recovery/total seconds, matched hardware and independent cost review. The chosen cap is 160 objective/gradient calls and amortization count is conservatively one, but cost amounts remain null.
- Production data and GPU/QUDA authorization are separate gates. No contract-complete fixture passes them.

This diagnostic records the chosen method contract and tests its schema; it does not rank the methods, establish the physical reference, verify real-link operator properties, or provide GPU timing. The target remains `NOT_READY/UNCERTIFIED`, with `physical_score=null` and sealed data closed.
