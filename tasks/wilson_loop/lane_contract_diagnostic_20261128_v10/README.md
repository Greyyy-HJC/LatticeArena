# Wilson two-lane contract diagnostic

**STRUCTURE CHECK ONLY. `physical_score=null`; no Wilson measurements, sealed data, GPU, or QUDA inputs were accessed.**

The selected protocol is recorded in `selected_protocol()`: Lane C is absolute stout `chi(2,5)` on periodic 16^3x16 Wilson beta=6 data, with dimensionless halfwidth 0.02; Lane F is the absolute stout ground-state discrete force `U0(3)-U0(2)` at midpoint R=2.5. Identity is retained as a shared-spectrum/operator-overlap comparison, not subtracted from the primary force. Each lane receives alpha=0.025 from the total 0.05 family budget. Both remain distinct estimands. These are design choices; measurement, spectrum, thermal, split, paired-block and review receipts remain absent.

## Selected C/F decision flow

```text
paired loop IDs and 10,000 circular block-bootstrap draws (length 5; lengths 2/10 sensitivity)
  -> C: positive ensemble means -> logs -> finite-T stout chi(2,5) -> dimensionless halfwidth <=0.02
  -> F: independent 3x3 GEVP + Nt16/Nt32 joint spectrum -> q/h/x and model bounds
  -> propagate log(1+q+h+x) through the four stout Delta_R Delta_t corners -> compare force-unit systematic/statistical budget
  -> missing receipt, nonpositive stout force-scale lower bound, >1% invalid draws, or failed precision => lane-specific ABSTAIN
  -> independent review and data/GPU authorization remain separate; physical_score stays null
```

For Lane F, q/h/x bounds propagate through the four stout corners only. The identity four-corner propagation is retained as a comparison diagnostic and does not double the primary budget. The q/h/x limits are relative to a positive simultaneous lower confidence bound for the stout-force scale; they are not themselves force units. The 28 simultaneous systematic/model cells each receive alpha=0.0125/28. Statistical alpha=0.0125 is split equally between the stout-force interval and its positive scale lower-bound gate.

Structural completeness is not physics validation. Even a fully populated synthetic fixture can only reach `CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW`; the tool never sets a certified or ready state.

## Current no-data RunState projection

The separate generic `RunState` projection records all five overall physical-certification stages as `abstained`; each stage has an explicit reason in `contract_diagnostics.json`. It reports `physical_ready=false`, `score=null`, and planned/valid/failed/abstained=`5/0/0/5`. This is a snapshot of missing receipts/authorization, not five failed Wilson measurements. A future authorized calibration or confirmation must start a fresh run state.

|stage|state|current blocker|
|---|---|---|
|operator|abstained|selected kernel descriptors lack independent production-link validation|
|reference|abstained|F-lane positive stout-force scale lower-bound receipt is missing|
|numerical|abstained|authorized C-loop means and independent F-lane spectrum/thermal receipts are missing|
|statistical|abstained|paired C/F interval results and full failed-draw denominators are missing|
|comparison|abstained|fresh confirmation IDs and data/resource owner authorization are missing|

## Text diagnostic and decision flow

```text
selected protocol defaults -> acquire shared action, geometry, boundary, split and paired-block receipts
  -> Lane C: stout chi(2,5), positive-mean logs, 0.02 dimensionless precision, 97.5% interval
  -> Lane F: absolute stout ground-state force at R=2.5, with identity shared-spectrum comparison, 3x3 GEVP and matched Nt=32 thermal check
  -> populate independently reviewed q/h/x/model bounds and their force-unit propagation
  -> schema-complete? no: remain NOT_READY and list blockers
  -> schema-complete? yes: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only
  -> independent physics review + authorized measurements + sealed confirmation are separate later gates
  -> physical_score remains null until those gates are independently satisfied
```

`lane_summary.csv` reports planned/valid/failed/abstained counts for each synthetic fixture and lane. `run_receipt.json` records those 16 structural checks plus a separate five-stage physical `RunState` denominator of `5/0/0/5` (planned/valid/failed/abstained), source and artifact hashes, and exact blocker reasons. `complete` means all planned units are accounted for, not that a lane or method succeeded. `artifact_hashes.json` also covers the receipt. The C interval precision target must be labeled `dimensionless` because its signed log-Creutz observable is dimensionless. A missing bound is never treated as zero. Bound rows require a scope, hash-shaped evidence receipt and review receipt; those strings are checked only for shape and are not opened or authenticated here. q/h/x receipts are dimensionless spectral inputs; the force-budget check requires their separately propagated excited-state, thermal, mixed, and spectral-model-truncation bounds in `dimensionless lattice force a_s*a_t*F` and the statistical precision target. Since the selected force is itself a fixed-lattice/finite-volume estimand, finite-a and finite-volume extrapolation corrections are outside its scope; any continuum claim needs a separately versioned estimand. Familywise alpha allocations must sum to no more than `alpha_total`.

## Fixture results

|fixture|lane|planned|valid|failed|abstained|status|certified|score|blocker count|
|---|---|---:|---:|---:|---:|---|---|---|---:|
|current_selected_method_draft|finite_T_Creutz|1|0|1|0|NOT_READY|False|None|3|
|current_selected_method_draft|ground_state_discrete_static_force|1|0|1|0|NOT_READY|False|None|26|
|synthetic_C_only_complete|finite_T_Creutz|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_C_only_complete|ground_state_discrete_static_force|1|0|1|0|NOT_READY|False|None|25|
|synthetic_C_and_F_schema_complete|finite_T_Creutz|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_C_and_F_schema_complete|ground_state_discrete_static_force|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_C_reversed_sign|finite_T_Creutz|1|0|1|0|NOT_READY|False|None|1|
|synthetic_C_reversed_sign|ground_state_discrete_static_force|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_F_reversed_sign|finite_T_Creutz|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_F_reversed_sign|ground_state_discrete_static_force|1|0|1|0|NOT_READY|False|None|1|
|synthetic_F_wrong_precision_units|finite_T_Creutz|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_F_wrong_precision_units|ground_state_discrete_static_force|1|0|1|0|NOT_READY|False|None|1|
|synthetic_C_wrong_precision_units|finite_T_Creutz|1|0|1|0|NOT_READY|False|None|1|
|synthetic_C_wrong_precision_units|ground_state_discrete_static_force|1|1|0|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_null_common_kernel_ids|finite_T_Creutz|1|0|1|0|NOT_READY|False|None|2|
|synthetic_null_common_kernel_ids|ground_state_discrete_static_force|1|0|1|0|NOT_READY|False|None|3|

### What the fixtures establish

- The current draft stays `NOT_READY` and exposes unset values as blockers.
- A schema-complete synthetic C lane does not make Lane F ready.
- Synthetic C+F completeness validates stencil and budget bookkeeping only; it does not provide real Q/H/X spectra, independent bounds, action receipts, uncertainty calibration, or physics certification.
- Selected R/T/kernel, alpha, precision and contamination budgets are method defaults; their data, spectral and independent-review receipts remain missing, so both production lanes remain `NOT_READY/UNCERTIFIED` and the ground-state force stays uncertified.

### Fail-closed regression diagnostics

- Reversing all four C coefficients is rejected because the declared convention is `chi = -[log W(R+1,T+1)+log W(R,T)-log W(R+1,T)-log W(R,T+1)]`, matching `finite_nt_toy.py`.
- Reversing the two stout F coefficients is rejected because the primary force is U0_stout(R+1)-U0_stout(R); identity has its own two-term comparison stencil.
- A C statistical precision target labeled with energy/force units is rejected because the four-point log-Creutz estimator is dimensionless.
- An F statistical precision target labeled dimensionless is rejected against the force-unit estimand and is excluded from the force-unit budget sum until units match.
- A null shared `kernel_ids` field produces explicit common blockers; the inspector does not crash or mark either lane complete.
- These are synthetic schema mutations. They diagnose contract handling only and do not read Wilson data or estimate a physical uncertainty.

## Next work after the selected protocol defaults

1. Keep the selected C estimand fixed: obtain an independently authorized calibration manifest for stout `chi(2,5)` and preserve four paired loop means, positive-mean/log checks, failed-draw reasons, the 10,000-draw length-5 bootstrap, and the length-2/10 interval hull. Do not reuse observed D0–99 as confirmation.
2. Keep the absolute stout F estimand fixed: obtain separate calibration receipts for the R=2,3 three-operator GEVP, two/three-state and time-window diagnostics, matched Nt=32 thermal comparison, mixed-sector bound, and all force-unit propagated error cells. Propagate q/h/x and model truncation through the four stout corners; save identity-kernel results as a shared-spectrum consistency diagnostic with planned/valid/failed/abstained counts.
3. Have an independent reviewer check coverage, force-unit propagation, correlated terms and any double counting; only after calibration passes request separate authorization for fresh, unobserved confirmation IDs and GPU/QUDA resources.
4. Until these receipts and authorizations exist, retain both lane statuses as `NOT_READY/UNCERTIFIED`, keep the force lane `GROUND_STATE_FORCE_UNCERTIFIED`, do not open sealed data, and leave `physical_score=null`.
