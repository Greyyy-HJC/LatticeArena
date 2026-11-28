# Wilson two-lane contract diagnostic

**STRUCTURE CHECK ONLY. `physical_score=null`; no Wilson measurements, sealed data, GPU, or QUDA inputs were accessed.**

The contract preserves the user-selected lanes separately: finite-`T` Creutz as its own observable and the ground-state discrete static-force difference as a distinct target. Structural completeness is not physics validation. Even a fully populated synthetic fixture can only reach `CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW`; the tool never sets a certified or ready state.

## Text diagnostic and decision flow

```text
CURRENT NULL DRAFT
  -> identify shared action, geometry, boundaries, units, kernels, split and paired-block receipts
  -> Lane C: chi = -[log W(R+1,T+1)+log W(R,T)-log W(R+1,T)-log W(R,T+1)]; freeze primary R/T/kernel and remaining inference rules
  -> Lane F: freeze candidate-minus-reference force stencil and units, q/h/x receipts, thermal/mixed/cutoff/volume bounds, alpha allocation and total error budget
  -> schema-complete? no: remain NOT_READY and list blockers
  -> schema-complete? yes: CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW only
  -> independent physics review + authorized measurements + sealed confirmation are separate later gates
  -> physical_score remains null until those gates are independently satisfied
```

`lane_summary.csv` reports planned/valid/failed/abstained counts for each fixture and lane. A missing bound is never treated as zero. Bound rows require a scope, hash-shaped evidence receipt and review receipt; those strings are checked only for shape and are not opened or authenticated here. q/h/x receipts are dimensionless spectral inputs; the force-budget check requires their separately propagated `excited_state_force` bound and sums that with thermal/mixed/finite-a/volume/other force-level systematic bounds and the statistical precision target. Familywise alpha allocations must sum to no more than `alpha_total`.

## Fixture results

|fixture|lane|planned|valid|failed|status|certified|score|blocker count|
|---|---|---:|---:|---:|---|---|---|---:|
|current_null_draft|finite_T_Creutz|1|0|1|NOT_READY|False|None|26|
|current_null_draft|ground_state_discrete_static_force|1|0|1|NOT_READY|False|None|28|
|synthetic_C_only_complete|finite_T_Creutz|1|1|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_C_only_complete|ground_state_discrete_static_force|1|0|1|NOT_READY|False|None|14|
|synthetic_C_and_F_schema_complete|finite_T_Creutz|1|1|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_C_and_F_schema_complete|ground_state_discrete_static_force|1|1|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_C_reversed_sign|finite_T_Creutz|1|0|1|NOT_READY|False|None|1|
|synthetic_C_reversed_sign|ground_state_discrete_static_force|1|1|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_F_reversed_sign|finite_T_Creutz|1|1|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_F_reversed_sign|ground_state_discrete_static_force|1|0|1|NOT_READY|False|None|1|
|synthetic_F_wrong_precision_units|finite_T_Creutz|1|1|0|CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW|False|None|0|
|synthetic_F_wrong_precision_units|ground_state_discrete_static_force|1|0|1|NOT_READY|False|None|1|
|synthetic_null_common_kernel_ids|finite_T_Creutz|1|0|1|NOT_READY|False|None|2|
|synthetic_null_common_kernel_ids|ground_state_discrete_static_force|1|0|1|NOT_READY|False|None|3|

### What the fixtures establish

- The current draft stays `NOT_READY` and exposes unset values as blockers.
- A schema-complete synthetic C lane does not make Lane F ready.
- Synthetic C+F completeness validates stencil and budget bookkeeping only; it does not provide real Q/H/X spectra, independent bounds, action receipts, uncertainty calibration, or physics certification.
- The accepted primary `R/T/kernel`, precision/multiplicity choices and physical contamination limits remain unfrozen, so production `NOT_READY/UNCERTIFIED` is unchanged.

### Fail-closed regression diagnostics

- Reversing all four C coefficients is rejected because the declared convention is `chi = -[log W(R+1,T+1)+log W(R,T)-log W(R+1,T)-log W(R,T+1)]`, matching `finite_nt_toy.py`.
- Reversing all four F coefficients is rejected because the contrast is candidate force minus reference force.
- An F statistical precision target labeled dimensionless is rejected against the force-unit estimand and is excluded from the force-unit budget sum until units match.
- A null shared `kernel_ids` field produces explicit common blockers; the inspector does not crash or mark either lane complete.
- These are synthetic schema mutations. They diagnose contract handling only and do not read Wilson data or estimate a physical uncertainty.

## Next work

1. Review the physical definitions and freeze C's primary rectangle, sign convention, kernel family, positive-loop rule, interval and multiplicity design.
2. Freeze F's discrete-force contrast, then obtain independent per-kernel/per-R q/h/x, finite-temporal, mixed, lattice-spacing and volume evidence with a prospective error budget.
3. Populate receipts from authorized production provenance; review them independently before any sealed evaluation.
