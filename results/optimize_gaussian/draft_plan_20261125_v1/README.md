# Optimize generic TaskSpec/RunSpec draft

**Status: `DRAFT_NO_LAUNCH`.** This is a data-free schema integration of the selected D04/D06/D11 method defaults. It contains no measurement inputs and cannot launch training; `physical_score=null`.

## What this closes

The selected research contract is now represented in LatticeArena's generic `core.specs.task_spec` and `core.specs.run_spec` objects, with a SHA-256 map for the contract and source dependencies. The task spec keeps the two agreed comparisons: trainable-W against same-budget SLSQP, and trainable-W against the fixed degree-4 Chebyshev baseline. The example Chebyshev candidate is illustrative only.

The generic `validation` range is explicitly limited to the already-open D60-D99 **development-selection** role. It is not final confirmation. Sealed roles are not included in the plan. The ranges are metadata only; no data bytes, manifests, anchor chain, gap, validation or holdout files were opened.

## Text flow

```text
selected research contract + source files -> SHA-256 dependency map
  -> generic TaskSpec (D0-D59 fit; D60-D99 development selection)
  -> generic RunSpec with empty data-hash map
  -> missing reference/input/runner receipts => launch_allowed=false
  -> five physical RunState stages abstained => physical_score=null
```

The schema-plan denominator is planned/valid/failed/abstained=`1/1/0/0`: one in-memory spec object. It is not a run or measurement. The physical gate denominator is `5/0/0/5`. Data-input denominator is `0/0/0/0` because this diagnostic reads no data.

## Remaining implementation and evidence

This checkout does not contain `tasks/optimize_gaussian/runspec_builder.py`, a task-specific worker, or `train.main`; the plan therefore does not point to a child command and cannot become a production RunSpec. This corrects broad historical TODO wording that described those components as present. The next software change requires restoring or implementing the actual Optimize entrypoint and binding an authorized, unsealed input manifest to its hash. The open physical gates separately require the E-chain reference, D11 production-link review, D06 paired outcomes/cost ledger, and explicit data/GPU authorization. No gate is inferred from schema validity.
