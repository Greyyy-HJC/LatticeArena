# Optimize generic TaskSpec/RunSpec draft

**Status: `DRAFT_NO_LAUNCH`.** This is a data-free schema integration of the selected D04/D06/D11 method defaults. It contains no measurement inputs and cannot launch training; `physical_score=null`.

## What this closes

The selected research contract is now represented in LatticeArena's generic `core.specs.task_spec` and `core.specs.run_spec` objects, with a SHA-256 map for the contract and source dependencies. The task spec keeps the two agreed comparisons: trainable-W against same-budget SLSQP, and trainable-W against the fixed degree-4 Chebyshev baseline. The example Chebyshev candidate is illustrative only.

The generic `validation` range is explicitly limited to the already-open D60-D99 **development-selection** role. It is not final confirmation. Sealed roles are not included in the plan. The ranges are metadata only; no data bytes, manifests, anchor chain, gap, validation or holdout files were opened.

The source hash map binds the selected trainable-W CPU parameterization and directional features as well as the D04/D06 reducers, bounded-risk and Markov formulas, evidence adapter, snapshot reader, shared evidence gate, receipt/state models, source-integrity code, and `core/npz_safety.py`. The trainable-W implementation is still an isolated coefficient/operator prototype; its coefficient-MSE target is not the selected physical risk and it is not wired as the production training arm. Those files are copied into the report snapshots to make this distinction and code identity reviewable. The NPZ safety helper checks ZIP/NPY structure and the decompressed-byte budget before NumPy allocates arrays. The reducers use fixed selected constants, paired draw-ID checks, full planned denominators, no epsilon repair, and abstention on incomplete or invalid draws. The D06 reader consumes one explicitly captured calibration NPZ, checks its schema, split, ordered arms/times, and macroblock IDs, then derives the three-arm bounded score matrix; a declared optimizer failure receives the preselected loss 1 while remaining in the assigned-arm denominator. This CPU reader has no path fallback and rejects closed split names. D04/D11 adapters can parse their summary JSON directly from the same captured source snapshot, rejecting duplicate keys, non-finite JSON constants, mismatched caller summaries and stale canonical content digests. This only binds the summary object to captured bytes; it does not calculate D04 fits or the D11 production operator summary. None of these modules launches solvers or establishes macroblock independence.

## Text flow

```text
selected contract + trainable-W prototype/features + reducer/reader/integrity/NPZ parser sources -> SHA-256 map and source snapshots
  -> generic TaskSpec (D0-D59 fit; D60-D99 development selection)
  -> D06 snapshot NPZ -> fixed arm/time/ID checks -> bounded paired block scores
  -> D04/D11 captured JSON summary -> strict decode/content digest -> metadata gate only
  -> D04/D06 array reducers remain source-bound; production draw generation is not wired
  -> generic RunSpec with empty data-hash map
  -> missing reference/input/runner receipts => launch_allowed=false
  -> five physical RunState stages abstained => physical_score=null
```

The schema-plan denominator is planned/valid/failed/abstained=`1/1/0/0`: one in-memory spec object. It is not a run or measurement. The physical gate denominator is `5/0/0/5`. Data-input denominator is `0/0/0/0` because this diagnostic reads no data.

## Remaining implementation and evidence

This checkout does not contain `tasks/optimize_gaussian/runspec_builder.py`, a task-specific worker, or `train.main`; the plan therefore does not point to a child command and cannot become a production RunSpec. The new D06 reader is a bounded-array reader, not a solver/evaluator and not the missing task runner. The D04/D11 summary parser does not compute the raw-correlator fit or production-link operator. The next software change requires integrating this array contract with a reviewed paired-results producer and RunSpec while preserving the closed-split preflight. The open physical gates separately require the E-chain reference, D11 production-link review, D06 paired solver/cost receipts, and explicit data/GPU authorization. No gate is inferred from schema validity.
