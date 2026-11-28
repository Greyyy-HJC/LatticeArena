# Optimize candidate contract: diagnostic registration

**Status: `DIAGNOSTIC_SCHEMA_ONLY`; selected D11 method defaults are recorded, while production review and evidence remain open; `physical_score=null`; no data or GPU access.**

This package registers one degree-4 Bernstein/Chebyshev CPU candidate with the generic LatticeArena operator/candidate schema. The example controls `(0, .25, .5, .75, 1)` are illustrative; they were not fitted to a reference or measurement. The descriptor requires a spatial field with layout `(z,y,x,3,nrhs)`, periodic shifts, and checked unitary 3x3 links. Its proposed response is `P4(I+H/12)` with `0<=b0<=...<=b4=1`.

The text flow is:

```text
selected research-contract defaults -> validate five Bernstein controls -> emit candidate descriptor -> CPU operator helper
                                      |                                      |
                                      +-- invalid controls reject            +-- D04/D06 receipts, D11 review and evaluator/auth gates stay open
```

The schema fixture denominator is `planned/valid/failed/abstained = 1/1/0/0`; it counts one illustrative descriptor only. Coefficient-fit denominator is `0/0/0/0`: no D04 callback was supplied. `run_receipt.json` records the scenario assumptions, source hashes (including the selected research contract), denominator units, text flow and payload hashes. Any failed, abstained or not-run planned units require reason rows whose counts match exactly. Receipt status `complete` means all planned units are accounted for, not that a method succeeded. This adapter does not create an Optimize `TaskSpec`, connect a measurement evaluator, authorize production runs, or establish that the proposed periodic three-dimensional operator matches production conventions. The receipt itself is covered by `artifact_hashes.json` to avoid a recursive self-hash.
