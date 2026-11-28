# Optimize candidate contract: diagnostic registration

**Status: `DIAGNOSTIC_SCHEMA_ONLY`; D11 remains unfrozen; `physical_score=null`; no data or GPU access.**

This package registers one degree-4 Bernstein/Chebyshev CPU candidate with the generic LatticeArena operator/candidate schema. The example controls `(0, .25, .5, .75, 1)` are illustrative; they were not fitted to a reference or measurement. The descriptor requires a spatial field with layout `(z,y,x,3,nrhs)`, periodic shifts, and checked unitary 3x3 links. Its proposed response is `P4(I+H/12)` with `0<=b0<=...<=b4=1`.

The text flow is:

```text
versioned diagnostic contract -> validate five Bernstein controls -> emit candidate descriptor -> CPU operator helper
                                      |                                      |
                                      +-- invalid controls reject            +-- D04/D06/D11/evaluator/auth gates stay open
```

The schema fixture denominator is `planned/valid/failed/abstained = 1/1/0/0`; it counts one illustrative descriptor only. Coefficient-fit denominator is `0/0/0/0`: no D04 callback was supplied. This adapter does not create an Optimize `TaskSpec`, connect a measurement evaluator, authorize production runs, or establish that the proposed periodic three-dimensional operator matches production conventions. All payload files are listed in `artifact_hashes.json`.
