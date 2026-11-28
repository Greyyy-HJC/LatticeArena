# Pion anchor scalar reader: CPU diagnostic v1

**Classification:** synthetic CPU reader fixture only. The candidate scalar
map remains unreviewed, and `physical_score` remains null.

## Data flow

```text
captured manifest + task protocol + PP correlator NPZ
  -> same-snapshot hash links, schema, split, ID and axis checks
  -> fold each saved-state correlator evenly at T=16
  -> m_eff(4) = acosh((C_fold(3)+C_fold(5))/(2 C_fold(4)))
  -> retain one scalar or one explicit failed-state reason per planned row
  -> diagnostic-only result; no coverage or score gate
```

The reader is [`anchor_scalar_reader.py`](../anchor_scalar_reader.py). It only
parses `SourceByteSnapshot` bytes and never reopens paths. The input schema
declares one real zero-momentum PP time series per saved gauge configuration,
with ascending state IDs, `S16T16`, `kappa=0.14`, and the exact captured
`protocol.json` bytes. The manifest's protocol digest is checked against that
third snapshot source, rather than accepted as a standalone declaration. The
output binds the source bytes and a versioned map identity/hash. It rejects
complex values explicitly and never clips, repairs, or removes an invalid
state. Each planned row contributes exactly one `valid` or `failed` outcome.

## Synthetic checks

An exact periodic single-cosh fixture with `m=0.5` returns `m_eff(4)=0.5`.
For the 12-state negative fixture, eleven rows map to `0.5` and one constant
correlator maps to zero, outside the frozen `[0.1,3.0]` support. The complete
state denominator is therefore `12/11/1/0` (planned/valid/failed/abstained),
and the reported mean is null; the invalid row is not dropped. A separate
20-state positive fixture yields `20/20/0/0`. The reader's focused test file
reports **7 passed**, including captured-path replacement, split rejection,
expected-count mismatch, complex input and invalid denominator/support cases.

This tests only deterministic formula and byte-path behavior. It does not
show that the PP effective mass is unbiased for the global candidate-blind
joint PP/A4P target, establish `C/R<=0.01`, prove support for all physical
states, or validate anchor-chain `gamma_ps`, burn-in/TV, score-anchor
conditional independence, provenance, reviewer identity or authorization.
Consequently the reader cannot construct a statistical envelope accepted by
the production gate. Pion remains `NOT_READY/UNCERTIFIED`; sealed data remain
closed.
