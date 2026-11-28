# D-development runner replay audit

**Status:** local numerical replay completed; the PR archive is still not a standalone rerun package. The replay only used open D-development IDs 0–99 and did not open E anchor, gap, validation, confirmation, or shared holdout inputs.

## Input and source identity

The archived provenance source hashes match the recovered runners in the local root checkout:

| Runner | SHA-256 |
| --- | --- |
| `experiment.py` | `cd125c69a3b64cc37e1fbae8c2ac426f13c44003f42c606280fabc436a57a900` |
| `experiment_analysis_amendment_20261120.py` | `6ce2a238c1a717a815815d7a66cfbaaa4bf1b286e630018e10640a48e4912784` |
| `two_state_development_diagnostic.py` | `72bbd0842adc95e14519cd8c08a0ef3648994a7e02c7bf2ea59b6a1618f33371` |
| `effective_mass_development_diagnostic.py` | `03947100a7debae800b4edd42482ac4c831be227dab10e6b4ce93c4d0cad0210` |

The replay copied only `development.npz` and its matching `development_provenance.json` into a temporary directory. [`replay_protocol.json`](replay_protocol.json) now preserves the protocol object reconstructed from the open D provenance; its canonical fingerprint matches the saved measurement fingerprint `18ed89c914be7d11522256e1b159e2037cd26e89dbd6f46521a442110088f442`. The raw bytes of the original protocol JSON were not available, so the archived file is a semantic protocol reconstruction, not a byte-for-byte source file. The current working-tree protocol bytes have a different fingerprint. The temporary directory was removed after the comparison. No anchor, gap, validation, or holdout file was copied or read.

The exact 15-file Python source closure is now preserved under [`source_snapshot/`](source_snapshot/README.md), with per-file SHA-256 values and source revision in `source_manifest.json`. The active modules in the PR worktree still differ from that snapshot (the three diagnostic runners, `identity.py`, and `reference.py` are absent there, while several available dependencies have different hashes), so a reviewer must explicitly use the snapshot on `PYTHONPATH` to reproduce the archived code path. This closes the code-provenance gap but not the input/runtime gap.

## Numerical comparison

| Quantity | Archived result | Isolated replay | Assessment |
| --- | ---: | ---: | --- |
| Single-state nominal mass | 1.0961829088336696 | 1.0961829088334816 | Difference `−1.88e−13` |
| Single-state bootstrap SE | 0.00304575179818424 | 0.0030457518272125526 | Relative difference about `9.5e−9` |
| Single-state chi-square | 88.19484281629123 | 88.19484281629343 | Difference about `2.2e−12` |
| Approximate fit p-value | 1.0806540952847216e−15 | 1.0806540952836142e−15 | Same failed-fit conclusion |
| Window shift, `[4,5,6,7,8]` | 20.406117061399204σ | 20.40611824247658σ | Relative difference about `5.8e−8` |
| Window shift, `[6,7,8]` | 15.783952357018881σ | 15.783950793927055σ | Relative difference about `9.9e−8` |

The two-state summary values (masses, standard errors, fit p-values and condition numbers) agreed within `math.isclose(rel_tol=2e-9, abs_tol=2e-12)`. The effective-mass point, median, q16 and q84 summaries agreed exactly in the compared arrays (maximum absolute difference `0.0`). These are numerical summary comparisons, not byte-identical artifact reproduction: serialized JSON/NPZ hashes differed.

The decision path is unchanged:

```text
single-state reference gate FAIL
  -> two-state in-sample fit improves goodness-of-fit but retains late-window boundary/conditioning concern
  -> effective-mass time/channel curvature remains
  -> no independent anchor or physical score is certified
```

## Reproduction boundary

To make this archive independently rerunnable, provide the open D-development input bundle under a separately approved distribution path and pin the Python/NumPy/SciPy/Matplotlib runtime. Until then, retain the report/figures, exact source snapshot, and this audit as reviewable evidence, but do not call the PR archive a standalone reproduction. The replay does not change `NOT_READY/UNCERTIFIED`, `physical_score=null`, or the sealed-data boundary.
