# Wilson stout kernel identity crosswalk

## Decision

The literal kernel names differ, but the retrospective C-lane development receipt and the selected protocol identify the same implemented spatial stout operator at the source-and-parameter level:

```text
selected protocol role: spatial_stout_rho0.10_n5
development receipt name: stout_rho0.1_n5
normalized operator signature: (kind=stout, rho=0.1, steps=5, domain=spatial links)
```

This resolves the naming ambiguity for the archived D0–99 development preflight. It does **not** provide an independent production-link validation, fresh confirmation, or physical certification. The Wilson contract therefore stays `DRAFT_NOT_READY`, both lanes remain uncertified, and `physical_score` remains null.

## Evidence checked

| Planned check | Observed evidence | Result |
|---|---|---|
| Selected C-lane operator | `tasks/wilson_loop/lane_contract_v2.json` sets `primary_kernel` and C-lane `kernel_id` to `spatial_stout_rho0.10_n5`. | Resolved |
| Development receipt descriptor | `tasks/wilson_loop/c_lane_open_development_diagnostic_20261125_v1/protocol.json` records `stout_rho0.1_n5` with `{kind: stout, rho: 0.1, steps: 5}`; the pinned source verifies exactly one protocol entry with those parameters. | Resolved |
| Stout implementation identity | The protocol's pinned `stout.py` SHA-256 is `7d9f5f8ea0c0e006b09f58abc087e5825fd74c5efecf085db176c7c23ed69f35`. It matches the Git blob at `54f0e97ba85b7f8f724c31d9f9d3512f040583a9:tasks/wilson_loop/stout.py`. | Resolved |
| Dispatch implementation identity | The pinned `kernels.py` SHA-256 is `a6b54eb606f1b7c790de596185d8c0190601fde4495d9f26eba89ba16abdd7dc`. It matches the Git blob at `54f0e97ba85b7f8f724c31d9f9d3512f040583a9:tasks/wilson_loop/kernels.py`. | Resolved |
| Does the display name alter the operator? | `build_kernel` accepts `name` as an allowed metadata field, then dispatches by `kind` and passes only `rho` and `steps` to `stout`. | Resolved |
| Spatial convention | The source accepts links shaped `(3,t,z,y,x,3,3)`, updates the three spatial link directions synchronously, and does not accept or modify temporal links. The weak-field relation `rho ~= alpha/4` is documented as an approximation, not an exact smoothing-radius equivalence. | Resolved |

Static audit denominator: planned 6 checks, 6 resolved, 0 failed, 0 abstained. The check is reproducible from the contract JSON, the development protocol/manifest, and the two pinned source blobs. The manifest marks the package `RETROSPECTIVE_DEVELOPMENT_ONLY_NOT_CONFIRMATION`.

## Interpretation and remaining gate

The two strings are not claimed to be literal aliases in the application. They normalize to the same structured operator descriptor, and the run's pinned implementation dispatch ignores the descriptive `name`. Thus the historical C preflight used the selected stout algorithm, parameter, and spatial-link convention.

The separate production gate remains open: no independent source/action/kernel receipt validates that a newly authorized production run would use this exact implementation and convention, and no fresh confirmation data receipt exists. Keep `operator` and `comparison` abstained until those receipts and review are supplied. Do not change the historical development result or reinterpret the APE weak-field matching as nonperturbative equivalence.

Text flow:

```text
contract ID + development descriptor
  -> normalize to (stout, rho=0.1, steps=5, spatial links)
  -> verify pinned implementation hashes
  -> development-kernel identity resolved
  -> production receipt/review still absent
  -> operator certification abstains; physical score stays null
```
