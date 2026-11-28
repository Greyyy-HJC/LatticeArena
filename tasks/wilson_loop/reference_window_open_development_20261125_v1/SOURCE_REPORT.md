# Wilson reference-window diagnostics

**Status:** `PROPOSAL_NOT_FROZEN`; `scope=development_diagnostic_only`; `production_score=null`.

## Input and method

- Split: development IDs 0–99; generation `dd38c39f21982307c5abc2d5093b93c2aa12ef8848d6f9136519b1f8054b619e`.
- Input files: NPZ `da8cc5d72481475a8cd6d821692db2e21555c0bd5b1849747dc75a946dbd7209`, provenance `082ad7060e776f6e724ee02ac1de9c94e706332eb875019f3b9d1c45d4e9e82b`, commit marker `0ff2766e4550d0a687d051f44376af5528cb047b05c556912b50ad5c9833951a`.
- Identity loops: 100 configurations, 20 consecutive blocks of length 5; 5000 shared bootstrap draws. Index matrix SHA-256: `211bc173b11aba8bc4a63622d5d17ceed1f2ee27086b8f86ad97c7c5e2ff5ef5`.
- Candidate threshold: 2.5; equality passes by the unfrozen proposal rule.

- Potentials and paired differences are reported in lattice units (`aV` and `aΔV`); R and T are lattice separations.

## Candidate comparison

| Proposal | Candidate result | Statistic | Failed contrasts | Invalid draws |
| --- | --- | ---: | ---: | ---: |
| A: nominal point-max | PROPOSAL_DIAGNOSTIC_PASS | 2.0965152 | 0 | 0 |
| B: all-pair point-max | PROPOSAL_DIAGNOSTIC_PASS | 2.0965152 | 0 | 0 |
| C: centered joint q95 | REJECT_WINDOW_INSTABILITY | 2.7408156 | 0 | 0 |
| C: uncentered joint q95 | REJECT_WINDOW_INSTABILITY | 4.0843804 | 0 | 0 |

These are candidate diagnostic outcomes on the frozen development sample. They do not select a production rule, establish a physical reference, or authorize access to another split.

## Figures and interpretation

- `reference_windows.png` shows window potentials with marginal bootstrap SD and paired window differences with their shared-draw SD.
- `reference_window_z.png` shows the signed A-proposal contrast z values; grey cells are abstentions.
- `reference_joint_max.png` compares centered and uncentered C joint maxima when defined.
- Common shifts shared by all windows can remain invisible to window-stability diagnostics. The threshold and C quantile have no family-wise coverage calibration here.

## Limits and access record

- The runner verifies the frozen protocol/source hashes and transaction-internal NPZ, provenance, generation, per-row loop hashes, declared manifest hash, unique trajectory IDs, and per-row gauge SHA-256 fields. It does not reopen the manifest or re-read raw gauge bytes.
- No validation, gap, or shared-holdout files were accessed. No production score was computed or written.
- Any `PROPOSAL_DIAGNOSTIC_PASS` is not a physical certification; a failed or undefined contrast remains visible in `reference_failures.csv` and the receipt.
