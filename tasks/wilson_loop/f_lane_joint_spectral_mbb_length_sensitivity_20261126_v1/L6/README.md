# Wilson F-lane MBB length-sensitivity cell

**Synthetic CPU diagnostic only.** This report reuses the hash-verified six-scenario stress bundle. It reads no Wilson measurements, sealed split, GPU, or QUDA data. `physical_score=null`; F remains `GROUND_STATE_FORCE_UNCERTIFIED`.

This is the pre-registered `L=6` cell in the fixed `L={6,8,10}` scan. The `L=8` baseline was inspected before this scan; this child is not an independent confirmation and cannot select a preferred length. All cells use the same deterministic holdout panel seeds, 40 calibration panels and 40 holdout panels per scenario, `B=40`, and upper order-statistic rank 40. Bootstrap seed keys are shared across lengths; the sampled circular chunk layout changes with L.

## Results

![Pointwise interval coverage by method](method_coverage.png)

Planned/valid/failed/abstained: holdout panels `240/240/0/0`; bootstrap refits `9600/9522/78/0`; moving-block intervals `960/860/0/100`. Interval coverage is conditional on valid intervals; the table retains the full denominator.

| Scenario | Operator | T | Planned/valid/failed/abstained | Coverage (95% Wilson MC interval) | Median upper / truth |
|---|---:|---:|---:|---:|---:|
| `matched_3state_independent` | 0 | 5 | 40/40/0/0 | 1.000 [0.912, 1.000] | 1.194 |
| `matched_3state_independent` | 0 | 6 | 40/40/0/0 | 1.000 [0.912, 1.000] | 1.204 |
| `matched_3state_independent` | 2 | 5 | 40/40/0/0 | 0.975 [0.871, 0.996] | 1.083 |
| `matched_3state_independent` | 2 | 6 | 40/40/0/0 | 0.975 [0.871, 0.996] | 1.088 |
| `omitted_4th_state` | 0 | 5 | 40/40/0/0 | 0.950 [0.835, 0.986] | 1.171 |
| `omitted_4th_state` | 0 | 6 | 40/40/0/0 | 0.950 [0.835, 0.986] | 1.178 |
| `omitted_4th_state` | 2 | 5 | 40/40/0/0 | 0.950 [0.835, 0.986] | 1.077 |
| `omitted_4th_state` | 2 | 6 | 40/40/0/0 | 0.950 [0.835, 0.986] | 1.077 |
| `near_degenerate_first_gap` | 0 | 5 | 40/38/0/2 | 0.921 [0.792, 0.973] | 1.985 |
| `near_degenerate_first_gap` | 0 | 6 | 40/38/0/2 | 0.921 [0.792, 0.973] | 2.001 |
| `near_degenerate_first_gap` | 2 | 5 | 40/38/0/2 | 0.947 [0.827, 0.985] | 1.857 |
| `near_degenerate_first_gap` | 2 | 6 | 40/38/0/2 | 0.947 [0.827, 0.985] | 1.870 |
| `small_ground_overlap_operator_2` | 0 | 5 | 40/24/0/16 | 0.917 [0.742, 0.977] | 1.298 |
| `small_ground_overlap_operator_2` | 0 | 6 | 40/24/0/16 | 0.917 [0.742, 0.977] | 1.316 |
| `small_ground_overlap_operator_2` | 2 | 5 | 40/24/0/16 | 0.917 [0.742, 0.977] | 1.676 |
| `small_ground_overlap_operator_2` | 2 | 6 | 40/24/0/16 | 0.917 [0.742, 0.977] | 1.670 |
| `ar1_block_rho_0p75` | 0 | 5 | 40/33/0/7 | 0.909 [0.764, 0.969] | 1.324 |
| `ar1_block_rho_0p75` | 0 | 6 | 40/33/0/7 | 0.909 [0.764, 0.969] | 1.331 |
| `ar1_block_rho_0p75` | 2 | 5 | 40/33/0/7 | 0.909 [0.764, 0.969] | 1.174 |
| `ar1_block_rho_0p75` | 2 | 6 | 40/33/0/7 | 0.909 [0.764, 0.969] | 1.172 |
| `student_t5_independent` | 0 | 5 | 40/40/0/0 | 0.950 [0.835, 0.986] | 1.142 |
| `student_t5_independent` | 0 | 6 | 40/40/0/0 | 0.950 [0.835, 0.986] | 1.145 |
| `student_t5_independent` | 2 | 5 | 40/40/0/0 | 0.975 [0.871, 0.996] | 1.063 |
| `student_t5_independent` | 2 | 6 | 40/40/0/0 | 1.000 [0.912, 1.000] | 1.067 |

## Interpretation and limits

This cell varies only the circular block length around the previously used `L=8`. It uses the same already-inspected synthetic holdout panels. Any coverage change is descriptive sensitivity to the resampling scheme under this generator; it is not a new coverage guarantee, not a tuning target, and not evidence about HMC-chain correlation. Failed/missing bootstrap q estimates remain `+infinity` when forming the upper order statistic, and all planned intervals remain in the denominator.

No result from this report may enter the q reader or force gate. `L=6`, `L=8`, and `L=10` remain diagnostic-only regardless of which cell looks best. The pre-registered scan stops after these three values; further block-length or B tuning on this generator/holdout is out of scope. Real F-lane evidence still requires authorized raw cross-correlator snapshots, independently reviewed spectrum/overlap, finite-`N_t` thermal `h`, mixed-sector `x`, paired force/reference uncertainty, contamination bounds, and independent review.

Text flow: `verify stress-bundle hashes -> regenerate the fixed holdout panels from shared seeds -> circular MBB with this pre-fixed L and B=40 -> refit joint 3-state GLS -> retain failed q draws as +infinity -> compare pointwise coverage, Wilson MC intervals, width, and abstention -> certification abstains`.

Reproduce this child through `PYTHONPATH=. python -B -m tasks.wilson_loop.f_lane_joint_spectral_mbb_length_sensitivity --output tasks/wilson_loop/f_lane_joint_spectral_mbb_length_sensitivity_20261126_v1`. Source and artifact hashes are recorded in `protocol.json`, `summary.json`, and `artifact_hashes.json`.
