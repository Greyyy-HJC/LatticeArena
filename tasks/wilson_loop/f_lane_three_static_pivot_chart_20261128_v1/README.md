# Largest-component overlap phase chart: CPU method check

**Status:** deterministic parameterization check only. This prototype does not change the selected finite-T Creutz or ground-state force estimand, and it does not provide a fitted force interval or physical evidence.

## Why add this chart

The current three-static-state parameterization fixes operator 0 as the positive-real phase anchor for every transition vector. Its initializer therefore rejects a valid nonzero vector such as

```text
Z = (0, 0.70 + 0.10 i, -0.20 + 0.10 i)
```

even though the full 3 × 3 positive matrix outer product contains the state. The new chart fixes the phase using the largest-magnitude component of each transition vector. It stores that component as a positive log-amplitude and stores the other two complex components directly. An all-zero vector has no identifiable phase chart and is rejected.

## Data flow

```text
finite-Nt transition vectors
  -> choose argmax_i |Z_i| separately for each (R,m,n)
  -> rotate the common phase so the chosen component is positive real
  -> encode log(|Z_pivot|) plus the remaining two complex components
  -> decode the same positive spectral matrices and analytic Jacobian
  -> compare against the fixed-operator-0 model and central differences
  -> method diagnostic only; no physical gate input
```

## Observed checks

The chart preserves the existing finite-Nt model to maximum absolute matrix-vector difference **1.11e-16** on a deterministic 67-coordinate fixture that exercises all three pivot choices (counts `10/1/1`). Its analytic Jacobian differs from a central finite-difference check by at most **5.03e-11** in absolute and unit-scaled maximum norm.

A separate exact component fixture sets operator 0 of one transition to zero while operator 1 is nonzero. The pivot chart reconstructs its finite-Nt matrix exactly; the current fixed-operator-0 packer rejects that vector. The all-zero vector is explicitly rejected because its state amplitude has no phase anchor and cannot identify a nonzero contribution.

Focused tests: `tests/test_wilson_f_lane_three_static_pivot_chart.py` **5 passed**. The adjacent Optimize numeric-overflow tests also pass, for **11 passed** across the two focused test files. Source and report identity are listed in `source_hashes.json` and `artifact_hashes.json`.

## What this does and does not establish

This is an exact coordinate-coverage improvement for a positive finite-Nt spectral model. It does not demonstrate that nonlinear profile fits converge more often, improve interval containment, or have calibrated coverage. It does not establish production source/sink conventions or an independent overlap-floor receipt. Before adopting the chart in a fit, compare it against the fixed-operator-0 chart on the same frozen open synthetic matrices, using identical covariance, force grid, starts, solver budget, and acceptance rules; report complete / failed / abstained / clipped denominators. Keep the profile and F lane uncertified until the independent physical receipts and review gates pass.
