# Wilson reference-window diagnostic archive (D development, 2026-11-25)

**Status:** retrospective development diagnostic only; reference proposal not frozen; `production_score=null`. This package archives an existing local report and its review figures. It is not a rerun.

## Result

The analyzed input was the already-open Wilson development split D0–99 (100 configurations), grouped into 20 consecutive blocks of length 5 and evaluated with 5,000 shared bootstrap draws (seed `2026112021`). The candidate threshold was 2.5 and equality passed under the unfrozen proposal rule.

| Proposal | Statistic | Diagnostic outcome | Interpretation |
|---|---:|---|---|
| A: nominal point-max | 2.0965152 | proposal diagnostic pass | Below candidate threshold; not family-wise calibrated |
| B: all-pair point-max | 2.0965152 | proposal diagnostic pass | Below candidate threshold; not family-wise calibrated |
| C: centered joint q95 | 2.7408156 | reject window instability | Above candidate threshold |
| C: uncentered joint q95 | 4.0843804 | reject window instability | Above candidate threshold |

All four reported comparisons had zero failed contrasts and zero invalid bootstrap draws. These counts describe this development calculation only.

## Figures

- `reference_windows.png`: for each spatial separation, plots window potential estimates with marginal bootstrap standard deviations and paired window differences with their shared-draw standard deviations. The R=3 windows show a visible ordering reversal; the errors are correlated through shared draws.
- `reference_window_z.png`: heat map of signed A-proposal paired standardized contrasts across R and the two nominal window comparisons; the maximum absolute value is 2.0965.
- `reference_joint_max.png`: centered and uncentered joint-maximum distributions. Their linear 95th percentiles are 2.74 and 4.08, both above 2.5.

## Decision flow

```text
open D0–99 transaction + source hashes
  -> form 20 length-5 blocks and 5,000 shared draws
  -> evaluate candidate A/B point-max and C joint-q95 summaries
  -> retain failed/invalid counts and plot window contrasts
  -> retrospective diagnostic only; do not freeze the production rule
  -> require family-wise calibration + independent calibration/confirmation
```

## Evidence boundary

The archived source report records the development generation and the NPZ, provenance, commit-marker, and bootstrap-index hashes. The same D0–99 generation was already used for development and for the separate C-lane retrospective diagnostic; it cannot serve as independent calibration or confirmation. Window-to-window stability cannot detect a common shift shared by every window. Neither the 2.5 threshold nor the C quantile has family-wise coverage calibration here.

No validation, gap, shared-holdout, confirmation, or sealed input was accessed for this archive. It does not select a physical reference, certify ground-state contamination, or change Wilson F's `GROUND_STATE_FORCE_UNCERTIFIED` status.

**Reproducibility limitation:** the original runner was not present in the local report directory and is not included here. This archive preserves the report, summary decision table and figures for review, but it cannot regenerate them from tracked files. The original full per-draw JSON, bootstrap index binary and row-level outputs are intentionally not copied. A future calibration package must include a reviewed runner and fresh, authorized, split-isolated inputs.
