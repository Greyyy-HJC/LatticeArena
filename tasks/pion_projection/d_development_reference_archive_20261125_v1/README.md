# Pion S16T16D open-development reference diagnostics archive

**Status:** D-development diagnostic only; the frozen single-state reference gate failed; this package is not an E-anchor receipt, coverage receipt, or physical score. It archives existing local results and figures; it is not a rerun.

## Observed development result

The open split contains 100 configurations, IDs 0–99, generation `ab8a41d27e0488ca3646ac83475e680f677b17a913d05913f7731ec87a4448ce`. The candidate-blind nominal single-state fit returned `a_t m=1.0961829 ± 0.00305`, but approximate goodness-of-fit was `1.081e-15`; reference windows `[4,5,6,7,8]` and `[6,7,8]` shifted by 20.41 and 15.78 paired standard errors. The frozen gate therefore remains failed. The measurement amendment records that the earlier length-3 block sensitivity could not form equal complete blocks from 100 rows; it uses complete partitions only and drops no row.

An exploratory two-state fit on the same D IDs improves approximate fit p-values to 0.90–0.997 and reduces the two reported ground-state window shifts to 0.86σ and 1.92σ. However, the `[6,7,8]` fit reaches the excited-state upper mass bound and has condition number about `2.99e5`; improved in-sample fit does not identify the second state. The effective-mass scan also shows strong time and sink-power dependence: at `t=7`, bootstrap medians span `1.034–1.093` across powers, compared with the failed common single-state fit. These are correlated channels and times on one development sample, not independent plateau evidence.

## Figures and decision flow

`development_diagnostics.png` compares candidate reference windows and block-length sensitivity; `two_state_diagnostic.png` contrasts the single- and two-state window fits; `effective_mass_diagnostic.png` shows channel/time curvature; `effective_mass_late_time.png` shows the late-time intervals.

```text
open D IDs 0–99 -> candidate-blind single-state gate fails
  -> exploratory two-state fit improves in-sample fit
  -> late-window excited mass hits bound and fit is ill-conditioned
  -> effective-mass curves retain time/channel curvature
  -> do not promote this D fit to independent E anchor or m* certificate
```

## Scope and limitation

The original reports say E-chain anchor and D validation/shared holdout were not opened. This archive does not read or contain anchor, validation, gap, shared-holdout, or confirmation data. The nonlinear fit is estimated on the same D development chain; its success at a selected model cannot establish independent anchor coverage, Markov gap/TV assumptions, or a common-bias bound `C`. It also does not justify a vector-anchor interval image `g(box)` until the exact fit map, covariance, parameter domains and acceptance gates are frozen and globally identified.

The archived package includes report text and figures plus hashes of the source inputs/outputs; it excludes raw correlator arrays and per-bootstrap fit arrays. The original runners were recovered in the local root checkout, numerically replayed in an isolated temporary directory, and the 15-file Python import closure is preserved in [`source_snapshot/`](source_snapshot/README.md). [`replay_protocol.json`](replay_protocol.json) reconstructs the protocol object from the open D provenance and matches its canonical fingerprint; see [runner replay audit](runner_replay_audit.md). The PR archive still cannot run by itself because the D input arrays and a pinned runtime environment are not included. Keep Pion `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs closed.
