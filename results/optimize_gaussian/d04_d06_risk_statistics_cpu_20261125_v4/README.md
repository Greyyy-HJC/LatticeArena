# D04/D06 pure-array reducer diagnostic

**Evidence class:** deterministic CPU arithmetic fixture only. No gauge or correlator data, reference fit, bootstrap resampling, solver, GPU, sealed split, or physical score was used. The selected production contract remains `NOT_READY/UNCERTIFIED`.

## Text flow

```text
hand-entered q(theta,t) draws at t=(2,3,4,5)
  -> sample variance (ddof=1) + squared bias from one
  -> D04 scalar risk
hand-entered paired comparator/trainable-W risk arrays + matching draw IDs
  -> per-length comparator/candidate ratios
  -> linear 0.025 quantile for L=2,5,10
  -> minimum across lengths compared with r_star=10/9
invalid or incomplete inputs -> planned denominator retained -> ABSTAIN, no epsilon
```

The D04 arithmetic fixture used planned/valid/failed/abstained draws `3/3/0/0` and returned risk `0.784722222`. The illustrative D06 fixtures used four paired draws per block length (`12` planned total, rather than the contract's `30,000`). The all-above-threshold fixture returned hull lower quantile `1.303750` and array-only status `D06_THRESHOLD_CLEARED_ARRAY_ONLY`. Lowering the L=10 ratios returned `1.053750` and `D06_THRESHOLD_NOT_CLEARED`. A zero risk retained denominator `12/11/1/0`, returned no hull, and abstained.

These examples verify arithmetic and fail-closed behavior. They do **not** show that the chosen percentile quantile has calibrated coverage, generate the required paired outer block-bootstrap/inner D04 draws, establish independent configuration units, compare methods, or certify the D04 reference. Those remain open implementation and evidence gates.

## Denominators

| Fixture | Draw planned / valid / failed / abstained | Result |
| --- | ---: | --- |
| D04 hand calculation | 3 / 3 / 0 / 0 | one reducer result, uncertified |
| D06 synthetic clear | 12 / 12 / 0 / 0 | threshold clears on constructed numbers only |
| D06 synthetic below threshold | 12 / 12 / 0 / 0 | no clearance |
| D06 zero-risk case | 12 / 11 / 1 / 0 | abstain, ratio undefined |
