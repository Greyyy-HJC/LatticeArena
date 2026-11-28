# Optimize D06 influence-HAC coverage over the AR(1) grid

**Evidence class:** synthetic stationary Gaussian AR(1) only. This reuses the open `250`-panel-per-cell baseline grid and regenerates the same q-panels from saved seeds. It reads no lattice data, fits, GPU/QUDA, or sealed split. The percentile comparator has `1000` outer draws per block length, below the selected 10,000-draw protocol.

## Text flow

```text
archived n=[40, 60, 100] x phi=[0.0, 0.4, 0.8] x 250 synthetic AR(1) panels/cell
  -> regenerate paired arms and verify exact generating risk ratios
  -> finite-L D04 risk + analytic configuration influence, L=2/5/10
  -> paired ratio influence -> HAC bandwidth 2/5/10/20/30 -> lower-bound hull
  -> compare with percentile-hull rows from the same panels
  -> cellwise marginal/F=2 coverage, Monte Carlo intervals, paired coverage changes
```

## F=2 familywise coverage

The familywise target is at least 95%; Wilson intervals quantify the Monte Carlo error in each cell. Full marginal coverage, every panel lower bound, and cell denominators are in `coverage_summary.csv` and `replicates.csv`.

| n | phi | method | coverage | Wilson 95% MC interval | P/V/F/A |
| ---: | ---: | --- | ---: | --- | --- |
| 40 | 0 | percentile | 0.952 | [0.918, 0.972] | 250/250/0/0 |
| 40 | 0 | hac | 0.984 | [0.960, 0.994] | 250/250/0/0 |
| 40 | 0.4 | percentile | 0.896 | [0.852, 0.928] | 250/250/0/0 |
| 40 | 0.4 | hac | 0.976 | [0.949, 0.989] | 250/250/0/0 |
| 40 | 0.8 | percentile | 0.792 | [0.737, 0.838] | 250/250/0/0 |
| 40 | 0.8 | hac | 0.952 | [0.918, 0.972] | 250/250/0/0 |
| 60 | 0 | percentile | 0.956 | [0.923, 0.975] | 250/250/0/0 |
| 60 | 0 | hac | 0.944 | [0.908, 0.966] | 250/250/0/0 |
| 60 | 0.4 | percentile | 0.916 | [0.875, 0.944] | 250/250/0/0 |
| 60 | 0.4 | hac | 0.968 | [0.938, 0.984] | 250/250/0/0 |
| 60 | 0.8 | percentile | 0.756 | [0.699, 0.805] | 250/250/0/0 |
| 60 | 0.8 | hac | 0.944 | [0.908, 0.966] | 250/250/0/0 |
| 100 | 0 | percentile | 0.948 | [0.913, 0.969] | 250/250/0/0 |
| 100 | 0 | hac | 0.900 | [0.857, 0.931] | 250/250/0/0 |
| 100 | 0.4 | percentile | 0.932 | [0.894, 0.957] | 250/250/0/0 |
| 100 | 0.4 | hac | 0.908 | [0.866, 0.938] | 250/250/0/0 |
| 100 | 0.8 | percentile | 0.788 | [0.733, 0.834] | 250/250/0/0 |
| 100 | 0.8 | hac | 0.828 | [0.776, 0.870] | 250/250/0/0 |

## Paired familywise coverage change

Positive values favor HAC in this synthetic grid. Intervals bootstrap the paired MC panels within each cell and do not represent physical uncertainty.

| n | phi | HAC - percentile | paired panel bootstrap 95% |
| ---: | ---: | ---: | --- |
| 40 | 0 | +0.032 | [+0.008, +0.060] |
| 40 | 0.4 | +0.080 | [+0.048, +0.116] |
| 40 | 0.8 | +0.160 | [+0.116, +0.208] |
| 60 | 0 | -0.012 | [-0.040, +0.016] |
| 60 | 0.4 | +0.052 | [+0.024, +0.084] |
| 60 | 0.8 | +0.188 | [+0.140, +0.240] |
| 100 | 0 | -0.048 | [-0.076, -0.024] |
| 100 | 0.4 | -0.024 | [-0.048, -0.004] |
| 100 | 0.8 | +0.040 | [+0.008, +0.072] |

## Interpretation boundary

The high-correlation `n=60, phi=0.8` targeted 10,000-draw comparison is reported separately in [`d06_hac_coverage_cpu_20261125_v4`](d06_hac_coverage_cpu_20261125_v4/README.md). This grid checks whether its apparent improvement persists over smaller `n` and lower correlations, but does not validate HAC as a D06 production interval. Newey-West relies on an asymptotic linear approximation and weak-dependence/moment conditions; neither the synthetic grid nor the finite-draw percentile comparator establishes those conditions for a gauge chain. Heavy tails, nonstationarity, fit/reference uncertainty and optimizer failures remain untested. Retain `UNCALIBRATED/NOT_READY`, `physical_score=null`, and the sealed-data boundary.
