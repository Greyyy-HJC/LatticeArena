# Optimize D06 paired percentile-hull coverage pilot

**Evidence class:** synthetic stationary Gaussian AR(1) only. No lattice data, correlators, fits, GPU, QUDA, or sealed splits were accessed. This pilot does not certify physical coverage. It used 1000 outer draws and 250 Monte Carlo panels per `(n, phi)` cell; the selected production protocol requires 10,000 paired outer draws per block length.

## Text flow

```text
known stationary paired Gaussian AR(1) process for q(theta,t), t=2..5
  -> n in [40, 60, 100] and phi in [0.0, 0.4, 0.8], shared noise across all three arms
  -> for each L=2,5,10, paired circular-block outer resamples
  -> recompute empirical D04 risk using exact moments of the inner circular-block bootstrap law
  -> paired comparator/trainable-W risk ratios; linear 0.025 quantile per L
  -> take the minimum L lower quantile; check it against known generating-process ratio
  -> evaluate marginal and F=2 simultaneous coverage; retain all planned panels
```

The inner block-bootstrap variance is computed analytically: if `k=n/L` circular blocks are drawn with replacement, the conditional variance of their mean is the population variance of the `n` circular length-`L` block means divided by `k`. This is the infinite-inner-draw limit of the selected D04 bootstrap statistic; it isolates outer percentile-hull behavior from finite inner-bootstrap noise. The outer procedure uses 1000 finite draws per block length. This pilot uses 1000 outer draws per length; the selected contract uses 10,000.

For each arm and time, the truth is `bias^2 + Var(sample mean)`; the AR(1) variance is evaluated exactly as `sigma^2/n^2 * [n + 2*sum_(h=1)^(n-1)((n-h)*phi^h)]`. The same stationary `phi` is used for all arms; a common Gaussian component keeps the arms paired. Both user-selected contrasts are evaluated jointly, with `alpha=0.025` per contrast and `r_star=10/9`.

## Empirical coverage

The table reports the fraction of Monte Carlo panels whose lower hull did not exceed the known risk ratio, Wilson 95% intervals for the Monte Carlo coverage rate, and `planned/valid/failed/abstained` panel counts.

| n | phi | interval family | coverage | Monte Carlo Wilson 95% | panels P/V/F/A |
| ---: | ---: | --- | ---: | ---: | ---: |
| 40 | 0.0 | SLSQP | 0.960 | [0.928, 0.978] | 250/250/0/0 |
| 40 | 0.0 | Chebyshev | 0.984 | [0.960, 0.994] | 250/250/0/0 |
| 40 | 0.0 | familywise | 0.952 | [0.918, 0.972] | 250/250/0/0 |
| 40 | 0.4 | SLSQP | 0.956 | [0.923, 0.975] | 250/250/0/0 |
| 40 | 0.4 | Chebyshev | 0.932 | [0.894, 0.957] | 250/250/0/0 |
| 40 | 0.4 | familywise | 0.896 | [0.852, 0.928] | 250/250/0/0 |
| 40 | 0.8 | SLSQP | 0.880 | [0.834, 0.915] | 250/250/0/0 |
| 40 | 0.8 | Chebyshev | 0.884 | [0.838, 0.918] | 250/250/0/0 |
| 40 | 0.8 | familywise | 0.792 | [0.737, 0.838] | 250/250/0/0 |
| 60 | 0.0 | SLSQP | 0.972 | [0.943, 0.986] | 250/250/0/0 |
| 60 | 0.0 | Chebyshev | 0.980 | [0.954, 0.991] | 250/250/0/0 |
| 60 | 0.0 | familywise | 0.956 | [0.923, 0.975] | 250/250/0/0 |
| 60 | 0.4 | SLSQP | 0.944 | [0.908, 0.966] | 250/250/0/0 |
| 60 | 0.4 | Chebyshev | 0.956 | [0.923, 0.975] | 250/250/0/0 |
| 60 | 0.4 | familywise | 0.916 | [0.875, 0.944] | 250/250/0/0 |
| 60 | 0.8 | SLSQP | 0.856 | [0.807, 0.894] | 250/250/0/0 |
| 60 | 0.8 | Chebyshev | 0.868 | [0.820, 0.904] | 250/250/0/0 |
| 60 | 0.8 | familywise | 0.756 | [0.699, 0.805] | 250/250/0/0 |
| 100 | 0.0 | SLSQP | 0.976 | [0.949, 0.989] | 250/250/0/0 |
| 100 | 0.0 | Chebyshev | 0.972 | [0.943, 0.986] | 250/250/0/0 |
| 100 | 0.0 | familywise | 0.948 | [0.913, 0.969] | 250/250/0/0 |
| 100 | 0.4 | SLSQP | 0.956 | [0.923, 0.975] | 250/250/0/0 |
| 100 | 0.4 | Chebyshev | 0.964 | [0.933, 0.981] | 250/250/0/0 |
| 100 | 0.4 | familywise | 0.932 | [0.894, 0.957] | 250/250/0/0 |
| 100 | 0.8 | SLSQP | 0.880 | [0.834, 0.915] | 250/250/0/0 |
| 100 | 0.8 | Chebyshev | 0.872 | [0.825, 0.908] | 250/250/0/0 |
| 100 | 0.8 | familywise | 0.788 | [0.733, 0.834] | 250/250/0/0 |

## Individual block-length coverage at the targeted high-dependence point

The L=2/5/10 rows below are the un-hulled marginal intervals at `n=60, phi=0.8`. The complete per-cell/per-length results are in `block_length_coverage.csv`.

| interval family | L | coverage | Monte Carlo Wilson 95% | panels P/V/F/A |
| --- | ---: | ---: | ---: | ---: |
| SLSQP | 2 | 0.740 | [0.682, 0.790] | 250/250/0/0 |
| Chebyshev | 2 | 0.764 | [0.708, 0.812] | 250/250/0/0 |
| familywise | 2 | 0.608 | [0.546, 0.666] | 250/250/0/0 |
| SLSQP | 5 | 0.836 | [0.785, 0.877] | 250/250/0/0 |
| Chebyshev | 5 | 0.832 | [0.781, 0.873] | 250/250/0/0 |
| familywise | 5 | 0.724 | [0.666, 0.776] | 250/250/0/0 |
| SLSQP | 10 | 0.852 | [0.803, 0.891] | 250/250/0/0 |
| Chebyshev | 10 | 0.864 | [0.816, 0.901] | 250/250/0/0 |
| familywise | 10 | 0.748 | [0.691, 0.798] | 250/250/0/0 |

The raw per-panel ratios and each block-length lower quantile are in `replicates.csv`; exact generator inputs and source hashes are in `protocol.json`. The Monte Carlo interval quantifies finite simulation noise only. These Gaussian AR(1) cases do not represent the gauge-chain dependence, fitted-reference uncertainty, optimizer failures, or the production evaluator. The observed undercoverage is a warning for this nested percentile implementation under the specified AR(1) model, not a measured property of the lattice chain and not a PASS for either comparison.
