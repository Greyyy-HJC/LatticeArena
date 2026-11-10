# S16T16 pion projection: first agent attempt

**2026-11-09. Diagnostic pilot; no official score and no SOTA claim.**

The task now runs on real unfixed S16T16 configurations. A development-selected
signed multiscale sink has a lower provisional loss on the held-out configurations
than the tuned Gaussian baseline. However, the common ground-state reference
fails the predeclared checks, and the improvement interval relative to that
baseline includes no improvement. The result motivates another controlled
experiment; it does not establish suppressed true excited-state overlap.

## What was run

- Fixed origin point source, Wilson valence kappa 0.14, 12 spin-color inversions,
  antiperiodic time boundaries, residual target 1e-10.
- Development configurations 0–29; configurations 30–34 unused as a separation
  gap; validation configurations 35–49 first evaluated after freezing the proposal.
- Seven gauge-covariant kernels `H^n`, with n = 0, 2, 4, 8, 16, 32, 64 and
  `H = (I + 0.2 * spatial_hopping)/2.2`.
- Contiguous five-configuration blocks; 1000 bootstrap draws and a 200 x 128
  nested bootstrap for conditional loss-ratio intervals.

All 45 gauge files passed the default NERSC checks. All 540 propagator columns
passed an explicit residual check; the maximum was `9.964e-11`. The recorded
configuration-loop time totaled about 46.3 seconds on the RTX 3060, including
about 14.7 seconds for inversion/cache loading plus residual checks and 15.7
seconds for sink work. These are local warm-cache/QUDA-tuning-dependent timings,
not a hardware-independent performance result. The CuPy pool reservation reached
about 492 MiB; this excludes QUDA's separate memory allocations.

Seven automated tests passed. As an end-to-end regression, the identity operator
was separately submitted through the generic API on all 45 measured configurations:
its correlators matched the fixed local baseline within `1e-12` relative tolerance,
and its diagnostic loss ratio was one to floating-point precision. The failed
reference correctly kept its official score null as well.

## The proposed operator

Development selected `H^32` as the best single diffusion/Gaussian kernel. The agent
then optimized a signed sum of all seven scales, including a negative local
counterterm. Its kernel is `K = sum_n u_n H^n / A_n`, where `A_n` are fixed
development amplitudes. The normalized coefficients were:

| n | u_n |
| --- | ---: |
| 0 | -0.373289 |
| 2 | 0.0190495 |
| 4 | 0.192788 |
| 8 | 0.335503 |
| 16 | 0.385543 |
| 32 | 0.311827 |
| 64 | 0.128578 |

The coefficients sum to one and have L1 norm 1.747, below the fixed limit of 8.
All eight optimization starts converged. The raw, directly applicable kernel
coefficients are preserved in the [candidate definition](../results/pion_projection/s16t16_v1/agent_candidate.json),
and all calibration and search choices are in the
[frozen submission](../results/pion_projection/s16t16_v1/submission.json).
The negative local term can cancel part of the short-distance contribution;
this interpretation is a hypothesis, not a measured spectral decomposition.

The submitted kernel was also applied directly to the cached propagators on
configurations 0 and 35 through the generic time-slice API. Its full 16-time
correlators agreed with the QUDA-basis linear combination to relative differences
below `3.83e-15`. This checks that the reported candidate is an actual operator,
not only a postprocessed fit curve. The unoptimized generic path took about
33 seconds per configuration; the QUDA fast path is much faster for this family.

## Held-out diagnostics

The following use the provisional common reference `am0 = 1.07671(534)` and the
fixed scoring window t = 2–5. “Departure” means the RMS of `q(t)-1`, with
`q = C/(A f0)`, and must not be read as a physical excited-state fraction.

| Sink | Variance term | RMS departure | Total diagnostic loss | SNR at t=2 | SNR at t=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local | 0.001784 | 70.34% | 0.49662 | 111.98 | 36.86 |
| Tuned Gaussian, H^32 | 0.0005189 | 5.92% | 0.0040183 | 50.94 | 29.62 |
| Agent signed mixture | 0.0004336 | 1.18% | 0.00057190 | 46.46 | 29.91 |

The Gaussian-to-agent loss ratio is **7.03**, with a conditional paired 95%
bootstrap percentile interval **[0.978, 14.09]**. The interval includes one.
Only three validation blocks are available, so the interval is coarse. It also
does not include uncertainty from an inadequate reference model or the unknown
long-distance Markov-chain autocorrelation.

Most of the improvement is in the departure term. It is not a simultaneous SNR
win: relative to the tuned Gaussian, the agent's SNR at t=2 drops by about 9%,
and relative to the local sink early-time SNR is substantially lower. The total
loss trades raw SNR against the ground-state-normalized departure and uncertainty.

![Pilot diagnostics](../results/pion_projection/s16t16_v1/pilot_results.png)

The effective-mass panel plots the effective mass of the mean correlator, without
error bars; missing points are undefined cosh effective masses, not clipped values.
The signed candidate has a negative t=0 contact correlator and an unusually low
t=2 effective mass. These are allowed for the mixed point/signed-sink correlator
but illustrate why matching a finite reference window does not prove a clean
spectral projection. The q-panel error bars are one bootstrap standard deviation.

## Why scores remain null

The predeclared common reference fit uses the local, H^8, and H^16 channels:

| Reference window | Fitted am0 | Bootstrap SE | Approximate fit p |
| --- | ---: | ---: | ---: |
| 4–8 | 1.10337 | 0.00656 | 1.33e-32 |
| 5–8, nominal | 1.07671 | 0.00534 | 3.38e-6 |
| 6–8 | 1.06009 | 0.00424 | 0.310 |

Relative to the nominal window, the paired shifts are 10.66 and 9.45 standard
errors, exceeding the fixed 2.5 threshold. The nominal reference also fails its
approximate goodness-of-fit threshold. Selecting only the last window because
it has a favorable p-value would violate the already frozen protocol.

The maximum development forward/backward z statistic is 6.09, at t=2 in H^4,
above the threshold of 4. This is a diagnostic flag rather than proof that the
ensemble violates reflection symmetry: the point-source sampling, only six
blocks, and unknown autocorrelations require further investigation. Folded
curves remain descriptive, and this failure is another reason to withhold scores.

The evaluator records `unscored_reference_failed` and `score: null` for every
candidate. The large loss ratios against the local sink are **not** leaderboard
scores. The data cannot establish a well-controlled true ground-state target
for the proposed scoring rule under the fixed v1 protocol.

## Artifacts and next research step

See the [executable protocol and commands](../tasks/pion_projection/README.md),
[validation results](../results/pion_projection/s16t16_v1/validation_results.json),
[development results](../results/pion_projection/s16t16_v1/development_results.json),
and [exportable PDF figure](../results/pion_projection/s16t16_v1/pilot_results.pdf).
Per-configuration correlators, checksums, solver diagnostics, reference bootstrap
draws, and nested loss draws accompany them in the same result directory.

The next priority is to establish a controlled reference, not to continue tuning
on these 15 validation configurations. A follow-up protocol should investigate
autocorrelation and forward/backward sampling, test a shared multi-state analysis
on development data, and validate against more configurations and/or a longer
temporal lattice at matched physics. It should compare the frozen signed
candidate and Gaussian baseline on independent data, with new scoring calibration
fixed before evaluation. The current attempt has not covered the full arbitrary
kernel search space or a comprehensive set of literature SOTA methods.
