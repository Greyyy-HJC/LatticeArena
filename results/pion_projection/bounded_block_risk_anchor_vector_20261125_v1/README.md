# Pion optional Markov vector-anchor box

**Status: conditional formula fixture only. `physical_score=null`; no physical/anchor data, sealed inputs, GPU or QUDA were accessed.**

This diagnostic applies the existing corrected Markov Bernstein radius to each fixed feature of a vector from the same hypothetical anchor chain. A union bound with family size `d=3` creates a simultaneous box. The single path-level initial-law TV budget `0.001` is subtracted once from `alpha_anchor=0.01`, leaving stationary alpha `0.009`; per-coordinate tails use family size three. The assumed pseudo-gap lower bound is `0.20`, and the assumed post-burn-in count is 8. These are illustrative inputs only.

The 4,000 rows are deterministic sine/cosine arithmetic, not configurations, a Markov path, or Monte Carlo. Their sample means only position the example intervals; they do not test coverage. The feature support ranges are also assumed.

| feature | assumed support | fixture mean | conditional interval | capped radius |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0, 1 | 0.500000 | 0.267453, 0.732547 | 0.232547 |
| 1 | -1, 1 | 0.099939 | -0.365156, 0.565034 | 0.465095 |
| 2 | 0, 1 | 0.700044 | 0.467496, 0.932591 | 0.232547 |

**Text flow:** `fixed bounded per-state feature vector -> Markov Bernstein interval for each coordinate -> Bonferroni joint mean box -> certified interval propagation through the frozen PP/A4P fit -> m_star interval and bias bridge C`. The implementation ends at the vector mean box. It deliberately returns `target_fit_interval=null` and `common_bias_bound_C=null` because no verified range solver is supplied.

The vector route avoids asserting that the nonlinear ensemble fit itself is a per-configuration scalar. It still needs proven feature supports, anchor-chain gap/TV receipts, an interval-certified image of the selected multichannel fit over the joint box, correspondence to the same `m_star`, and a common-bias/score-independence receipt. Any missing assumption leaves the Pion task `NOT_READY/UNCERTIFIED`.

Receipt denominators: vector box `1/1/0/0`; feature coordinates `3/3/0/0` (planned/valid/failed/abstained). `run_receipt.json` records source hashes, assumptions, denominators, artifact hashes and the text flow. Receipt `complete` accounts for this synthetic fixture only; it is not a measurement or coverage result.
