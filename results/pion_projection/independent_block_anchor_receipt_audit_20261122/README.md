# Independent-block and anchor receipt audit (2026-11-22)

## Scope and decision

This is a metadata/report audit only. It reads the already observed Pion D development receipt for IDs 0–99, the matching frozen measurement plan, and the previously observed source–sink development receipt for IDs 0–59. It does not read E correlators, D gap/validation correlators, or shared holdout data; it does not fit, simulate, or measure anything.

**Denominator/status:** this audit adds no physical dataset or bootstrap draw (`new planned/valid/failed/abstained = 0/0/0/0`; this is no execution, not a zero failure rate). The existing D receipt has 100/100 expected configuration records; it defines 20 length-5 candidate blocks, but the independently certified block count is `NOT_ESTABLISHED`. The amended E plan names 60 anchor IDs, but the E correlator receipt is `NOT_RUN/NOT_OPENED`, not 60 failed configurations. The old measured multi-source v1 has 60 provenance records; it is not a receipt for the matched-v2 candidate.

**Decision:** existing CPU evidence does not identify an independently calibrated evaluation-block source or a bound on anchor common bias. Nothing in this audit changes `ABSTAIN/NOT_READY`, `physical_score=null`, or the sealed-data status.

## Evidence table

| Candidate unit/source | Directly recorded | Independence or bias claim supported | Missing or limiting evidence | Can supply anchor `A/B_A`? |
|---|---|---|---|---|
| Pion point-source M2, D development | 100 ordered configurations, IDs 0–99; one HMC trajectory ID per configuration, 401–896 in fixed increments of 5; per-row gauge and propagator SHA-256; generation `ab8a41d27e0488ca3646ac83475e680f677b17a913d05913f7731ec87a4448ce`; recorded protocol fingerprint `18ed89c914be7d11522256e1b159e2037cd26e89dbd6f46521a442110088f442`. | Configuration identity and a fixed saved-trajectory interval are evidenced. Five consecutive saved configurations can be grouped into each of 20 nonoverlapping blocks under the recorded `block_length=5` convention. | Fixed spacing does not prove independence. No score-relevant integrated autocorrelation bound or a frozen criterion linking it to block length is established here. The single-cosh reference gate already failed on this D sample; blocks cannot repair that model-identification failure. | No. These are development observations, not an independent anchor. |
| Source and RHS within each D configuration | Frozen plan records a point source at the origin and 12 spin-color right-hand sides; it says to average sources within a configuration before blocking and explicitly says sources are not configurations. | The 12 inversions are algebraic/source components for one configuration-level observation, not 12 independent statistical samples. | There is one source position in this protocol. No independent source-position replication is recorded. | No. |
| Length-5 block bootstrap on D0–99 | The analysis uses 20 complete blocks and 1000 bootstrap draws with seed 20261120; the bootstrap resamples block indices. | It propagates uncertainty conditional on treating those 20 blocks as exchangeable resampling units. | Bootstrap draws are not new independent blocks. The resampling procedure cannot establish the exchangeability/independence assumption it uses. Existing block-length sensitivity is a stability diagnostic, not a proof of decorrelation. | No. |
| Independent E anchor in the amended plan | The plan names E IDs 0–59 as a separate anchor split; the D closure README states that the E anchor was not opened. | Only the planned identity of a future anchor is evidenced. | No E correlator receipt was examined or produced in this audit; therefore no anchor endpoint, `A`, or common-bias bound `B_A` is available. An HMC-chain receipt alone would not supply these correlator-level quantities. | Not with current inspected evidence. |
| Previously measured multi-source generation (`s16t16_ms`) | 60 development records, IDs 0–59; no trajectory IDs, generation ID, or recorded protocol hash in the inspected provenance. Its embedded protocol fingerprint recomputed for this audit is `be71b3e561f77e53b6f7543de74fbf84ee24c15d0d7657eb20bcc1ca23878cad` (not present as a receipt field). Its recorded H4/H16 `rho` values are 1.4291792/2.8583584. | It is a distinct recorded data/protocol identity from point-source M2. The D0–59 and multi-source 0–59 gauge raw-file digests have 0/60 exact matches; this excludes exact raw-byte reuse for those IDs only. | The missing trajectory map prevents a chain/block relation check. Different raw-file hashes alone do not prove independent Markov histories or physics-level independence. This observed generation is protocol v1, not the proposed matched v2; the old source–sink matrix anti-Hermitian identity gate failed at roughly 0.29–0.32 against the 0.05 criterion (see the linked report). | No. It is not a passing matched-source anchor, and it does not bound common bias. |
| Proposed matched source–sink protocol v2 | Candidate protocol file SHA-256/fingerprint: `f3150c45b8e15218de9a7c54f1844ea2fc3c7b7c6dbe99f35de8d2d7ca1b00d9`; it specifies shared low-level smearing alpha 0.2, with H4/H16 `rho` values 1.2060454/2.4120908. The file itself says CPU-reviewed candidate only and no GPU measurement is authorized. | It identifies a candidate future measurement contract. | No generation or measurement receipt binds any arrays to this v2 protocol. It cannot be substituted for the failed v1 arrays or counted as an anchor. | No. |

The point-source and multi-source records also have different ensemble labels (`S16T16D` versus the earlier `S16T16` path), seeds, sink-power lists, protocol versions, and source-smearing records. Keep their identities separate. A protocol hash identifies a frozen configuration; by itself it does not establish independent HMC blocks or calibrate common-mode bias.

## Data flow and statistical units

```text
one D configuration
  ├─ one source position (origin)
  ├─ 12 spin-color RHS -> one configuration-level correlator row
  └─ one saved HMC trajectory ID; neighboring saved rows differ by 5 trajectories

100 D rows -> 20 contiguous, nonoverlapping blocks of 5 rows -> 1000 resampled block-index draws
                ^                                                   ^
                └─ candidate analysis unit, not proven independent └─ uncertainty draws, not new data

separate E IDs 0–59 (planned anchor) -> no inspected correlator receipt -> A and B_A remain unknown
```

## Direct findings, inference, and unresolved questions

- **Directly evidenced:** the D receipt is complete for IDs 0–99 and carries per-configuration hashes and trajectory IDs; the saved trajectory spacing is exactly five; the selected block length is five configurations; the source/RHS and bootstrap roles are defined by the plan and analysis code.
- **Inference only:** length-5 contiguous blocks are a reproducible blocking convention and may reduce short-range correlation. Their independence is not demonstrated by the spacing or by bootstrap resampling.
- **Not evidenced:** a score-specific decorrelation bound; an independent evaluated anchor; anchor common-bias bound `B_A`; or a matched-v2 source–sink measurement receipt.

The actionable next decision is methodological, not another CPU rerun: before any future measurement, freeze which observable's autocorrelation controls the block length, the rule that certifies an effective independent-block count, and the physical/auxiliary evidence used to bound `B_A`. Until those values and the reference/model-identification plan are reviewed, retain the current abstention and data seal.

## Source records

The original audit inspected the already-observed D0–99 development receipt and frozen measurement plan, the previously observed multi-source receipt, and their source–sink identity report. Those input records and correlators are not copied into this PR; the table above preserves the metadata findings and explicitly records the missing independence and anchor evidence. No sealed E/gap/validation/holdout input is included here.
