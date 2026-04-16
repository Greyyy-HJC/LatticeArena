# pion_2pt Dataset Contract

`pion_2pt` uses task-local gauge data plus a small metadata file that tells the
fixed PyQUDA measurement workflow how to interpret the ensemble.

## Local-only quenched test ensemble

The current default regression target is a local quenched `8^3 x 32` ensemble
used for development and smoke tests. It is intentionally not tracked by git.

Recommended layout:

- `quenched_wilson_b6_8x32/`
  - default local quenched `8^3 x 32` smoke-test ensemble
  - `ensemble.json`
  - one or more gauge configurations in NERSC / PyQUDA-readable format
- `quenched_wilson_b6_16x16/`
  - optional legacy local quenched `16^3 x 16` smoke-test ensemble
  - `ensemble.json`
  - one or more gauge configurations in NERSC / PyQUDA-readable format
- `benchmark/`
  - reserved for future leaderboard datasets

The repository `.gitignore` excludes concrete files under
`tasks/pion_2pt/dataset/quenched_wilson_b6_8x32/`,
`tasks/pion_2pt/dataset/quenched_wilson_b6_16x16/`,
`tasks/pion_2pt/dataset/local_test/`, and `tasks/pion_2pt/dataset/benchmark/`
so large local data stays untracked.

## Metadata contract

The fixed workflow expects `ensemble.json` with fields of this shape:

```json
{
  "name": "quenched_wilson_b6_8x32",
  "latt_size": [8, 8, 8, 32],
  "anisotropy": {
    "xi_0": 1.0,
    "nu": 1.0
  },
  "clover": {
    "mass": -0.038888,
    "csw_r": 1.02868,
    "csw_t": 1.02868,
    "t_boundary": -1
  },
  "source_times": [0, 16],
  "benchmark_momentum": [3, 3, 3],
  "gauge_glob": "wilson_b6.[0-9]*",
  "format": "nersc"
}
```

Notes:

- `benchmark_momentum` is a lattice momentum integer triplet for now, not a
  scale-set physical momentum.
- `gauge_glob` is resolved relative to the directory containing
  `ensemble.json`.
- `format` is currently expected to be `nersc`.

## Current benchmark target

Until scale setting is available, the boosted benchmark target is the pion with
momentum `(3, 3, 3)` in lattice units. Metrics will prioritize:

- better signal-to-noise ratio
- smaller excited-state contamination
