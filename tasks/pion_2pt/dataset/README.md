# pion_2pt Dataset Contract

This task will eventually require gauge-field data together with either stored
quark propagators or a supported workflow for generating them on demand.

The intended dataset layout is:

- `test_small/` for validation and quick regression checks
- `benchmark/` for leaderboard scoring

The benchmark focus is boosted pions with representative momentum shells around
`|p| ~ 1 GeV`.

Current status:

- dataset contract: documented
- concrete public dataset packaging: pending
- fixed measurement workflow integration: pending
