# LatticeArena

Multi-task benchmark for lattice QCD operator optimization.

## Build & Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # run all task validation tests
```

## Architecture

```
latticearena/          Shared framework (TaskBase ABC, leaderboard)
tasks/<name>/          Self-contained benchmark tasks
  dataset/             Gauge configurations (gitignored, see README)
  scripts/             Runnable PyQUDA measurement scripts
  interface.py         Optimization interface ABC
  operators/           Submitted operator implementations
  tests/               Validation tests (legality checks)
  benchmark/           Metrics and scoring
```

## Current Tasks

- `wilson_loop` — Optimize spatial Wilson line operators for ground-state overlap in static quark-antiquark Wilson loops. Based on arXiv:2602.02436.
- `pion_2pt` — Optimize boosted pion interpolating operators to improve SNR and reduce excited-state contamination for pion two-point correlators (focus: |p| ~ 1 GeV).

## Conventions

- All lattice code uses PyQUDA (https://github.com/CLQCD/PyQUDA)
- Gauge fields stored in PyQUDA's even/odd ordering: `(Nd, 2, Lt/2, Lz, Ly, Lx/2, Nc, Nc)`
- Contributors only modify files in `tasks/<name>/operators/`
- Every operator must pass `tasks/<name>/tests/` before benchmark scoring
- Python 3.10+, type hints encouraged
