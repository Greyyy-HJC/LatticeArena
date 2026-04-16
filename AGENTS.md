# AGENTS

Guidance for coding agents working in the `LatticeArena` repository.

Primary references:

- [`CLAUDE.md`](CLAUDE.md) for repository workflow and naming conventions
- [`SPEC.md`](SPEC.md) for the formal task contract

When these differ in specificity, follow `SPEC.md` for task structure and
benchmark behavior, then keep repository-level docs aligned.

## Repository Rules

- Every task must expose the same six first-class components:
  - `dataset/` for task inputs and dataset documentation
  - `scripts/` for the framework-owned fixed workflow
  - `interface.py` for the optimization-target submission interface
  - `submissions/` for baseline and contributed implementations
  - `tests/` for legality and validation checks
  - `benchmark/` for scoring logic and benchmark outputs
- `benchmark = scripts + submission + metrics`
- `core/` is the shared framework layer for cross-task code such as task
  registration, leaderboard helpers, and reusable test utilities
- Incomplete tasks must still expose the full skeleton. Use explicit WIP
  placeholders instead of ad hoc structure.

## Environment

Use the local virtual environment when running Python tooling:

```bash
source .venv/bin/activate
```

Common commands:

```bash
pytest
pytest tasks/wilson_loop/tests/
python tasks/wilson_loop/benchmark/run.py --submission plain
python tasks/gsfit_2pt/scripts/fit.py --submission plain
```

## Quick Reference

- Python 3.10+
- Runtime dependencies include `numpy`, `scipy`, `gvar`, `lsqfit`, `gmpy2`,
  `pyquda`, and `pyquda-utils`
- Use the `submissions/` directory for all contributed implementations
- Use the `--submission` CLI flag everywhere
- Put reusable validation helpers in `tests/validation.py`
- Put score computation in `benchmark/metrics.py`
