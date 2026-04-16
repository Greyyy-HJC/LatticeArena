# AGENTS

Guidance for coding agents working in the `LatticeArena` repository.

Primary references:

- [`CLAUDE.md`](CLAUDE.md) for repository workflow and naming conventions
- [`SPEC.md`](SPEC.md) for the formal task contract
- [`PROJECT_LOG.md`](PROJECT_LOG.md) for long-form task notes, design decisions, and change history
- [`PYQUDA_NOTES.md`](PYQUDA_NOTES.md) for repo-wide PyQUDA / `pyquda-utils` integration notes

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

## Pre-push Check

- Before every `git push`, check the latest GitHub Actions status for your
  branch or PR.
- Run the relevant local validation first. At minimum, use `ruff check .`,
  `ruff format --check .`, and the task-specific `pytest` target for the files
  you touched so you do not stack new commits on top of a known red CI.

## Quick Reference

- Python 3.10+
- Runtime dependencies include `numpy`, `scipy`, `gvar`, `lsqfit`, `gmpy2`,
  `pyquda`, and `pyquda-utils`
- Use the `submissions/` directory for all contributed implementations
- Use the `--submission` CLI flag everywhere
- Put reusable validation helpers in `tests/validation.py`
- Put score computation in `benchmark/metrics.py`
- Treat `benchmark/run.py` as a validation gate: live benchmark runners should
  refuse to score submissions that fail task validation

## Documentation hygiene (required)

- After completing any coding task (feature / refactor / bugfix / docs), **always
  do a quick check**: should `PROJECT_LOG.md` be updated?
  - Update it when you made a **design decision**, changed a **task contract /
    interface / validation expectation**, performed a **migration**, or made any
    **important behavioral change** that future agents should not have to
    rediscover.

## Language policy (required)

- All repository documentation must be written in **English** (no Chinese), so it
  is searchable and consistent across contributors and agents.
