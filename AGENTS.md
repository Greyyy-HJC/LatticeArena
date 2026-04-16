# AGENTS

Guidance for coding agents working in the `LatticeArena` repository.

Primary references:

- [`CLAUDE.md`](CLAUDE.md) — repository workflow, conventions, and agent editing guidance
- [`SPEC.md`](SPEC.md) — formal benchmark/task contracts and submission rules

When these differ in specificity, prefer the more local and task-facing contract
in `SPEC.md`, then keep repository-level wording in sync when editing docs.

## Environment

Use the local virtual environment when running Python tooling:

```bash
source .venv/bin/activate
```

Common commands:

```bash
pytest                                              # all tests
pytest tasks/wilson_loop/tests/                     # single task
python tasks/wilson_loop/benchmark/run.py --operator plain
```

## Quick Reference

- Python 3.10+; runtime deps include `numpy`, `scipy`, `gvar`, `lsqfit`, `gmpy2`, `pyquda`, `pyquda-utils`
- Repo layout, task model, data conventions, and editing guidance: see [`CLAUDE.md`](CLAUDE.md)
- Task interfaces, validation, and scoring contracts: see [`SPEC.md`](SPEC.md)
