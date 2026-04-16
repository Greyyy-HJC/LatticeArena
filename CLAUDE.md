# LatticeArena

Multi-task benchmark repository for lattice QCD optimization targets.

## Build and Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Canonical Task Layout

Every task under `tasks/<name>/` must follow this structure:

```text
tasks/<task_name>/
  dataset/             Task inputs and dataset contract
  scripts/             Framework-owned fixed workflow
  interface.py         Submission interface for the optimization target
  submissions/         Baselines and contributed submissions
  tests/               Validation and legality checks
    validation.py      Reusable validation helpers
    test_validation.py Main validation test module
  benchmark/
    metrics.py         Score computation
    run.py             Benchmark CLI entrypoint
    results/           Saved benchmark outputs
```

## Shared Framework

`core/` contains repository-level framework code shared by multiple tasks. Use
it for cross-task abstractions such as:

- task registration and registry helpers
- shared benchmark result types
- leaderboard utilities
- reusable testing helpers

Do not put task-specific measurement or analysis logic in `core/`; that logic
belongs under the relevant task in `tasks/<name>/`.

## Naming Rules

- Use `submission` as the generic cross-task term.
- Keep task-specific physics names in class names and task docs.
- Use the `submissions/` directory for contributed implementations.
- Use the `--submission` CLI flag consistently.
- `benchmark/run.py` loads a submission, runs the fixed workflow, calls
  `benchmark/metrics.py`, and writes a scored result.
- `scripts/` contains framework-owned code only. It is not the submission area.

## Authoring Rules

- Contributors should only need to modify files under `tasks/<name>/submissions/`
  for ordinary benchmark participation.
- Every submission must pass `tasks/<name>/tests/` before it is benchmarked.
- Live benchmark runners must re-check validation before scoring and exit
  early on invalid submissions.
- Tasks that are still under construction should keep the full skeleton and use
  explicit `NotImplementedError` placeholders where behavior is pending.

## Documentation hygiene (required)

- After completing any coding task, **always check whether `PROJECT_LOG.md`
  should be updated** (design decisions, task contract/interface changes,
  migrations, or any important behavioral change).

## Language policy (required)

- All repository documentation must be written in **English** (no Chinese).

## Current Tasks

- `wilson_loop`: reference measurement-task layout
- `pion_2pt`: WIP measurement task with interface and validation in place
- `gsfit_2pt`: analysis task with synthetic dataset, fixed fit pipeline, and
  benchmark scoring
