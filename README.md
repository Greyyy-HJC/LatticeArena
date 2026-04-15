# LatticeArena

A public benchmark for optimizing lattice QCD operators. Submit your operator design, pass the tests, and climb the leaderboard.

## How It Works

LatticeArena hosts a collection of **tasks**. Each task defines a lattice QCD measurement where the choice of operator matters — better operators give cleaner signals and more accurate results.

For each task, we provide:
- **Dataset** — lattice gauge ensembles for testing and scoring
- **Measurement scripts** — complete PyQUDA code for the physics pipeline
- **A clear interface** — the one function you need to implement
- **Automated tests** — your operator must satisfy physics constraints (e.g., gauge covariance)
- **Benchmark scoring** — a numerical score that ranks your submission on the leaderboard

You only need to write one file: your operator implementation.

## Available Tasks

| Task | What you optimize | Dataset | Status |
|---|---|---|---|
| [`wilson_loop`](tasks/wilson_loop/) | Spatial Wilson line operator for static quark-antiquark potential | Pure gauge SU(3) | In progress |

## Quick Start

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests for an existing operator
pytest tasks/wilson_loop/tests/

# Run the benchmark
python tasks/wilson_loop/benchmark/run.py --operator plain
```

## Contributing an Operator

1. Pick a task (e.g., `wilson_loop`)
2. Read the task's `README.md` for physics background
3. Look at existing operators in `tasks/<task>/operators/` for examples
4. Create your operator file in the same directory, implementing the interface
5. Run the validation tests: `pytest tasks/<task>/tests/ -k your_operator`
6. Run the benchmark to get your score
7. Open a PR

## Project Structure

```
tasks/<task_name>/
  dataset/          Gauge ensembles (download separately, see README inside)
  scripts/          Full measurement pipeline (PyQUDA)
  interface.py      The abstract class you implement
  operators/        Your submission goes here
  tests/            Must pass before scoring
  benchmark/        Computes your score
```

## Requirements

- Python 3.10+
- [PyQUDA](https://github.com/CLQCD/PyQUDA) with a working QUDA installation
- numpy, scipy

## License

MIT
