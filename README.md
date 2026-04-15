# LatticeArena

A public benchmark for optimizing lattice QCD operators. Submit your operator design, pass the tests, and climb the leaderboard.

## Why This Exists

LatticeArena is designed to be useful in two complementary ways.

Internally, once a benchmark is well-defined, it becomes much easier to automate research loops: an agent can generate candidate ideas, run the benchmark, compare results on the leaderboard, and keep iterating toward better designs with minimal manual bookkeeping.

Externally, the same benchmark suite acts as an optimization testbed for the broader community. If someone has a new ansatz, fitting strategy, architecture, or other unconventional idea, they can plug it into a controlled task and immediately see how it performs against a shared baseline.

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
| [`pion_2pt`](tasks/pion_2pt/) | Boosted pion interpolating operator for pion two-point correlators (target \|p\| ~ 1 GeV) | Gauge + quark propagators | In progress |
| [`two_pt_gsfit`](tasks/two_pt_gsfit/) | Ground-state fit configuration for pion two-point correlators (`fit range`, priors, `N_states`) | Synthetic bootstrap/jackknife-like pion 2pt samples with known truth | In progress |

## Quick Start

```bash
# Create a local virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel

# Install the project and dev tooling
pip install -e ".[dev]"

# Run tests for an existing operator
pytest tasks/wilson_loop/tests/

# Run the benchmark
python tasks/wilson_loop/benchmark/run.py --operator plain
```

## Environment Setup

All Python dependencies are declared in [pyproject.toml](/home/genie/git/LatticeArena/pyproject.toml), so users can create their own `.venv` and install directly from the repo.

### Python Dependencies

- Runtime deps: `numpy`, `scipy`, `pyquda`, `pyquda-utils`
- Dev deps: `pytest`, `pytest-cov`
- Optional viz deps: `matplotlib`

Install commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[dev]"
```

If your system Python does not provide `venv`, you can create the same `.venv` with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### QUDA / PyQUDA Prerequisites

`pyquda` is a Python wrapper around a local QUDA installation. That means `pip install -e ".[dev]"` will only succeed after QUDA itself is installed on the machine and visible to the build.

Before installing `pyquda`, make sure you have:

- a working QUDA build
- `libquda.so` available under the QUDA install tree
- `QUDA_PATH` pointing to that install root
- `LD_LIBRARY_PATH` including `$QUDA_PATH/lib`

Typical shell setup:

```bash
export QUDA_PATH=/abs/path/to/quda
export LD_LIBRARY_PATH="$QUDA_PATH/lib:$LD_LIBRARY_PATH"
```

Then install the project inside `.venv`:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

If QUDA is not installed yet, tasks that depend on `pyquda` will not run. The pure fitting task [`two_pt_gsfit`](tasks/two_pt_gsfit/) is structurally independent of QUDA, but the repository-wide dependency set still declares `pyquda` and `pyquda-utils` as standard runtime requirements.

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
