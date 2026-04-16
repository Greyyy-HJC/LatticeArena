"""Benchmark driver for pion_2pt task (WIP).

The pion_2pt benchmark requires a measurement pipeline that computes pion
two-point correlators from gauge configurations using PyQUDA.  Once
``scripts/measure.py`` is implemented, this driver will:

1. Load gauge configurations from ``dataset/``
2. Run the measurement pipeline with the submitted implementation
3. Compute benchmark metrics (SNR, plateau quality, excited-state contamination)
4. Output a scored JSON result for the leaderboard
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pion_2pt benchmark (WIP)")
    parser.add_argument("--submission", type=str, required=True, help="Submission module name")
    parser.parse_args()

    print(
        "pion_2pt benchmark is not yet implemented. "
        "Requires a PyQUDA measurement pipeline in scripts/measure.py."
    )


if __name__ == "__main__":
    main()
