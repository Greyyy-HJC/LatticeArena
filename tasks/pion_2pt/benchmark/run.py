"""Benchmark driver for the pion_2pt task.

Uses synthetic correlators to evaluate operator quality without requiring
a live QUDA installation. Operator profiles modulate excited-state
contamination and noise in the synthetic data.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latticearena.leaderboard import save_result
from latticearena.task import BenchmarkResult

from tasks.pion_2pt.benchmark.metrics import benchmark_submission
from tasks.pion_2pt.interface import PionInterpolatingOperator


def load_submission(operator_name: str) -> PionInterpolatingOperator:
    """Import and instantiate a submission from operators/<name>.py."""

    module = importlib.import_module(f"tasks.pion_2pt.operators.{operator_name}")
    for value in module.__dict__.values():
        if (
            isinstance(value, type)
            and issubclass(value, PionInterpolatingOperator)
            and value is not PionInterpolatingOperator
        ):
            return value()
    raise ValueError(f"No PionInterpolatingOperator implementation found in operator '{operator_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pion_2pt benchmark")
    parser.add_argument("--operator", type=str, required=True, help="Operator module name under operators/")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic data generation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/pion_2pt/benchmark/results"),
        help="Directory to save the benchmark JSON result",
    )
    args = parser.parse_args()

    submission = load_submission(args.operator)
    summary = benchmark_submission(submission, seed=args.seed)
    result = BenchmarkResult(
        task_name="pion_2pt",
        operator_name=submission.meta.name,
        score=summary["score"],
        metrics=summary,
    )
    path = save_result(result, args.output_dir)

    print(
        json.dumps(
            {
                "task": result.task_name,
                "operator": result.operator_name,
                "score": result.score,
                "output": str(path),
                "metrics": result.metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
