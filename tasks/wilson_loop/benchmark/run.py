"""Benchmark driver for the wilson_loop task."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leaderboard import save_result
from core.task import BenchmarkResult

from tasks.wilson_loop.benchmark.metrics import benchmark_submission
from tasks.wilson_loop.interface import SpatialOperator
from tasks.wilson_loop.scripts.measure import parse_value_list


def load_submission(submission_name: str) -> SpatialOperator:
    """Import and instantiate a submission from submissions/<name>.py."""

    module = importlib.import_module(f"tasks.wilson_loop.submissions.{submission_name}")
    for value in module.__dict__.values():
        if (
            isinstance(value, type)
            and issubclass(value, SpatialOperator)
            and value is not SpatialOperator
        ):
            return value()
    raise ValueError(
        f"No SpatialOperator submission found in module '{submission_name}'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wilson_loop benchmark")
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Submission module name under submissions/",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("tasks/wilson_loop/dataset/test_small"),
        help="Gauge config directory or one cfg_XXXX.npy file in task ordering.",
    )
    parser.add_argument(
        "--r-values",
        type=str,
        default="1,2,3",
        help="Comma-separated spatial separations.",
    )
    parser.add_argument(
        "--t-values",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated temporal extents.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on the number of configs loaded from a dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/wilson_loop/benchmark/results"),
        help="Directory to save the benchmark JSON result",
    )
    args = parser.parse_args()

    submission = load_submission(args.submission)
    summary = benchmark_submission(
        submission,
        dataset_path=str(args.dataset_path),
        r_values=parse_value_list(args.r_values),
        t_values=parse_value_list(args.t_values),
        max_configs=args.max_configs,
    )
    result = BenchmarkResult(
        task_name="wilson_loop",
        submission_name=submission.meta.name,
        score=summary["score"],
        metrics=summary,
    )
    path = save_result(result, args.output_dir)

    print(
        json.dumps(
            {
                "task": result.task_name,
                "submission": result.submission_name,
                "score": result.score,
                "output": str(path),
                "metrics": result.metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
