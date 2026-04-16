"""Benchmark driver for the pion_2pt task."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leaderboard import save_result
from core.task import BenchmarkResult

from tasks.pion_2pt.benchmark.metrics import benchmark_submission
from tasks.pion_2pt.scripts.measure import (
    load_submission,
    parse_momentum_list,
    parse_time_list,
)
from tasks.pion_2pt.task import Pion2PtTask


def _default_dataset_path() -> Path:
    candidate = Path("tasks/pion_2pt/dataset/quenched_wilson_b6_16x16")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "No default pion_2pt dataset found. Provide --dataset-path pointing to a "
        "dataset directory (containing ensemble.json) or an ensemble.json file. "
        "Expected: tasks/pion_2pt/dataset/quenched_wilson_b6_16x16."
    )


def run_validation_tests(*, skip_tests: bool) -> None:
    """Run pion_2pt validation tests before benchmarking unless skipped."""

    if skip_tests:
        print(
            "WARNING: Skipping pytest validation gate via --skip-tests. "
            "Benchmark results may be invalid."
        )
        return

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tasks/pion_2pt/tests/test_validation.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "pion_2pt benchmark refused to run because pre-benchmark "
            "validation tests failed "
            f"(exit code {result.returncode}). "
            "Fix tests first or rerun with --skip-tests."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pion_2pt benchmark")
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Submission module name under submissions/",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=_default_dataset_path(),
        help="Dataset directory or ensemble.json path.",
    )
    parser.add_argument(
        "--momenta",
        type=str,
        default=None,
        help="Semicolon-separated lattice momenta, for example '3,3,3;0,0,0'.",
    )
    parser.add_argument(
        "--source-times",
        type=str,
        default=None,
        help="Comma-separated source times. Defaults to ensemble metadata.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on the number of gauge configurations to load.",
    )
    parser.add_argument(
        "--resource-path",
        type=Path,
        default=None,
        help="Optional PyQUDA resource cache path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tasks/pion_2pt/benchmark/results"),
        help="Directory to save the benchmark JSON result.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help=(
            "Skip pre-benchmark pytest validation "
            "(tasks/pion_2pt/tests/test_validation.py)"
        ),
    )
    args = parser.parse_args()

    run_validation_tests(skip_tests=args.skip_tests)
    submission = load_submission(args.submission)
    task = Pion2PtTask()
    if not task.validate(submission):
        raise SystemExit(
            "pion_2pt benchmark refused to run because the submission failed "
            f"validation. Run `pytest {task.tests_path}` before benchmarking."
        )
    summary = benchmark_submission(
        submission,
        dataset_path=str(args.dataset_path),
        momentum_modes=parse_momentum_list(args.momenta) if args.momenta else None,
        source_times=parse_time_list(args.source_times) if args.source_times else None,
        max_configs=args.max_configs,
        resource_path=args.resource_path,
    )
    result = BenchmarkResult(
        task_name="pion_2pt",
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
