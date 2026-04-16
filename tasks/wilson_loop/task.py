"""Task registration for the wilson_loop benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.task import BenchmarkResult, TaskBase, register_task
from tasks.wilson_loop.benchmark.metrics import benchmark_submission
from tasks.wilson_loop.interface import SpatialOperator
from tasks.wilson_loop.tests.validation import validate_submission


@register_task
class WilsonLoopTask(TaskBase):
    """Benchmark task for Wilson loop spatial operators."""

    @property
    def tests_path(self) -> Path:
        """Return the task test directory used for validation guidance."""
        return self.root / "tests"

    @property
    def name(self) -> str:
        return "wilson_loop"

    def validate(self, operator: Any) -> bool:
        if not isinstance(operator, SpatialOperator):
            return False
        return not validate_submission(operator)

    def benchmark(
        self, operator: Any, dataset_path: str | Path | None = None
    ) -> BenchmarkResult:
        if not isinstance(operator, SpatialOperator):
            raise TypeError("wilson_loop benchmark expects a SpatialOperator instance.")
        if not self.validate(operator):
            raise ValueError(
                "wilson_loop benchmark refused to run because the submission "
                f"failed validation. Run `pytest {self.tests_path}` before "
                "benchmarking."
            )

        summary = benchmark_submission(
            operator,
            dataset_path=str(dataset_path or self.dataset_path / "test_small"),
            r_values=[1, 2, 3],
            t_values=[0, 1, 2, 3, 4],
            max_configs=None,
        )
        return BenchmarkResult(
            task_name=self.name,
            submission_name=operator.meta.name,
            score=summary["score"],
            metrics=summary,
        )
