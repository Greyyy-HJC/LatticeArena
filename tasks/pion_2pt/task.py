"""Task registration for the pion_2pt benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.task import BenchmarkResult, TaskBase, register_task
from tasks.pion_2pt.benchmark.metrics import benchmark_submission
from tasks.pion_2pt.interface import PionInterpolatingOperator
from tasks.pion_2pt.tests.validation import validate_submission


@register_task
class Pion2PtTask(TaskBase):
    """Boosted pion two-point benchmark task."""

    @property
    def tests_path(self) -> Path:
        """Return the task test directory used for validation guidance."""
        return self.root / "tests"

    @property
    def name(self) -> str:
        return "pion_2pt"

    def validate(self, operator: Any) -> bool:
        """Check that the submission implements the required interface."""
        if not isinstance(operator, PionInterpolatingOperator):
            return False
        return validate_submission(operator)

    def benchmark(
        self, operator: Any, dataset_path: str | Path | None = None
    ) -> BenchmarkResult:
        if not isinstance(operator, PionInterpolatingOperator):
            raise TypeError(
                "pion_2pt benchmark expects a PionInterpolatingOperator instance."
            )
        if not self.validate(operator):
            raise ValueError(
                "pion_2pt benchmark refused to run because the submission "
                f"failed validation. Run `pytest {self.tests_path}` before "
                "benchmarking."
            )

        summary = benchmark_submission(
            operator,
            dataset_path=str(dataset_path or self.dataset_path / "test_small"),
        )
        return BenchmarkResult(
            task_name=self.name,
            submission_name=operator.meta.name,
            score=summary["score"],
            metrics=summary,
        )
