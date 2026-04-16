"""Task registration for the gsfit_2pt benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.task import BenchmarkResult, TaskBase, register_task

from .benchmark.metrics import benchmark_submission
from .interface import Pion2PtGroundStateFit
from .tests.validation import validate_submission


@register_task
class Gsfit2PtTask(TaskBase):
    """Benchmark task for pion 2pt ground-state fit configurations."""

    @property
    def tests_path(self) -> Path:
        """Return the task test directory used for validation guidance."""
        return self.root / "tests"

    @property
    def name(self) -> str:
        return "gsfit_2pt"

    def validate(self, operator: Any) -> bool:
        return validate_submission(operator)

    def benchmark(
        self,
        operator: Pion2PtGroundStateFit,
        dataset_path: str | Path | None = None,
    ) -> BenchmarkResult:
        if not isinstance(operator, Pion2PtGroundStateFit):
            raise TypeError(
                "gsfit_2pt benchmark expects a Pion2PtGroundStateFit instance."
            )
        if not self.validate(operator):
            raise ValueError(
                "gsfit_2pt benchmark refused to run because the submission "
                f"failed validation. Run `pytest {self.tests_path}` before "
                "benchmarking."
            )
        summary = benchmark_submission(operator)
        return BenchmarkResult(
            task_name=self.name,
            submission_name=operator.meta.name,
            score=summary["score"],
            metrics=summary,
        )
