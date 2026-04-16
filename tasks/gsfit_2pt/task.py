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
    def name(self) -> str:
        return "gsfit_2pt"

    def validate(self, operator: Any) -> bool:
        return validate_submission(operator)

    def benchmark(
        self,
        operator: Pion2PtGroundStateFit,
        dataset_path: str | Path | None = None,
    ) -> BenchmarkResult:
        summary = benchmark_submission(operator)
        return BenchmarkResult(
            task_name=self.name,
            submission_name=operator.meta.name,
            score=summary["score"],
            metrics=summary,
        )
