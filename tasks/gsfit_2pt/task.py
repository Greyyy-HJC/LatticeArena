"""Task registration for the gsfit_2pt benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latticearena.task import BenchmarkResult, TaskBase, register_task

from .benchmark.core import benchmark_submission
from .interface import Pion2PtGroundStateFit, validate_config


@register_task
class Gsfit2PtTask(TaskBase):
    """Benchmark task for pion 2pt ground-state fit configurations."""

    @property
    def name(self) -> str:
        return "gsfit_2pt"

    def validate(self, operator: Any) -> bool:
        if not isinstance(operator, Pion2PtGroundStateFit):
            return False

        try:
            _ = operator.meta
            validate_config(operator.config)
        except (AttributeError, TypeError, ValueError):
            return False
        return True

    def benchmark(
        self,
        operator: Pion2PtGroundStateFit,
        dataset_path: str | Path | None = None,
    ) -> BenchmarkResult:
        summary = benchmark_submission(operator)
        return BenchmarkResult(
            task_name=self.name,
            operator_name=operator.meta.name,
            score=summary["score"],
            metrics=summary,
        )
