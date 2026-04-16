"""Task registration for the pion_2pt benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latticearena.task import BenchmarkResult, TaskBase, register_task

from tasks.pion_2pt.benchmark.metrics import benchmark_submission
from tasks.pion_2pt.interface import PionInterpolatingOperator


@register_task
class Pion2PtTask(TaskBase):
    """Boosted pion two-point benchmark task."""

    @property
    def name(self) -> str:
        return "pion_2pt"

    def validate(self, operator: Any) -> bool:
        """Check that the operator implements the required interface."""
        return (
            isinstance(operator, PionInterpolatingOperator)
            or (hasattr(operator, "build") and hasattr(operator, "setup"))
        )

    def benchmark(self, operator: Any, dataset_path: str | Path | None = None) -> BenchmarkResult:
        """Run the synthetic pion_2pt benchmark."""
        summary = benchmark_submission(operator)
        return BenchmarkResult(
            task_name="pion_2pt",
            operator_name=getattr(operator, "meta", None) and operator.meta.name or "unknown",
            score=summary["score"],
            metrics=summary,
        )
