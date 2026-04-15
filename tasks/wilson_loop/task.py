"""Task registration for the wilson_loop benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latticearena.task import BenchmarkResult, TaskBase, register_task


@register_task
class WilsonLoopTask(TaskBase):
    """Benchmark task for Wilson loop spatial operators."""

    @property
    def name(self) -> str:
        return "wilson_loop"

    def validate(self, operator: Any) -> bool:
        return hasattr(operator, "compute") and hasattr(operator, "setup")

    def benchmark(self, operator: Any, dataset_path: str | Path | None = None) -> BenchmarkResult:
        raise NotImplementedError(
            "Use the wilson_loop benchmark scripts once the measurement pipeline is implemented."
        )
