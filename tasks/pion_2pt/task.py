"""Task registration for the pion_2pt benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latticearena.task import BenchmarkResult, TaskBase, register_task


@register_task
class Pion2PtTask(TaskBase):
    """Boosted pion two-point benchmark task."""

    @property
    def name(self) -> str:
        return "pion_2pt"

    def validate(self, operator: Any) -> bool:
        """Placeholder validation hook.

        Validation is currently implemented through pytest under
        ``tasks/pion_2pt/tests``. This method is reserved for future API wiring.
        """
        return hasattr(operator, "build") and hasattr(operator, "setup")

    def benchmark(self, operator: Any, dataset_path: str | Path | None = None) -> BenchmarkResult:
        """Benchmark entrypoint placeholder.

        Full benchmark wiring is provided in tasks/pion_2pt/benchmark/run.py.
        """
        raise NotImplementedError(
            "Use `python tasks/pion_2pt/benchmark/run.py --operator <name>` "
            "for pion_2pt benchmarking."
        )
