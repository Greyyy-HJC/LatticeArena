"""Task registration for the pion_2pt benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.task import BenchmarkResult, TaskBase, register_task
from tasks.pion_2pt.interface import PionInterpolatingOperator
from tasks.pion_2pt.tests.validation import validate_submission


@register_task
class Pion2PtTask(TaskBase):
    """Boosted pion two-point benchmark task (WIP)."""

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
        """Benchmark is not yet implemented and requires a PyQUDA measurement pipeline."""
        raise NotImplementedError(
            "pion_2pt benchmark is not yet implemented. "
            "Requires a PyQUDA measurement pipeline in scripts/measure.py."
        )
