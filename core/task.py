"""Base class and registry for benchmark tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Result of running a benchmark on one submission."""

    task_name: str
    submission_name: str
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)


class TaskBase(ABC):
    """Abstract base class for all benchmark tasks.

    Each task lives in tasks/<name>/ and must implement validation
    and benchmarking for its optimization target.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Task identifier, matching the directory name under tasks/."""
        ...

    @property
    def root(self) -> Path:
        """Root directory of this task."""
        return Path(__file__).resolve().parent.parent / "tasks" / self.name

    @property
    def dataset_path(self) -> Path:
        return self.root / "dataset"

    @abstractmethod
    def validate(self, operator: Any) -> bool:
        """Run validation tests on a submission.

        Returns True if the submission passes all task constraints
        (gauge equivariance, correct output shape, etc.).
        """
        ...

    @abstractmethod
    def benchmark(
        self, operator: Any, dataset_path: str | Path | None = None
    ) -> BenchmarkResult:
        """Run the benchmark and return a scored result.

        Implementations should validate the submission before running the
        benchmark and raise an error when validation fails.

        Args:
            operator: An instance of the task's submission interface.
            dataset_path: Override path to gauge configs. Uses default if None.

        Returns:
            BenchmarkResult with score and detailed metrics.
        """
        ...


_task_registry: dict[str, type[TaskBase]] = {}


def register_task(task_cls: type[TaskBase]) -> type[TaskBase]:
    """Decorator to register a task class."""
    instance = task_cls()
    _task_registry[instance.name] = task_cls
    return task_cls


def get_task(name: str) -> TaskBase:
    """Get a task instance by name."""
    if name not in _task_registry:
        available = ", ".join(_task_registry.keys()) or "(none registered)"
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return _task_registry[name]()


def list_tasks() -> list[str]:
    """List all registered task names."""
    return list(_task_registry.keys())
