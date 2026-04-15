"""Leaderboard: aggregate and rank benchmark results across tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path

from .task import BenchmarkResult


@dataclass(frozen=True)
class TaskLeaderboardSummary:
    """Summary of one task's current leaderboard state."""

    task_name: str
    results_dir: Path
    results: list[BenchmarkResult]

    @property
    def ranked_results(self) -> list[BenchmarkResult]:
        return sorted(self.results, key=lambda result: result.score, reverse=True)

    @property
    def best_result(self) -> BenchmarkResult | None:
        ranked = self.ranked_results
        return ranked[0] if ranked else None


def save_result(result: BenchmarkResult, output_dir: str | Path) -> Path:
    """Save a benchmark result to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.operator_name}.json"
    path.write_text(json.dumps(asdict(result), indent=2, default=str))
    return path


def load_results(results_dir: str | Path) -> list[BenchmarkResult]:
    """Load all benchmark results from a directory."""
    results_dir = Path(results_dir)
    results = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        results.append(BenchmarkResult(**data))
    return results


def print_leaderboard(results: list[BenchmarkResult]) -> None:
    """Print a ranked leaderboard table to stdout."""
    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    print(f"{'Rank':<6}{'Operator':<30}{'Score':<12}")
    print("-" * 48)
    for i, r in enumerate(ranked, 1):
        print(f"{i:<6}{r.operator_name:<30}{r.score:<12.4f}")


def collect_task_summaries(task_names: list[str] | None = None) -> list[TaskLeaderboardSummary]:
    """Collect leaderboard summaries across registered tasks."""

    import tasks  # noqa: F401  # task registration side effects
    from .task import get_task, list_tasks

    selected_tasks = task_names if task_names is not None else list_tasks()
    summaries: list[TaskLeaderboardSummary] = []

    for task_name in selected_tasks:
        task = get_task(task_name)
        results_dir = task.root / "benchmark" / "results"
        results = load_results(results_dir) if results_dir.exists() else []
        summaries.append(TaskLeaderboardSummary(task_name=task_name, results_dir=results_dir, results=results))

    return summaries
