"""Leaderboard: aggregate and rank benchmark results across tasks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .task import BenchmarkResult


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
