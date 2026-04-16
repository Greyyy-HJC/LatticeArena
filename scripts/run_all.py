"""Run benchmarks across all registered tasks."""

import argparse
from pathlib import Path
import sys


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main():
    _bootstrap_repo_root()
    import tasks  # noqa: F401  # import for task registration side effects
    from core.leaderboard import load_results, print_leaderboard
    from core.task import get_task, list_tasks

    parser = argparse.ArgumentParser(description="Run LatticeArena benchmarks")
    parser.add_argument(
        "--task", type=str, default=None, help="Run a specific task (default: all)"
    )
    args = parser.parse_args()

    selected_tasks = [args.task] if args.task else list_tasks()

    if not selected_tasks:
        print(
            "No tasks registered. Available tasks will appear as they are implemented."
        )
        return

    for task_name in selected_tasks:
        task = get_task(task_name)
        results_dir = task.root / "benchmark" / "results"
        if results_dir.exists():
            results = load_results(results_dir)
            if results:
                print(f"\n=== {task_name} ===")
                print_leaderboard(results)
            else:
                print(f"\n=== {task_name} === (no results yet)")
        else:
            print(f"\n=== {task_name} === (no results directory)")


if __name__ == "__main__":
    main()
