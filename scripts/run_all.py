"""Run benchmarks across all registered tasks."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tasks  # noqa: F401  # import for task registration side effects
from core.task import get_task, list_tasks
from core.leaderboard import print_leaderboard, load_results


def main():
    parser = argparse.ArgumentParser(description="Run LatticeArena benchmarks")
    parser.add_argument("--task", type=str, default=None, help="Run a specific task (default: all)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else list_tasks()

    if not tasks:
        print("No tasks registered. Available tasks will appear as they are implemented.")
        return

    for task_name in tasks:
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
