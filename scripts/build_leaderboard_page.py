"""Build a standalone leaderboard HTML page for all registered tasks."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import sys


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def format_score(result: object | None) -> str:
    return f"{result.score:.2f}" if result is not None else "No score yet"


def metric_from_result(result: object | None, key: str) -> str:
    if result is None:
        return "N/A"
    value = result.metrics.get(key)
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return "N/A"


def build_html() -> str:
    _bootstrap_repo_root()
    from core.leaderboard import collect_task_summaries

    summaries = collect_task_summaries()
    total_tasks = len(summaries)
    populated_tasks = sum(1 for summary in summaries if summary.best_result is not None)
    total_submissions = sum(len(summary.results) for summary in summaries)
    best_overall = max(
        (
            summary.best_result
            for summary in summaries
            if summary.best_result is not None
        ),
        key=lambda result: result.score,
        default=None,
    )

    hero_cards = [
        ("Registered Tasks", str(total_tasks)),
        ("Tasks With Scores", str(populated_tasks)),
        ("Tracked Submissions", str(total_submissions)),
        ("Best Overall Score", format_score(best_overall)),
    ]

    cards_html = "\n".join(
        f"""
        <article class="task-card">
          <p class="eyebrow">{escape(summary.task_name)}</p>
          <h2>{escape(summary.best_result.submission_name) if summary.best_result else "No submissions yet"}</h2>
          <p class="score">{format_score(summary.best_result)}</p>
          <div class="meta-grid">
            <span>Submissions</span><strong>{len(summary.results)}</strong>
            <span>Bias</span><strong>{metric_from_result(summary.best_result, "aggregate_relative_bias")}</strong>
            <span>Failure</span><strong>{metric_from_result(summary.best_result, "aggregate_failure_rate")}</strong>
          </div>
        </article>
        """
        for summary in summaries
    )

    table_rows: list[str] = []
    for summary in summaries:
        if summary.ranked_results:
            for rank, result in enumerate(summary.ranked_results, start=1):
                table_rows.append(
                    f"""
                    <tr>
                      <td>{escape(summary.task_name)}</td>
                      <td>{rank}</td>
                      <td>{escape(result.submission_name)}</td>
                      <td>{result.score:.4f}</td>
                      <td>{metric_from_result(result, "aggregate_relative_bias")}</td>
                      <td>{metric_from_result(result, "aggregate_relative_sigma")}</td>
                      <td>{metric_from_result(result, "aggregate_failure_rate")}</td>
                    </tr>
                    """
                )
        else:
            table_rows.append(
                f"""
                <tr>
                  <td>{escape(summary.task_name)}</td>
                  <td>N/A</td>
                  <td>No results yet</td>
                  <td>N/A</td>
                  <td>N/A</td>
                  <td>N/A</td>
                  <td>N/A</td>
                </tr>
                """
            )

    metrics_html = "\n".join(
        f"""
        <article class="metric-card">
          <p>{escape(label)}</p>
          <strong>{escape(value)}</strong>
        </article>
        """
        for label, value in hero_cards
    )

    table_html = "\n".join(table_rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LatticeArena Leaderboard</title>
  <style>
    :root {{
      --bg: #f6f0e8;
      --paper: rgba(255,255,255,0.78);
      --ink: #1b2630;
      --muted: #5b6a73;
      --accent: #cb5a2e;
      --accent-soft: #eab08c;
      --line: rgba(27,38,48,0.12);
      --shadow: 0 24px 60px rgba(34, 34, 34, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(203,90,46,0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(32,116,141,0.12), transparent 26%),
        linear-gradient(180deg, #fbf7f2 0%, var(--bg) 100%);
    }}
    .shell {{
      width: min(1180px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 40px 0 72px;
    }}
    .hero {{
      padding: 32px;
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,247,240,0.94));
      border: 1px solid rgba(255,255,255,0.8);
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 12px;
      font-size: clamp(2.2rem, 3.6vw, 4.2rem);
      line-height: 0.98;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      max-width: 760px;
      margin: 0;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.6;
    }}
    .metric-grid, .task-grid {{
      display: grid;
      gap: 18px;
      margin-top: 24px;
    }}
    .metric-grid {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .metric-card, .task-card {{
      padding: 20px;
      border-radius: 22px;
      background: var(--paper);
      border: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    .metric-card p, .task-card .eyebrow {{
      margin: 0 0 8px;
      font-size: 0.78rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .metric-card strong {{
      font-size: 1.8rem;
    }}
    .section-title {{
      margin: 42px 0 14px;
      font-size: 1.2rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .task-grid {{
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    .task-card h2 {{
      margin: 0 0 8px;
      font-size: 1.35rem;
    }}
    .task-card .score {{
      margin: 0 0 16px;
      font-size: 2rem;
      font-weight: 700;
      color: var(--accent);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: auto auto;
      gap: 10px 12px;
      color: var(--muted);
    }}
    .meta-grid strong {{
      color: var(--ink);
      text-align: right;
    }}
    .table-shell {{
      margin-top: 18px;
      overflow: hidden;
      border-radius: 24px;
      background: rgba(255,255,255,0.76);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    thead {{
      background: linear-gradient(90deg, rgba(203,90,46,0.12), rgba(32,116,141,0.08));
    }}
    th, td {{
      padding: 16px 18px;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    tbody tr:hover {{
      background: rgba(255,255,255,0.5);
    }}
    footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    @media (max-width: 720px) {{
      .shell {{ width: min(100vw - 24px, 1180px); padding-top: 24px; }}
      th, td {{ padding: 14px 12px; font-size: 0.92rem; }}
      .hero {{ padding: 24px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>LatticeArena Leaderboard</h1>
      <p>Track the strongest submission for each benchmark task, then drill into the ranked table to compare operators, fit strategies, and aggregate metrics across the arena.</p>
      <div class="metric-grid">
        {metrics_html}
      </div>
    </section>

    <h2 class="section-title">Task Leaders</h2>
    <section class="task-grid">
      {cards_html}
    </section>

    <h2 class="section-title">All Ranked Results</h2>
    <section class="table-shell">
      <table>
        <thead>
          <tr>
            <th>Task</th>
            <th>Rank</th>
            <th>Submission</th>
            <th>Score</th>
            <th>Bias</th>
            <th>Sigma</th>
            <th>Failure</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </section>

    <footer>Generated from local benchmark result files under <code>tasks/&lt;task&gt;/benchmark/results/</code>.</footer>
  </main>
</body>
</html>
"""


def main() -> None:
    _bootstrap_repo_root()
    parser = argparse.ArgumentParser(
        description="Build the standalone LatticeArena leaderboard page"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("site/leaderboard.html"),
        help="Output HTML file",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html())
    print(f"Wrote leaderboard page to {args.output}")


if __name__ == "__main__":
    main()
