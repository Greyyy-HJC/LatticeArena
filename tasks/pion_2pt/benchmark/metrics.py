"""Benchmark scoring contract for pion_2pt.

The full benchmark is intentionally not implemented yet. This module exists so
every task exposes the same benchmark layout:

- ``benchmark/run.py`` for the CLI entrypoint
- ``benchmark/metrics.py`` for score computation

Once the fixed PyQUDA measurement pipeline is complete, this module will
convert measured correlators into benchmark metrics and a composite score.
"""

from __future__ import annotations

from typing import Any


def compute_metrics(measured: dict[str, Any]) -> dict[str, Any]:
    """Compute benchmark metrics from measured pion two-point data."""

    raise NotImplementedError(
        "pion_2pt metrics are not implemented yet. "
        "Complete scripts/measure.py before adding score computation."
    )


def benchmark_submission(submission: Any, measured: dict[str, Any] | None = None) -> dict[str, Any]:
    """Benchmark a pion_2pt submission once measured correlators are available."""

    raise NotImplementedError(
        "pion_2pt benchmark scoring is not implemented yet. "
        "Complete scripts/measure.py and benchmark/metrics.py together."
    )
