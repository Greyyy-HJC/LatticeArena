"""Validation tests for pion_2pt operators."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from latticearena.testing import identity_gauge_field
from tasks.pion_2pt.operators.plain import PlainBoostedPion


def test_plain_operator_shapes() -> None:
    latt_size = (8, 8, 8, 32)
    op = PlainBoostedPion(sigma=2.5)
    gauge = identity_gauge_field(latt_size)

    op.setup(gauge, latt_size=latt_size, lattice_spacing_fm=0.09)
    components = op.build(gauge, momentum_gev=(1.0, 0.0, 0.0), t_source=0)

    assert components.source_profile.shape == (8, 8, 8)
    assert components.sink_profile.shape == (8, 8, 8)
    assert components.gamma_matrix.shape == (4, 4)
    assert np.iscomplexobj(components.source_profile)


def test_plain_operator_normalization() -> None:
    latt_size = (8, 8, 8, 32)
    op = PlainBoostedPion(sigma=2.0)
    gauge = identity_gauge_field(latt_size)

    op.setup(gauge, latt_size=latt_size, lattice_spacing_fm=0.09)
    components = op.build(gauge, momentum_gev=(1.0, 0.0, 0.0), t_source=0)

    src_norm = np.linalg.norm(components.source_profile)
    sink_norm = np.linalg.norm(components.sink_profile)
    assert np.isclose(src_norm, 1.0, atol=1e-12)
    assert np.isclose(sink_norm, 1.0, atol=1e-12)


def test_plain_operator_requires_setup() -> None:
    op = PlainBoostedPion()
    gauge = identity_gauge_field((4, 4, 4, 8))

    try:
        op.build(gauge, momentum_gev=(1.0, 0.0, 0.0), t_source=0)
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError when build() is called before setup().")


def test_benchmark_smoke_cli() -> None:
    """Benchmark CLI produces valid JSON with a positive score."""
    result = subprocess.run(
        [sys.executable, "tasks/pion_2pt/benchmark/run.py", "--operator", "plain"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["task"] == "pion_2pt"
    assert data["operator"] == "plain"
    assert data["score"] > 0
    assert len(data["metrics"]["per_scenario"]) == 3
