import hashlib
import json
import math
from statistics import NormalDist

import pytest

from tasks.pion_projection.bounded_block_risk_anchor_robustness_exact import (
    ALPHA_ANCHOR,
    ANCHOR_SIGMA,
    RARE_PROBABILITIES,
    analyze_grid,
    build,
    exact_coverage,
    hoeffding_radius,
    normal_z_radius,
    two_point_support,
)


def test_two_point_generators_have_zero_mean_common_variance_and_declared_support():
    for p in RARE_PROBABILITIES.values():
        lower, upper = two_point_support(rare_probability=p)
        mean = p * upper + (1.0 - p) * lower
        variance = p * upper**2 + (1.0 - p) * lower**2
        assert mean == pytest.approx(0.0, abs=1e-15)
        assert variance == pytest.approx(ANCHOR_SIGMA**2)
        assert lower < 0 < upper


def test_interval_radii_match_two_sided_normal_and_hoeffding_formulas():
    n = 16
    z = NormalDist().inv_cdf(1.0 - ALPHA_ANCHOR / 2.0)
    lower, upper = two_point_support(rare_probability=0.005)
    assert normal_z_radius(samples=n) == pytest.approx(
        z * ANCHOR_SIGMA / math.sqrt(n)
    )
    assert hoeffding_radius(samples=n, lower=lower, upper=upper) == pytest.approx(
        (upper - lower) * math.sqrt(math.log(2 / ALPHA_ANCHOR) / (2 * n))
    )


def test_exact_enumeration_exposes_rare_spike_z_interval_undercoverage():
    normal_radius = normal_z_radius(samples=16)
    coverage, mass = exact_coverage(
        samples=16, rare_probability=0.005, radius=normal_radius,
    )
    assert mass == pytest.approx(1.0)
    assert coverage == pytest.approx((1.0 - 0.005) ** 16)
    assert coverage == pytest.approx(0.92293112397)


def test_hoeffding_interval_has_at_least_nominal_exact_coverage_on_grid():
    rows = analyze_grid()
    assert len(rows) == 12
    assert min(r["hoeffding_exact_coverage"] for r in rows) >= (
        1.0 - ALPHA_ANCHOR
    )
    assert min(r["normal_z_exact_coverage"] for r in rows) < 0.99


def test_build_writes_exact_coverage_plot_denominators_and_hashes(tmp_path):
    output = tmp_path / "anchor-robustness"
    summary = build(output)
    assert summary["cells"] == 12
    assert summary["physical_score"] is None
    assert summary["hoeffding_coverage_all_at_least_nominal"] is True
    assert (output / "coverage_and_radius_cost.png").is_file()
    denominator_csv = (output / "denominator_summary.csv").read_text()
    assert "exact binomial count state" in denominator_csv
    assert "Hypothetical iid observations" in denominator_csv
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert len(manifest) == 11
    for name, digest in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    report = (output / "README.md").read_text()
    assert "SYNTHETIC EXACT ENUMERATION ONLY" in report
    assert "92.2931%" in report
    assert "does not show that project anchor errors are iid" in report
    assert "physical_score=null" in report
