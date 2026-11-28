import numpy as np
import pytest

from tasks.wilson_loop import f_lane_absolute_force_bootstrap_diagnostic as bootstrap
from tasks.wilson_loop.f_lane_two_sector_fit_candidate import _validate_raw_snapshot


def test_circular_mbb_indices_are_deterministic_and_wrap_contiguously():
    first = bootstrap._circular_mbb_indices(np.random.default_rng(4), 16, 5)
    second = bootstrap._circular_mbb_indices(np.random.default_rng(4), 16, 5)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (16,)
    for start in range(0, len(first) - 1, 5):
        block = first[start:min(start + 5, len(first))]
        if len(block) > 1:
            np.testing.assert_array_equal(np.diff(block) % 16, np.ones(len(block) - 1))


def test_percentile_interval_is_extreme_order_and_abstains_on_any_failed_refit():
    interval = bootstrap._percentile_interval(list(map(float, range(39))))
    assert interval["status"] == "VALID"
    assert interval["lower_order_rank_1based"] == 1
    assert interval["upper_order_rank_1based"] == 39
    assert interval["lower"] == 0.0
    assert interval["upper"] == 38.0
    failed = bootstrap._percentile_interval([0.0, None, 2.0])
    assert failed["status"] == "ABSTAIN_BOOTSTRAP_FAILURES"
    assert failed["valid_draws"] == 2
    assert failed["failed_or_missing_draws"] == 1


def test_bootstrap_snapshot_roundtrips_hashed_hermitian_matrix_axes(tmp_path):
    matrices = np.zeros((96, 2, 2, 5, 3, 3), dtype=np.complex128)
    for index in range(3):
        matrices[..., index, index] = 1.0 + index
    contract_bytes = bootstrap.SOURCE_PATHS["lane_contract_v2.json"].read_bytes()
    snapshot, hashes = bootstrap._snapshot_from_matrices(
        tmp_path, matrices, contract_bytes, "F-synthetic-open-bootstrap-reader-test",
    )
    manifest, observed = _validate_raw_snapshot(
        snapshot, manifest_name="manifest", matrix_name="raw_matrices",
        contract_name="lane_contract",
    )
    assert manifest["matrix_sha256"] == hashes["raw_matrices"]
    assert observed.shape == matrices.shape
    np.testing.assert_array_equal(observed, matrices)


def test_percentile_interval_rejects_invalid_alpha_and_insufficient_tail_resolution():
    with pytest.raises(ValueError, match="alpha"):
        bootstrap._percentile_interval([1.0, 2.0], alpha=1.0)
    interval = bootstrap._percentile_interval([1.0] * 10)
    assert interval["status"] == "ABSTAIN_INSUFFICIENT_TAIL_RESOLUTION"
