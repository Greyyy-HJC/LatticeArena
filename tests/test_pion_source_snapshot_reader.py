from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from core.source_integrity import capture_source_snapshot
from tasks.pion_projection.common import fingerprint, load_data
from tasks.pion_projection.spec import bundle_for, task_spec_for
from tasks.pion_projection.common import protocol as default_protocol
from tasks.pion_projection import experiment


def test_load_data_parses_the_captured_npz_bytes_after_path_changes(tmp_path):
    protocol = {
        "development": [2, 3],
        "powers": [0, 2],
        "geometry": [4, 4, 4, 4],
    }
    expected = np.arange(16, dtype=np.float64).reshape(2, 2, 4)
    payload_stream = BytesIO()
    np.savez_compressed(
        payload_stream,
        ids=np.asarray([2, 3]),
        protocol_hash=np.asarray(fingerprint(protocol)),
        correlators=expected,
    )
    payload = payload_stream.getvalue()
    (tmp_path / "development.npz").write_bytes(payload)
    snapshot = capture_source_snapshot(
        tmp_path,
        {"development_measurements": sha256(payload).hexdigest()},
        {"development_measurements": "development.npz"},
    )

    (tmp_path / "development.npz").write_bytes(b"changed after snapshot capture")
    observed = load_data(
        tmp_path,
        "development",
        protocol,
        source_snapshot=snapshot,
        source_name="development_measurements",
    )

    np.testing.assert_array_equal(observed, expected)


def test_load_data_requires_a_named_snapshot_source(tmp_path):
    snapshot = capture_source_snapshot(tmp_path, {"dev": sha256(b"x").hexdigest()}, {})
    with pytest.raises(ValueError, match="source_name is required"):
        load_data(tmp_path, "development", {}, source_snapshot=snapshot)


def test_snapshot_load_data_rejects_inconsistent_npy_size_before_numpy_load(tmp_path):
    header = BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {"descr": np.dtype("<f8").str, "fortran_order": False,
         "shape": (10**8,)},
    )
    payload_stream = BytesIO()
    with ZipFile(payload_stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("correlators.npy", header.getvalue() + b"\\x00")
    payload = payload_stream.getvalue()
    (tmp_path / "development.npz").write_bytes(payload)
    snapshot = capture_source_snapshot(
        tmp_path,
        {"development_measurements": sha256(payload).hexdigest()},
        {"development_measurements": "development.npz"},
    )

    with pytest.raises(ValueError, match="NPY header size mismatch"):
        load_data(
            tmp_path,
            "development",
            {"development": [2, 3], "powers": [0, 2], "geometry": [4, 4, 4, 4]},
            source_snapshot=snapshot,
            source_name="development_measurements",
        )


def test_development_bundle_uses_only_captured_bytes_and_excludes_other_splits(tmp_path):
    protocol = default_protocol()
    ids = np.arange(protocol["development"][0], protocol["development"][1] + 1)
    expected = np.arange(len(ids) * len(protocol["powers"]) * protocol["geometry"][3],
                         dtype=np.float64).reshape(
                             len(ids), len(protocol["powers"]), protocol["geometry"][3]
                         )
    npz_stream = BytesIO()
    np.savez_compressed(
        npz_stream, ids=ids, protocol_hash=np.asarray(fingerprint(protocol)),
        correlators=expected,
    )
    measurement_bytes = npz_stream.getvalue()
    provenance_bytes = json.dumps({"protocol": protocol}, sort_keys=True).encode()
    (tmp_path / "development.npz").write_bytes(measurement_bytes)
    (tmp_path / "development_provenance.json").write_bytes(provenance_bytes)
    # A sentinel sealed split exists on disk but is not part of this snapshot
    # and the development-only reader has no argument that can request it.
    (tmp_path / "validation.npz").write_bytes(b"sealed validation sentinel")
    snapshot = capture_source_snapshot(
        tmp_path,
        {
            "dev_npz": sha256(measurement_bytes).hexdigest(),
            "provenance": sha256(provenance_bytes).hexdigest(),
        },
        {"dev_npz": "development.npz", "provenance": "development_provenance.json"},
    )

    (tmp_path / "development.npz").write_bytes(b"replacement after snapshot capture")
    (tmp_path / "development_provenance.json").write_text("{}")
    result = bundle_for(
        source_snapshot=snapshot,
        source_names={
            "development_provenance": "provenance",
            "development_measurements": "dev_npz",
        },
    )

    assert set(result["arrays"]) == {"development"}
    np.testing.assert_array_equal(result["arrays"]["development"], expected)
    manifest = result["manifest"]
    assert manifest["source_read_mode"] == "immutable_snapshot"
    assert "core/source_integrity.py" in manifest["code"]
    assert "core/npz_safety.py" in manifest["code"]
    assert manifest["source_files"]["development_measurements"]["sha256"] == sha256(
        measurement_bytes
    ).hexdigest()
    assert manifest["source_files"]["development_provenance"]["sha256"] == sha256(
        provenance_bytes
    ).hexdigest()
    assert "validation" not in manifest["splits"]

    with pytest.raises(ValueError, match="only"):
        bundle_for(
            source_snapshot=snapshot,
            source_names={
                "development_provenance": "provenance",
                "development_measurements": "dev_npz",
                "validation_measurements": "sealed-validation",
            },
        )


def test_development_bundle_rejects_uncaptured_and_malformed_provenance(tmp_path):
    snapshot = capture_source_snapshot(tmp_path, {"dev": sha256(b"x").hexdigest()}, {})
    with pytest.raises(ValueError, match="source was not captured as valid"):
        bundle_for(
            source_snapshot=snapshot,
            source_names={
                "development_provenance": "missing",
                "development_measurements": "dev",
            },
        )

    raw = b"not-json"
    (tmp_path / "provenance.json").write_bytes(raw)
    malformed = capture_source_snapshot(
        tmp_path, {"provenance": sha256(raw).hexdigest()},
        {"provenance": "provenance.json"},
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        bundle_for(
            source_snapshot=malformed,
            source_names={
                "development_provenance": "provenance",
                "development_measurements": "missing",
            },
        )


def test_develop_uses_only_snapshot_bytes_and_labels_snapshot_submission(tmp_path, monkeypatch):
    protocol = default_protocol()
    ids = np.arange(protocol["development"][0], protocol["development"][1] + 1)
    rng = np.random.default_rng(73)
    raw = rng.normal(loc=1.0, scale=0.1,
                     size=(len(ids), len(protocol["powers"]), protocol["geometry"][3]))
    expected_blocks = experiment.blocks(
        experiment.fold(raw), protocol["block_length"],
    )
    npz_stream = BytesIO()
    np.savez_compressed(
        npz_stream, ids=ids, protocol_hash=np.asarray(fingerprint(protocol)),
        correlators=raw,
    )
    measurement_bytes = npz_stream.getvalue()
    provenance_bytes = json.dumps({"protocol": protocol}, sort_keys=True).encode()
    input_root = tmp_path / "captured-input"
    input_root.mkdir()
    (input_root / "development.npz").write_bytes(measurement_bytes)
    (input_root / "development_provenance.json").write_bytes(provenance_bytes)
    sealed_sentinel = b"validation is deliberately outside the captured snapshot"
    (input_root / "validation.npz").write_bytes(sealed_sentinel)
    snapshot = capture_source_snapshot(
        input_root,
        {
            "dev_npz": sha256(measurement_bytes).hexdigest(),
            "provenance": sha256(provenance_bytes).hexdigest(),
        },
        {"dev_npz": "development.npz", "provenance": "development_provenance.json"},
    )
    (input_root / "development.npz").write_bytes(b"changed after capture")
    (input_root / "development_provenance.json").write_text("{}")

    class FakeReferenceFit:
        def __init__(self, data, config, window):
            np.testing.assert_array_equal(data, expected_blocks)
            assert config == protocol
            self.window = list(window)

        def bootstrap(self, index):
            draws = np.linspace(0.99, 1.01, len(index))
            return ({"mass": 1.0, "se": 0.01, "p_approx": 0.5,
                     "window": self.window}, draws)

    monkeypatch.setattr(experiment, "ReferenceFit", FakeReferenceFit)
    def fake_risk(data, weights, mass, boot_masses, index, config):
        np.testing.assert_array_equal(data, expected_blocks)
        assert config == protocol
        return {"risk": 0.1, "amplitude": 1.0}

    monkeypatch.setattr(experiment, "risk", fake_risk)

    def fake_optimize_mixture(data, scales, mass, boot_masses, index, config):
        np.testing.assert_array_equal(data, expected_blocks)
        assert config == protocol
        return (
            SimpleNamespace(x=np.full(len(scales), 1.0 / len(scales)), fun=0.1,
                            nit=1),
            1, 1,
        )

    monkeypatch.setattr(
        experiment, "optimize_mixture", fake_optimize_mixture,
    )

    def fake_late_metric(data, weights, p):
        np.testing.assert_array_equal(data, expected_blocks)
        return np.eye(3)

    monkeypatch.setattr(experiment, "late_metric", fake_late_metric)

    output = tmp_path / "new-snapshot-run"
    experiment.develop(
        output, source_snapshot=snapshot,
        source_names={
            "development_provenance": "provenance",
            "development_measurements": "dev_npz",
        },
    )

    frozen = json.loads((output / "submission.json").read_text())
    assert frozen["development_source_read_mode"] == "immutable_snapshot"
    assert frozen["development_sha256"] == sha256(measurement_bytes).hexdigest()
    assert frozen["analysis_task_spec_sha256"] == fingerprint(task_spec_for(protocol))
    assert frozen["development_source_hashes"] == {
        "development_provenance": sha256(provenance_bytes).hexdigest(),
        "development_measurements": sha256(measurement_bytes).hexdigest(),
    }
    assert frozen["calibration"]["n_development_blocks"] == len(expected_blocks)
    diagnostic = json.loads((output / "development_diagnostic.json").read_text())
    assert diagnostic["schema"] == "pion-development-snapshot-diagnostic-v1"
    assert diagnostic["classification"] == "OPEN_DEVELOPMENT_DIAGNOSTIC_NOT_CONFIRMATION"
    assert diagnostic["development_source_sha256"] == sha256(measurement_bytes).hexdigest()
    assert diagnostic["development_array_sha256"] == sha256(
        np.ascontiguousarray(raw).tobytes()
    ).hexdigest()
    assert diagnostic["development_configuration_ids_sha256"] == fingerprint(
        list(map(int, ids))
    )
    assert diagnostic["source_files"]["development_measurements"]["sha256"] == sha256(
        measurement_bytes
    ).hexdigest()
    assert diagnostic["denominators"]["captured_development_source_file"] == {
        "planned": 2, "valid": 2, "failed": 0, "abstained": 0,
    }
    assert diagnostic["denominators"]["development_configuration"] == {
        "planned": len(ids), "valid": len(ids), "failed": 0, "abstained": 0,
    }
    assert diagnostic["denominators"]["development_block"] == {
        "planned": len(expected_blocks), "valid": len(expected_blocks),
        "failed": 0, "abstained": 0,
    }
    assert diagnostic["denominators"]["finite_development_risk_summary"] == {
        "planned": len(protocol["powers"]) + 1,
        "valid": len(protocol["powers"]) + 1, "failed": 0, "abstained": 0,
    }
    for name, record in diagnostic["output_files"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == record["sha256"]
    assert "same captured correlators" in (output / "README.md").read_text()
    assert "physical_score`: `null`" in (output / "README.md").read_text()
    verification = experiment.verify_snapshot_development_artifacts(
        output, source_snapshot=snapshot,
        source_names={
            "development_provenance": "provenance",
            "development_measurements": "dev_npz",
        },
    )
    assert verification["status"] == "SNAPSHOT_ARTIFACTS_VERIFIED_NOT_CERTIFIED"
    assert verification["input_source_denominator"] == {
        "planned": 2, "valid": 2, "failed": 0, "abstained": 0,
    }
    original_readme = (output / "README.md").read_bytes()
    (output / "README.md").write_bytes(original_readme + b"tamper")
    with pytest.raises(ValueError, match="output hash mismatch: README.md"):
        experiment.verify_snapshot_development_artifacts(
            output, source_snapshot=snapshot,
            source_names={
                "development_provenance": "provenance",
                "development_measurements": "dev_npz",
            },
        )
    (output / "README.md").write_bytes(original_readme)
    assert not (output / "validation.npz").exists()
    assert (input_root / "validation.npz").read_bytes() == sealed_sentinel
    from tasks.pion_projection.candidate import run_candidate

    frozen["analysis_task_spec_sha256"] = "0" * 64
    (output / "submission.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="analysis task/source dependency contract changed"):
        run_candidate(output, "unused-definition.json", split="development")

    frozen["analysis_task_spec_sha256"] = fingerprint(task_spec_for(protocol))
    (output / "submission.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="Snapshot-only development"):
        run_candidate(output, "unused-definition.json", split="development")
    with pytest.raises(ValueError, match="Snapshot-only development"):
        experiment.evaluate(output)


def test_snapshot_loader_refuses_validation_and_gap_splits(tmp_path):
    payload = b"not opened because snapshot mode is development only"
    (tmp_path / "validation.npz").write_bytes(payload)
    snapshot = capture_source_snapshot(
        tmp_path, {"validation": sha256(payload).hexdigest()},
        {"validation": "validation.npz"},
    )
    for split in ("validation", "gap"):
        with pytest.raises(ValueError, match="restricted to the open development split"):
            load_data(
                tmp_path, split, {}, source_snapshot=snapshot,
                source_name="validation",
            )


def test_evaluate_rejects_stale_snapshot_reader_contract_before_loading_splits(tmp_path):
    protocol = default_protocol()
    development_bytes = b"not parsed: dependency mismatch must stop first"
    (tmp_path / "development.npz").write_bytes(development_bytes)
    (tmp_path / "submission.json").write_text(json.dumps({
        "protocol_hash": fingerprint(protocol),
        "development_sha256": sha256(development_bytes).hexdigest(),
        "analysis_task_spec_sha256": "0" * 64,
    }))

    with pytest.raises(ValueError, match="analysis task/source dependency contract changed"):
        experiment.evaluate(tmp_path)

    assert not (tmp_path / "validation.npz").exists()

