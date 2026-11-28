from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest

from core.io import fingerprint
from core.source_integrity import capture_source_snapshot
from tasks.wilson_loop import c_lane_open_development_diagnostic as diagnostic


def _synthetic_transaction():
    generation = "d" * 64
    loops = np.linspace(0.15, 0.45, 100 * 4 * 5, dtype=np.float64).reshape(100, 1, 4, 5)
    source_hashes = {name: (str(i + 1) * 64) for i, name in enumerate((
        "measure.py", "kernels.py", "stout.py", "statistics.py", "common.py", "identity.py",
    ))}
    protocol = {
        "development": [0, 99],
        "geometry": [16, 16, 16, 16],
        "beta": 6.0,
        "boundary_conditions": "periodic_all_directions",
        "r_values": [1, 2, 3, 4],
        "t_values": [4, 5, 6, 7, 8],
        "source_hashes": source_hashes,
        "kernels": [{"name": "stout_rho0.1_n5", "kind": "stout", "rho": 0.1, "steps": 5}],
    }
    protocol_hash = fingerprint(protocol)
    npz_stream = BytesIO()
    np.savez_compressed(
        npz_stream,
        ids=np.arange(100, dtype=np.int64),
        loops=loops,
        generation=np.asarray(generation),
        protocol_hash=np.asarray(protocol_hash),
    )
    npz_bytes = npz_stream.getvalue()
    dependencies = {name: digest for name, digest in source_hashes.items() if name != "stout.py"}
    provenance_bytes = json.dumps({
        "generation": generation,
        "protocol": protocol,
        "dependencies": dependencies,
        "records": [{"id": i, "hmc_trajectory": 10 * i} for i in range(100)],
    }, sort_keys=True, separators=(",", ":")).encode()
    npz_sha = sha256(npz_bytes).hexdigest()
    provenance_sha = sha256(provenance_bytes).hexdigest()
    loops_sha = sha256(np.ascontiguousarray(loops).tobytes()).hexdigest()
    commit_bytes = json.dumps({
        "split": "development",
        "generation": generation,
        "npz_sha256": npz_sha,
        "provenance_sha256": provenance_sha,
        "loops_sha256": loops_sha,
    }, sort_keys=True, separators=(",", ":")).encode()
    return generation, protocol_hash, loops, npz_bytes, provenance_bytes, commit_bytes


def _pin_synthetic_transaction(monkeypatch, transaction):
    generation, _, loops, npz_bytes, provenance_bytes, _ = transaction
    monkeypatch.setattr(diagnostic, "GENERATION", generation)
    monkeypatch.setattr(diagnostic, "EXPECTED_NPZ_SHA256", sha256(npz_bytes).hexdigest())
    monkeypatch.setattr(diagnostic, "EXPECTED_PROVENANCE_SHA256", sha256(provenance_bytes).hexdigest())
    monkeypatch.setattr(diagnostic, "EXPECTED_LOOPS_SHA256", sha256(
        np.ascontiguousarray(loops).tobytes()
    ).hexdigest())
    monkeypatch.setattr(diagnostic, "EXPECTED_SOURCE_SHA256", {
        name: (str(i + 1) * 64) for i, name in enumerate((
            "measure.py", "kernels.py", "stout.py", "statistics.py", "common.py", "identity.py",
        ))
    })


def test_wilson_development_reader_uses_the_captured_transaction_bytes(tmp_path, monkeypatch):
    transaction = _synthetic_transaction()
    _pin_synthetic_transaction(monkeypatch, transaction)
    _, protocol_hash, expected_loops, npz_bytes, provenance_bytes, commit_bytes = transaction
    contents = {
        "development_npz": ("development.npz", npz_bytes),
        "development_provenance": ("development_provenance.json", provenance_bytes),
        "development_commit": ("development.commit.json", commit_bytes),
    }
    for filename, payload in contents.values():
        (tmp_path / filename).write_bytes(payload)
    snapshot = capture_source_snapshot(
        tmp_path,
        {role: sha256(payload).hexdigest() for role, (_, payload) in contents.items()},
        {role: filename for role, (filename, _) in contents.items()},
    )

    for filename, _ in contents.values():
        (tmp_path / filename).write_bytes(b"changed after capture")
    ids, observed, observed_protocol_hash, trajectories, hashes = diagnostic.load_transaction(
        source_snapshot=snapshot,
        source_names={role: role for role in contents},
    )

    np.testing.assert_array_equal(ids, np.arange(100))
    np.testing.assert_array_equal(observed, expected_loops[:, 0])
    assert observed_protocol_hash == protocol_hash
    assert trajectories == [10 * i for i in range(100)]
    assert hashes["read_mode"] == "immutable_snapshot"
    assert hashes["development_npz_sha256"] == sha256(npz_bytes).hexdigest()
    assert hashes["development_provenance_sha256"] == sha256(provenance_bytes).hexdigest()
    assert hashes["development_commit_sha256"] == sha256(commit_bytes).hexdigest()


def test_wilson_development_reader_rejects_extra_split_or_missing_snapshot_entry(tmp_path, monkeypatch):
    transaction = _synthetic_transaction()
    _pin_synthetic_transaction(monkeypatch, transaction)
    _, _, _, npz_bytes, provenance_bytes, commit_bytes = transaction
    contents = {
        "development_npz": ("development.npz", npz_bytes),
        "development_provenance": ("development_provenance.json", provenance_bytes),
        "development_commit": ("development.commit.json", commit_bytes),
    }
    for filename, payload in contents.values():
        (tmp_path / filename).write_bytes(payload)
    expected = {role: sha256(payload).hexdigest() for role, (_, payload) in contents.items()}
    paths = {role: filename for role, (filename, _) in contents.items()}
    snapshot = capture_source_snapshot(tmp_path, expected, paths)
    with pytest.raises(ValueError, match="exactly the three development"):
        diagnostic.load_transaction(
            source_snapshot=snapshot,
            source_names={**{role: role for role in contents}, "validation_npz": "validation"},
        )

    incomplete = capture_source_snapshot(
        tmp_path,
        expected,
        {"development_npz": "development.npz", "development_provenance": "development_provenance.json"},
    )
    with pytest.raises(ValueError, match="incomplete captured development transaction"):
        diagnostic.load_transaction(
            source_snapshot=incomplete,
            source_names={role: role for role in contents},
        )


def test_wilson_diagnostic_run_reuses_snapshot_and_writes_text_and_figure(tmp_path, monkeypatch):
    transaction = _synthetic_transaction()
    _pin_synthetic_transaction(monkeypatch, transaction)
    monkeypatch.setattr(diagnostic, "N_DRAWS", 24)
    source_copy = tmp_path / "reader_source.py"
    source_payloads = {
        "snapshot_c_lane_open_development_diagnostic.py": b"# isolated synthetic-test source identity\n",
        "snapshot_core_npz_safety.py": b"# isolated npz helper identity\n",
        "snapshot_core_source_integrity.py": b"# isolated snapshot helper identity\n",
    }
    source_copy.write_bytes(
        source_payloads["snapshot_c_lane_open_development_diagnostic.py"]
    )
    monkeypatch.setattr(diagnostic, "__file__", str(source_copy))
    npz_safety_source = tmp_path / "npz_safety_source.py"
    npz_safety_source.write_bytes(source_payloads["snapshot_core_npz_safety.py"])
    monkeypatch.setattr(
        diagnostic.npz_safety_module, "__file__", str(npz_safety_source)
    )
    source_integrity_source = tmp_path / "source_integrity_source.py"
    source_integrity_source.write_bytes(source_payloads["snapshot_core_source_integrity.py"])
    monkeypatch.setattr(
        diagnostic.source_integrity_module, "__file__", str(source_integrity_source)
    )
    _, _, _, npz_bytes, provenance_bytes, commit_bytes = transaction
    payloads = {
        "development_npz": ("development.npz", npz_bytes),
        "development_provenance": ("development_provenance.json", provenance_bytes),
        "development_commit": ("development.commit.json", commit_bytes),
    }
    for filename, payload in payloads.values():
        (tmp_path / filename).write_bytes(payload)
    snapshot = capture_source_snapshot(
        tmp_path,
        {role: sha256(payload).hexdigest() for role, (_, payload) in payloads.items()},
        {role: filename for role, (filename, _) in payloads.items()},
    )
    for filename, _ in payloads.values():
        (tmp_path / filename).write_bytes(b"replacement after capture")

    code_paths = {
        source_copy.resolve(), npz_safety_source.resolve(), source_integrity_source.resolve()
    }
    mutated_code_paths = set()
    real_read_bytes = Path.read_bytes

    def mutate_code_after_capture(path):
        captured = real_read_bytes(path)
        resolved = path.resolve()
        if resolved in code_paths and resolved not in mutated_code_paths:
            path.write_bytes(b"source changed after capture")
            mutated_code_paths.add(resolved)
        return captured

    monkeypatch.setattr(Path, "read_bytes", mutate_code_after_capture)
    load_transaction = diagnostic.load_transaction

    def require_code_capture_before_input(*args, **kwargs):
        assert mutated_code_paths == code_paths
        return load_transaction(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "load_transaction", require_code_capture_before_input)

    output = tmp_path / "diagnostic"
    diagnostic.run(
        None, output, source_snapshot=snapshot,
        source_names={role: role for role in payloads},
    )
    protocol = json.loads((output / "protocol.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert protocol["input_hashes"]["read_mode"] == "immutable_snapshot"
    assert summary["status"] == "RETROSPECTIVE_DEVELOPMENT_ONLY_NOT_CONFIRMATION"
    assert summary["physical_score"] is None and summary["certified"] is False
    assert summary["invalid_draws"]["planned"] == 3 * 24
    assert protocol["analysis_source_sha256"]["core/npz_safety.py"] == sha256(
        (output / "snapshot_core_npz_safety.py").read_bytes()
    ).hexdigest()
    assert protocol["analysis_source_sha256"]["core/source_integrity.py"] == sha256(
        (output / "snapshot_core_source_integrity.py").read_bytes()
    ).hexdigest()
    assert protocol["source_script_sha256"] == sha256(
        (output / "snapshot_c_lane_open_development_diagnostic.py").read_bytes()
    ).hexdigest()
    assert mutated_code_paths == code_paths
    for filename, payload in source_payloads.items():
        assert (output / filename).read_bytes() == payload
    assert summary["analysis_source_sha256"] == protocol["analysis_source_sha256"]
    assert (output / "c_lane_bootstrap_intervals.png").is_file()
    assert (output / "README.md").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert "snapshot_c_lane_open_development_diagnostic.py" in manifest["files"]
    assert "snapshot_core_npz_safety.py" in manifest["files"]
    assert "snapshot_core_source_integrity.py" in manifest["files"]
    for name, expected in manifest["files"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.run(
            None, output, source_snapshot=snapshot,
            source_names={role: role for role in payloads},
        )
