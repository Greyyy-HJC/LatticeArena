from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tasks.optimize_gaussian.d11_operator_contract_audit import audit, build


def test_d11_audit_binds_and_retains_shared_numeric_validation(tmp_path):
    result = audit()
    output = tmp_path / "d11-audit"
    build(output)

    assert result["denominator"] == {
        "planned": 9, "valid": 7, "failed": 0, "abstained": 2,
        "unit": "one source/algebra contract obligation",
    }
    source_path = Path(__file__).resolve().parents[1] / "core/numerics.py"
    expected_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert result["source_hashes"]["core/numerics.py"] == expected_hash

    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["source_hashes"]["core/numerics.py"] == expected_hash
    assert hashlib.sha256(
        (output / "snapshot_core_numerics.py").read_bytes()
    ).hexdigest() == expected_hash

    artifacts = json.loads((output / "artifact_hashes.json").read_text())
    assert "snapshot_core_numerics.py" in artifacts
    for name, digest in artifacts.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
