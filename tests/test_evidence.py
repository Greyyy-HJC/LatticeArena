import copy
import hashlib
import json

from core.evidence import (
    EVIDENCE_SCHEMA,
    apply_evidence_gate,
    apply_evidence_bundle_gate,
    apply_evidence_gate_with_source_bytes,
    apply_evidence_bundle_gate_with_source_bytes,
    evidence_envelope,
    inspect_distinct_chain_identities,
    inspect_evidence_envelope,
)
from core.receipts import denominator, run_receipt, scenario_spec
from core.source_integrity import capture_source_snapshot
from core.state import RunState


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def make_envelope(*, task="pion_projection", kind="score_dependence",
                  chain_id="score-chain", split_id="development",
                  manifest_hash=HASH_A, unit="bounded score", review="reviewed",
                  reviewer_role_claim="independent"):
    review_record = (
        {"status": "pending"} if review == "pending" else
        {
            "status": "reviewed",
            "receipt_sha256": HASH_B,
            "scope_sha256": HASH_C,
            "reviewer_role_claim": reviewer_role_claim,
            "independence_claim_sha256": HASH_A,
        }
    )
    return evidence_envelope(
        task,
        kind,
        chain={
            "chain_id": chain_id,
            "split_id": split_id,
            "state_unit": "saved paired macroblock",
            "manifest_sha256": manifest_hash,
        },
        observable={
            "map_id": "fixture-score-v1",
            "map_sha256": HASH_B,
            "support": {"lower": 0.0, "upper": 1.0, "unit": unit},
        },
        method={
            "name": "corrected Markov pseudo-gap Bernstein bound",
            "version": "fixture-v1",
            "assumptions": ["the fixture chain is homogeneous"],
        },
        dependence={
            "mode": "markov",
            "assumptions": ["gap applies to the exact saved-state chain"],
            "gamma_ps_lower": 0.25,
            "tv_bound_after_burn_in": 0.01,
            "burn_in": {"unit": "saved macroblock", "count": 8},
        },
        confidence={
            "family_id": "fixture-family",
            "alpha_total": 0.05,
            "alpha_allocated": 0.01,
            "family_size": 5,
        },
        source_hashes={"manifest": manifest_hash, "score_map": HASH_B},
        review=review_record,
    )


def make_run_receipt(*, planned=2, valid=2):
    scenario = scenario_spec("evidence-gate-fixture", "panel", "metadata gate fixture")
    counts = denominator("panels", planned=planned, valid=valid)
    reasons = []
    if counts["not_run"]:
        reasons.append({"unit": "panels", "state": "not_run",
                        "count": counts["not_run"], "reason": "fixture not started"})
    return run_receipt(
        scenario, {"panels": counts}, artifact_hashes={"fixture": HASH_C},
        text_flow="fixture -> count planned/valid/not-run panels",
        reason_rows=reasons,
    )


def test_envelope_is_typed_and_never_certifies_physics():
    envelope = make_envelope()
    assert envelope["schema"] == EVIDENCE_SCHEMA
    result = inspect_evidence_envelope(
        envelope,
        expected={
            "task": "pion_projection",
            "evidence_kind": "score_dependence",
            "observable.support.unit": "bounded score",
            "dependence.gamma_ps_lower": 0.25,
        },
        required_source_hashes={"manifest": HASH_A},
    )
    assert result["status"] == "SCHEMA_COMPLETE_NOT_CERTIFIED"
    assert result["certified"] is False
    assert result["physical_score"] is None


def test_task_fixtures_preserve_distinct_observable_units():
    pion_anchor = make_envelope(
        task="pion_projection", kind="anchor_dependence", chain_id="E-anchor",
        split_id="anchor", manifest_hash=HASH_A, unit="dimensionless anchor scalar",
    )
    optimize_score = make_envelope(
        task="optimize_gaussian", kind="paired_block_risk",
        chain_id="D-paired-blocks", split_id="development", manifest_hash=HASH_B,
        unit="bounded loss in [0,1]",
    )
    wilson_force = make_envelope(
        task="wilson_loop", kind="thermal_force_bound", chain_id="D-Wilson",
        split_id="development", manifest_hash=HASH_C, unit="a_s*a_t*F",
    )
    for envelope, task, unit in (
        (pion_anchor, "pion_projection", "dimensionless anchor scalar"),
        (optimize_score, "optimize_gaussian", "bounded loss in [0,1]"),
        (wilson_force, "wilson_loop", "a_s*a_t*F"),
    ):
        result = inspect_evidence_envelope(
            envelope,
            expected={"task": task, "observable.support.unit": unit},
        )
        assert result["status"] == "SCHEMA_COMPLETE_NOT_CERTIFIED"


def test_missing_markov_receipt_fields_abstain():
    envelope = make_envelope()
    del envelope["dependence"]["gamma_ps_lower"]
    result = inspect_evidence_envelope(envelope)
    assert result["status"] == "ABSTAIN_INVALID_EVIDENCE_ENVELOPE"
    assert "gamma_ps_lower" in result["reasons"][0]


def test_wrong_chain_and_force_unit_abstain():
    envelope = make_envelope(task="wilson_loop", kind="thermal_force_bound")
    wrong_chain = inspect_evidence_envelope(
        envelope, expected={"chain.chain_id": "authorized-F-chain"},
    )
    wrong_unit = inspect_evidence_envelope(
        envelope, expected={"observable.support.unit": "a_s*a_t*F"},
    )
    assert wrong_chain["status"] == "ABSTAIN_EVIDENCE_MISMATCH"
    assert wrong_unit["status"] == "ABSTAIN_EVIDENCE_MISMATCH"


def test_stale_source_hash_and_pending_review_abstain():
    envelope = make_envelope(review="pending")
    result = inspect_evidence_envelope(
        envelope, required_source_hashes={"manifest": HASH_C},
    )
    assert result["status"] == "ABSTAIN_EVIDENCE_MISMATCH"
    assert any("source hash mismatch" in reason for reason in result["reasons"])
    assert any("review receipt is missing" in reason for reason in result["reasons"])


def test_distinct_chain_identity_is_not_treated_as_independence_proof():
    anchor = make_envelope(kind="anchor", chain_id="E-anchor", manifest_hash=HASH_A)
    score = make_envelope(kind="score", chain_id="D-score", manifest_hash=HASH_C)
    result = inspect_distinct_chain_identities(anchor, score)
    assert result["status"] == "IDENTITIES_DISTINCT_NOT_INDEPENDENCE_PROOF"
    assert result["identity_distinct"] is True
    assert result["independence_proven"] is False


def test_reused_chain_identity_abstains():
    anchor = make_envelope(kind="anchor", chain_id="same", manifest_hash=HASH_A)
    score = make_envelope(kind="score", chain_id="same", manifest_hash=HASH_B)
    result = inspect_distinct_chain_identities(anchor, score)
    assert result["status"] == "ABSTAIN_CHAIN_IDENTITY_COLLISION"
    assert "chain_id collision" in result["reasons"]
    assert result["independence_proven"] is False


def test_family_confidence_and_review_hashes_are_validated():
    import pytest

    envelope = make_envelope()
    malformed = copy.deepcopy(envelope)
    malformed["confidence"]["alpha_allocated"] = 0.06
    assert inspect_evidence_envelope(malformed)["status"] == "ABSTAIN_INVALID_EVIDENCE_ENVELOPE"

    malformed = copy.deepcopy(envelope)
    malformed["review"]["receipt_sha256"] = "stale"
    assert inspect_evidence_envelope(malformed)["status"] == "ABSTAIN_INVALID_EVIDENCE_ENVELOPE"

    malformed = copy.deepcopy(envelope)
    malformed["review"]["independence_claim_sha256"] = "stale"
    assert inspect_evidence_envelope(malformed)["status"] == "ABSTAIN_INVALID_EVIDENCE_ENVELOPE"

    with pytest.raises(ValueError, match="support.upper must exceed lower"):
        evidence_envelope(
            "fixture", "bad-support",
            chain={"chain_id": "c", "split_id": "s", "state_unit": "state",
                   "manifest_sha256": HASH_A},
            observable={"map_id": "m", "map_sha256": HASH_B,
                        "support": {"lower": 1, "upper": 1, "unit": "u"}},
            method={"name": "m", "version": "v", "assumptions": ["a"]},
            dependence={"mode": "deterministic", "assumptions": ["exact"]},
            confidence={"family_id": "f", "alpha_total": 0.05,
                        "alpha_allocated": 0.01, "family_size": 1},
            source_hashes={"source": HASH_A},
        )


def test_review_requires_an_explicit_independence_claim_but_does_not_authenticate_it():
    no_independence_claim = make_envelope()
    del no_independence_claim["review"]["independence_claim_sha256"]
    invalid = inspect_evidence_envelope(no_independence_claim)
    assert invalid["status"] == "ABSTAIN_INVALID_EVIDENCE_ENVELOPE"
    assert "independence_claim_sha256" in invalid["reasons"][0]

    non_independent = make_envelope(reviewer_role_claim="author")
    result = inspect_evidence_envelope(non_independent)
    assert result["status"] == "ABSTAIN_EVIDENCE_MISMATCH"
    assert "reviewer role claim is not independent" in result["reasons"]
    assert result["certified"] is False
    assert result["physical_score"] is None

    state = RunState()
    gated = apply_evidence_gate(
        state, "statistical", non_independent, make_run_receipt(),
    )
    assert gated["status"] == "ABSTAIN_EVIDENCE_GATE"
    assert gated["run_state"]["status"]["statistical"] == "abstained"


def test_dependence_assumptions_must_be_a_sequence_not_a_string():
    envelope = make_envelope()
    envelope["dependence"]["assumptions"] = "assume independent draws"
    result = inspect_evidence_envelope(envelope)
    assert result["status"] == "ABSTAIN_INVALID_EVIDENCE_ENVELOPE"
    assert "sequence of nonempty strings" in result["reasons"][0]


def test_evidence_bundle_checks_anchor_and_score_together_without_claiming_independence():
    anchor = make_envelope(kind="anchor_coverage", chain_id="E-anchor",
                           split_id="anchor", manifest_hash=HASH_A,
                           unit="dimensionless anchor scalar")
    score = make_envelope(kind="score_dependence", chain_id="D-score",
                          split_id="development", manifest_hash=HASH_C,
                          unit="bounded score")
    state = RunState()
    result = apply_evidence_bundle_gate(
        state,
        "statistical",
        {"anchor": anchor, "score": score},
        make_run_receipt(),
        expected_by_envelope={
            "anchor": {"task": "pion_projection", "evidence_kind": "anchor_coverage"},
            "score": {"task": "pion_projection", "evidence_kind": "score_dependence"},
        },
        required_source_hashes_by_envelope={
            "anchor": {"manifest": HASH_A},
            "score": {"manifest": HASH_C},
        },
        distinct_chain_pairs=[("anchor", "score")],
    )
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["run_state"]["status"]["statistical"] == "draft"
    assert result["chain_identity_checks"][0]["identity_distinct"] is True
    assert result["chain_identity_checks"][0]["independence_proven"] is False
    assert result["certified"] is False
    assert result["physical_score"] is None


def test_evidence_bundle_abstains_on_a_chain_identity_collision_or_bad_member():
    anchor = make_envelope(kind="anchor_coverage", chain_id="reused",
                           split_id="anchor", manifest_hash=HASH_A,
                           unit="dimensionless anchor scalar")
    score = make_envelope(kind="score_dependence", chain_id="reused",
                          split_id="development", manifest_hash=HASH_C)
    expected = {
        "anchor": {"task": "pion_projection", "evidence_kind": "anchor_coverage"},
        "score": {"task": "pion_projection", "evidence_kind": "score_dependence"},
    }
    state = RunState()
    collision = apply_evidence_bundle_gate(
        state, "statistical", {"anchor": anchor, "score": score},
        make_run_receipt(), expected_by_envelope=expected,
        distinct_chain_pairs=[("anchor", "score")],
    )
    assert collision["status"] == "ABSTAIN_EVIDENCE_BUNDLE"
    assert collision["run_state"]["status"]["statistical"] == "abstained"
    assert collision["chain_identity_checks"][0]["independence_proven"] is False

    state = RunState()
    score["observable"]["support"]["unit"] = "wrong unit"
    mismatch = apply_evidence_bundle_gate(
        state, "statistical", {"anchor": anchor, "score": score},
        make_run_receipt(), expected_by_envelope={
            "anchor": expected["anchor"],
            "score": {**expected["score"], "observable.support.unit": "bounded score"},
        },
    )
    assert mismatch["status"] == "ABSTAIN_EVIDENCE_BUNDLE"
    assert mismatch["run_state"]["status"]["statistical"] == "abstained"


def test_evidence_bundle_does_not_treat_null_required_hashes_as_optional():
    anchor = make_envelope(kind="anchor_coverage", chain_id="E-anchor",
                           split_id="anchor", manifest_hash=HASH_A,
                           unit="dimensionless anchor scalar")
    result = apply_evidence_bundle_gate(
        RunState(),
        "statistical",
        {"anchor": anchor},
        make_run_receipt(),
        expected_by_envelope={
            "anchor": {"task": "pion_projection", "evidence_kind": "anchor_coverage"},
        },
        required_source_hashes_by_envelope={"anchor": None},
    )
    assert result["status"] == "ABSTAIN_EVIDENCE_BUNDLE"
    assert result["run_state"]["status"]["statistical"] == "abstained"


def test_runstate_gate_only_advances_to_draft_when_envelope_and_denominator_are_complete():
    state = RunState()
    result = apply_evidence_gate(state, "statistical", make_envelope(), make_run_receipt())
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["run_state"]["status"]["statistical"] == "draft"
    assert result["run_state"]["score"] is None
    assert result["certified"] is False


def test_runstate_gate_abstains_for_incomplete_denominator_or_missing_envelope_field():
    state = RunState()
    partial = apply_evidence_gate(
        state, "statistical", make_envelope(), make_run_receipt(planned=2, valid=1),
    )
    assert partial["status"] == "ABSTAIN_EVIDENCE_GATE"
    assert partial["run_state"]["status"]["statistical"] == "abstained"
    assert any("not run" in reason for reason in partial["reasons"])

    state = RunState()
    broken = make_envelope()
    del broken["dependence"]["tv_bound_after_burn_in"]
    result = apply_evidence_gate(state, "statistical", broken, make_run_receipt())
    assert result["status"] == "ABSTAIN_EVIDENCE_GATE"
    assert result["run_state"]["score"] is None


def test_runstate_gate_rejects_empty_or_missing_run_receipt():
    state = RunState()
    empty = apply_evidence_gate(
        state, "statistical", make_envelope(), make_run_receipt(planned=0, valid=0),
    )
    assert empty["status"] == "ABSTAIN_EVIDENCE_GATE"
    assert any("no planned units" in reason for reason in empty["reasons"])

    state = RunState()
    missing = apply_evidence_gate(state, "statistical", make_envelope(), None)
    assert missing["status"] == "ABSTAIN_EVIDENCE_GATE"


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def test_source_byte_gate_verifies_files_before_advancing_to_draft(tmp_path):
    payload = b"synthetic manifest"
    (tmp_path / "manifest.json").write_bytes(payload)
    digest = _sha256_bytes(payload)
    envelope = make_envelope()
    envelope["source_hashes"]["manifest"] = digest
    state = RunState()

    result = apply_evidence_gate_with_source_bytes(
        state,
        "statistical",
        envelope,
        make_run_receipt(),
        source_root=tmp_path,
        source_paths={"manifest": "manifest.json"},
        required_source_hashes={"manifest": digest},
    )

    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_verification"]["status"] == "verified"
    assert result["source_verification"]["denominator"]["valid"] == 1
    assert result["source_byte_identity_verified"] is True
    assert result["certified"] is False
    assert result["physical_score"] is None


def test_source_snapshot_gate_uses_the_same_bytes_as_the_reader(tmp_path):
    payload = b'{"calibration_value": 0.25}'
    (tmp_path / "manifest.json").write_bytes(payload)
    digest = _sha256_bytes(payload)
    snapshot = capture_source_snapshot(
        tmp_path, {"manifest": digest}, {"manifest": "manifest.json"},
    )
    # This is the byte sequence the task reader parses before producing its
    # summary. Replacing the path afterward must not alter gate verification.
    task_reader_input = snapshot.read_bytes("manifest")
    task_reader_value = json.loads(task_reader_input)["calibration_value"]
    (tmp_path / "manifest.json").write_bytes(b"changed after task read")
    envelope = make_envelope()
    envelope["source_hashes"]["manifest"] = digest

    result = apply_evidence_gate_with_source_bytes(
        RunState(), "statistical", envelope, make_run_receipt(),
        source_snapshot=snapshot,
        required_source_hashes={"manifest": digest},
    )

    assert task_reader_input == payload
    assert task_reader_value == 0.25
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_verification"]["denominator"]["valid"] == 1
    assert result["source_byte_identity_verified"] is True


def test_source_byte_gate_abstains_on_hash_mismatch(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b"observed")
    expected = _sha256_bytes(b"expected")
    envelope = make_envelope()
    envelope["source_hashes"]["manifest"] = expected
    state = RunState()

    result = apply_evidence_gate_with_source_bytes(
        state,
        "statistical",
        envelope,
        make_run_receipt(),
        source_root=tmp_path,
        source_paths={"manifest": "manifest.json"},
        required_source_hashes={"manifest": expected},
    )

    assert result["status"] == "ABSTAIN_SOURCE_BYTES"
    assert result["source_verification"]["denominator"]["failed"] == 1
    assert result["run_state"]["status"]["statistical"] == "abstained"
    assert result["physical_score"] is None


def test_bundle_source_byte_gate_is_atomic_across_all_routes(tmp_path):
    anchor_bytes = b"synthetic anchor manifest"
    score_bytes = b"synthetic score manifest"
    (tmp_path / "anchor.json").write_bytes(anchor_bytes)
    (tmp_path / "score.json").write_bytes(score_bytes)
    anchor_hash = _sha256_bytes(anchor_bytes)
    score_hash = _sha256_bytes(score_bytes)
    anchor = make_envelope(kind="anchor_coverage", chain_id="E-anchor",
                           split_id="anchor", manifest_hash=anchor_hash,
                           unit="dimensionless anchor scalar")
    score = make_envelope(kind="score_dependence", chain_id="D-score",
                          split_id="development", manifest_hash=score_hash,
                          unit="bounded score")
    anchor["source_hashes"]["manifest"] = anchor_hash
    score["source_hashes"]["manifest"] = score_hash
    expected = {
        "anchor": {"task": "pion_projection", "evidence_kind": "anchor_coverage"},
        "score": {"task": "pion_projection", "evidence_kind": "score_dependence"},
    }
    required = {
        "anchor": {"manifest": anchor_hash},
        "score": {"manifest": score_hash},
    }
    state = RunState()

    result = apply_evidence_bundle_gate_with_source_bytes(
        state,
        "statistical",
        {"anchor": anchor, "score": score},
        make_run_receipt(),
        source_root=tmp_path,
        source_paths_by_envelope={
            "anchor": {"manifest": "anchor.json"},
            "score": {"manifest": "score.json"},
        },
        expected_by_envelope=expected,
        required_source_hashes_by_envelope=required,
        distinct_chain_pairs=[("anchor", "score")],
    )

    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_verification"]["denominator"]["planned"] == 2
    assert result["source_verification"]["denominator"]["valid"] == 2
    assert result["source_byte_identity_verified"] is True
    assert result["chain_identity_checks"][0]["independence_proven"] is False


def test_bundle_source_snapshots_are_verified_atomically_after_path_changes(tmp_path):
    anchor_bytes = b"anchor bytes consumed by reader"
    score_bytes = b"score bytes consumed by reader"
    (tmp_path / "anchor.json").write_bytes(anchor_bytes)
    (tmp_path / "score.json").write_bytes(score_bytes)
    anchor_hash = _sha256_bytes(anchor_bytes)
    score_hash = _sha256_bytes(score_bytes)
    anchor = make_envelope(kind="anchor_coverage", chain_id="E-anchor",
                           split_id="anchor", manifest_hash=anchor_hash,
                           unit="dimensionless anchor scalar")
    score = make_envelope(kind="score_dependence", chain_id="D-score",
                          split_id="development", manifest_hash=score_hash,
                          unit="bounded score")
    anchor["source_hashes"]["manifest"] = anchor_hash
    score["source_hashes"]["manifest"] = score_hash
    snapshots = {
        "anchor": capture_source_snapshot(
            tmp_path, {"manifest": anchor_hash}, {"manifest": "anchor.json"},
        ),
        "score": capture_source_snapshot(
            tmp_path, {"manifest": score_hash}, {"manifest": "score.json"},
        ),
    }
    assert snapshots["anchor"].read_bytes("manifest") == anchor_bytes
    assert snapshots["score"].read_bytes("manifest") == score_bytes
    (tmp_path / "anchor.json").write_bytes(b"changed after reader")
    (tmp_path / "score.json").write_bytes(b"changed after reader")

    result = apply_evidence_bundle_gate_with_source_bytes(
        RunState(), "statistical", {"anchor": anchor, "score": score},
        make_run_receipt(),
        source_snapshots_by_envelope=snapshots,
        expected_by_envelope={
            "anchor": {"task": "pion_projection", "evidence_kind": "anchor_coverage"},
            "score": {"task": "pion_projection", "evidence_kind": "score_dependence"},
        },
        required_source_hashes_by_envelope={
            "anchor": {"manifest": anchor_hash},
            "score": {"manifest": score_hash},
        },
        distinct_chain_pairs=[("anchor", "score")],
    )

    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_verification"]["denominator"]["valid"] == 2


def test_bundle_source_byte_gate_abstains_if_any_route_is_missing_or_mismatched(tmp_path):
    anchor_bytes = b"synthetic anchor manifest"
    (tmp_path / "anchor.json").write_bytes(anchor_bytes)
    anchor_hash = _sha256_bytes(anchor_bytes)
    score_hash = _sha256_bytes(b"expected score manifest")
    anchor = make_envelope(kind="anchor_coverage", chain_id="E-anchor",
                           split_id="anchor", manifest_hash=anchor_hash,
                           unit="dimensionless anchor scalar")
    score = make_envelope(kind="score_dependence", chain_id="D-score",
                          split_id="development", manifest_hash=score_hash,
                          unit="bounded score")
    anchor["source_hashes"]["manifest"] = anchor_hash
    score["source_hashes"]["manifest"] = score_hash
    expected = {
        "anchor": {"task": "pion_projection", "evidence_kind": "anchor_coverage"},
        "score": {"task": "pion_projection", "evidence_kind": "score_dependence"},
    }
    required = {
        "anchor": {"manifest": anchor_hash},
        "score": {"manifest": score_hash},
    }
    state = RunState()

    result = apply_evidence_bundle_gate_with_source_bytes(
        state,
        "statistical",
        {"anchor": anchor, "score": score},
        make_run_receipt(),
        source_root=tmp_path,
        source_paths_by_envelope={"anchor": {"manifest": "anchor.json"}},
        expected_by_envelope=expected,
        required_source_hashes_by_envelope=required,
        distinct_chain_pairs=[("anchor", "score")],
    )

    assert result["status"] == "ABSTAIN_SOURCE_BYTES"
    assert result["source_verification"]["denominator"]["planned"] == 2
    assert result["source_verification"]["denominator"]["valid"] == 1
    assert result["source_verification"]["denominator"]["abstained"] == 1
    assert result["run_state"]["status"]["statistical"] == "abstained"
