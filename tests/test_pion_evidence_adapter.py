import copy
import hashlib
import json

from core.evidence import evidence_envelope
from core.receipts import denominator, run_receipt, scenario_spec
from core.source_integrity import capture_source_snapshot
from core.state import RunState
from tasks.pion_projection.evidence_adapter import (
    apply_pion_statistical_evidence_gate,
    score_map_sha256,
)
from tasks.pion_projection.bounded_block_risk_contract import selected_protocol


H = [f"{i:064x}" for i in range(1, 30)]
ANCHOR_MAP_ID = "fixture-anchor-scalar-map-v1"
ANCHOR_MAP_HASH = H[6]


def _source_hashes():
    return {
        "score": {
            "manifest": H[0],
            "score_map": score_map_sha256(),
            "block_chain_receipt": H[1],
            "gamma_ps_receipt": H[2],
            "initial_law_tv_receipt": H[3],
            "review_receipt": H[4],
        },
        "anchor": {
            "manifest": H[5],
            "scalar_map": ANCHOR_MAP_HASH,
            "coverage_receipt": H[7],
            "common_bias_receipt": H[8],
            "anchor_chain_receipt": H[15],
            "gamma_ps_receipt": H[16],
            "initial_law_tv_receipt": H[17],
            "support_receipt": H[18],
            "score_anchor_independence_receipt": H[9],
            "review_receipt": H[10],
        },
    }


def _review(receipt_hash):
    return {
        "status": "reviewed",
        "receipt_sha256": receipt_hash,
        "scope_sha256": H[11],
        "reviewer_role_claim": "independent",
        "independence_claim_sha256": H[12],
    }


def _envelopes(*, score_gamma=0.2, score_tv=0.001, score_burn_in=8,
               anchor_gamma=0.2, anchor_tv=0.001, anchor_burn_in=8,
               score_chain_id="score-chain", anchor_chain_id="anchor-chain"):
    hashes = _source_hashes()
    score = evidence_envelope(
        "pion_projection", "score_dependence",
        chain={
            "chain_id": score_chain_id,
            "split_id": "score-development",
            "state_unit": "length-5 saved-configuration macroblock",
            "manifest_sha256": hashes["score"]["manifest"],
        },
        observable={
            "map_id": "pion-bounded-block-risk-selected-v1/bounded-score-map",
            "map_sha256": hashes["score"]["score_map"],
            "support": {"lower": -1, "upper": 1, "unit": "bounded score"},
        },
        method={
            "name": "markov_pseudo_gap_bernstein",
            "version": "fixture-v1",
            "assumptions": ["fixed score map over saved macroblocks"],
        },
        dependence={
            "mode": "markov",
            "assumptions": ["the receipt covers the exact saved-state chain"],
            "gamma_ps_lower": score_gamma,
            "tv_bound_after_burn_in": score_tv,
            "burn_in": {
                "unit": "length-5 saved-configuration macroblock",
                "count": score_burn_in,
            },
        },
        confidence={
            "family_id": "pion-bounded-block-risk-selected-v1/score-F3",
            "alpha_total": 0.05,
            "alpha_allocated": 0.04,
            "family_size": 3,
        },
        source_hashes=hashes["score"],
        review=_review(hashes["score"]["review_receipt"]),
    )
    anchor = evidence_envelope(
        "pion_projection", "anchor_coverage",
        chain={
            "chain_id": anchor_chain_id,
            "split_id": "anchor-evidence",
            "state_unit": "saved gauge-configuration state",
            "manifest_sha256": hashes["anchor"]["manifest"],
        },
        observable={
            "map_id": ANCHOR_MAP_ID,
            "map_sha256": ANCHOR_MAP_HASH,
            "support": {"lower": 0.1, "upper": 3.0, "unit": "a_t m_pi"},
        },
        method={
            "name": "markov_pseudo_gap_bernstein",
            "version": "fixture-v1",
            "assumptions": ["one fixed bounded scalar is emitted by each anchor-chain state"],
        },
        dependence={
            "mode": "markov",
            "assumptions": ["homogeneous anchor Markov chain with a reviewed pseudo-gap bound"],
            "gamma_ps_lower": anchor_gamma,
            "tv_bound_after_burn_in": anchor_tv,
            "burn_in": {
                "unit": "saved gauge-configuration state",
                "count": anchor_burn_in,
            },
        },
        confidence={
            "family_id": "pion-bounded-block-risk-selected-v1/anchor",
            "alpha_total": 0.05,
            "alpha_allocated": 0.01,
            "family_size": 1,
        },
        source_hashes=hashes["anchor"],
        review=_review(hashes["anchor"]["review_receipt"]),
    )
    return {"score": score, "anchor": anchor}, hashes


def _receipt(*, score_planned=12000, score_valid=12000, anchor_planned=4000,
             anchor_valid=4000):
    scenario = scenario_spec(
        "pion-statistical-evidence-fixture", "planned score or anchor unit",
        "synthetic task adapter interface fixture",
    )
    return run_receipt(
        scenario,
        {
            "score_family_blocks": denominator(
                "paired score value", planned=score_planned, valid=score_valid,
            ),
            "anchor_units": denominator(
                "anchor Markov-chain scalar state", planned=anchor_planned,
                valid=anchor_valid,
            ),
        },
        artifact_hashes={"fixture": H[13]},
        text_flow="fixture envelopes -> task protocol checks -> bundle gate -> draft or abstain",
    )


def _ready_state():
    state = RunState()
    for stage in ("operator", "reference", "numerical"):
        state.set(stage, "draft", evidence=f"synthetic {stage} prerequisite")
        state.set(stage, "validated", evidence_hash=H[14])
    return state


def _score_map_bytes():
    protocol = selected_protocol()
    score = protocol["score"]
    payload = {
        "protocol_id": protocol["protocol_id"],
        "prediction_domain": score["prediction_domain"],
        "units": score["units"],
        "R": score["R"],
        "formula": score["formula"],
        "clipping_interval": score["clipping_interval"],
        "failure_policy": score["failure_policy"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == score_map_sha256()
    return encoded


def _common_bias_receipt_bytes(map_hash, *, bound=0.01, target_hash=H[14]):
    payload = {
        "schema": "pion-anchor-common-bias-receipt-v1",
        "status": "reviewed_claim",
        "map_id": ANCHOR_MAP_ID,
        "map_sha256": map_hash,
        "target_reference_sha256": target_hash,
        "review_scope_sha256": H[11],
        "quantity": "abs(E[anchor_scalar_map]-m_star)/R",
        "bound_over_R": bound,
        "support_interval": [0.1, 3.0],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _materialize_source_bytes(root, envelopes, hashes):
    paths = {}
    anchor_map_hash = hashlib.sha256(b"synthetic:anchor:scalar_map").hexdigest()
    for route in ("score", "anchor"):
        route_paths = {}
        for name in tuple(hashes[route]):
            if route == "score" and name == "score_map":
                payload = _score_map_bytes()
            elif route == "anchor" and name == "scalar_map":
                payload = b"synthetic:anchor:scalar_map"
            elif route == "anchor" and name == "common_bias_receipt":
                payload = _common_bias_receipt_bytes(anchor_map_hash)
            else:
                payload = f"synthetic:{route}:{name}".encode()
            digest = hashlib.sha256(payload).hexdigest()
            relative_path = f"{route}-{name}.bin"
            (root / relative_path).write_bytes(payload)
            route_paths[name] = relative_path
            hashes[route][name] = digest
            envelopes[route]["source_hashes"][name] = digest
            if name == "manifest":
                envelopes[route]["chain"]["manifest_sha256"] = digest
            elif name == "review_receipt":
                envelopes[route]["review"]["receipt_sha256"] = digest
            elif route == "score" and name == "score_map":
                envelopes[route]["observable"]["map_sha256"] = digest
            elif route == "anchor" and name == "scalar_map":
                envelopes[route]["observable"]["map_sha256"] = digest
                anchor_map_hash = digest
        paths[route] = route_paths
    assert anchor_map_hash is not None
    return paths, anchor_map_hash


def _apply(state, envelopes, hashes, receipt=None, *, common_bias=0.01,
           anchor_map_hash=ANCHOR_MAP_HASH, source_root=None,
           source_paths_by_envelope=None, source_snapshots_by_envelope=None,
           require_source_bytes=False):
    return apply_pion_statistical_evidence_gate(
        state,
        envelopes,
        _receipt() if receipt is None else receipt,
        anchor_scalar_map_id=ANCHOR_MAP_ID,
        anchor_scalar_map_sha256=anchor_map_hash,
        common_bias_bound_over_R=common_bias,
        required_source_hashes_by_envelope=hashes,
        source_root=source_root,
        source_paths_by_envelope=source_paths_by_envelope,
        source_snapshots_by_envelope=source_snapshots_by_envelope,
        require_source_bytes=require_source_bytes,
    )


def test_pion_statistical_adapter_binds_selected_f3_defaults_and_stops_at_draft():
    envelopes, hashes = _envelopes()
    state = _ready_state()
    result = _apply(state, envelopes, hashes)
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["run_state"]["status"]["statistical"] == "draft"
    assert result["certified"] is False
    assert result["physical_score"] is None
    assert result["chain_identity_checks"][0]["independence_proven"] is False


def test_pion_statistical_adapter_abstains_without_upstream_reference_and_anchor_map():
    envelopes, hashes = _envelopes()
    state = RunState()
    result = _apply(state, envelopes, hashes)
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert state.status["statistical"] == "abstained"
    assert any("reference stage must be validated" in item for item in result["reasons"])


def test_pion_statistical_adapter_abstains_below_score_gap_or_above_common_bias_cap():
    envelopes, hashes = _envelopes(score_gamma=0.19)
    result = _apply(_ready_state(), envelopes, hashes)
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert any("gamma_ps" in item for item in result["reasons"])

    envelopes, hashes = _envelopes()
    result = _apply(_ready_state(), envelopes, hashes, common_bias=0.0101)
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert any("C/R" in item for item in result["reasons"])


def test_pion_statistical_adapter_applies_markov_anchor_gap_tv_and_burn_in_gates():
    for kwargs, expected_reason in (
        ({"anchor_gamma": 0.19}, "anchor: gamma_ps"),
        ({"anchor_tv": 0.0011}, "anchor: post-burn-in TV"),
        ({"anchor_burn_in": 7}, "anchor: burn-in"),
    ):
        envelopes, hashes = _envelopes(**kwargs)
        result = _apply(_ready_state(), envelopes, hashes)
        assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
        assert any(expected_reason in item for item in result["reasons"])


def test_pion_statistical_adapter_abstains_when_score_tv_or_burn_in_misses_plan():
    envelopes, hashes = _envelopes(score_tv=0.0011)
    result = _apply(_ready_state(), envelopes, hashes)
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert any("TV bound" in item for item in result["reasons"])

    envelopes, hashes = _envelopes(score_burn_in=7)
    result = _apply(_ready_state(), envelopes, hashes)
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert any("burn-in" in item for item in result["reasons"])


def test_pion_statistical_adapter_abstains_on_wrong_plan_or_chain_identity():
    envelopes, hashes = _envelopes()
    wrong_plan = _receipt(score_planned=11999, score_valid=11999)
    result = _apply(_ready_state(), envelopes, hashes, wrong_plan)
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert any("planned 11999" in item for item in result["reasons"])

    envelopes, hashes = _envelopes(anchor_chain_id="score-chain")
    result = _apply(_ready_state(), envelopes, hashes)
    assert result["status"] == "ABSTAIN_EVIDENCE_BUNDLE"
    assert result["run_state"]["status"]["statistical"] == "abstained"


def test_pion_statistical_adapter_does_not_accept_stale_map_hashes():
    envelopes, hashes = _envelopes()
    envelopes = copy.deepcopy(envelopes)
    envelopes["anchor"]["observable"]["map_sha256"] = H[18]
    result = _apply(_ready_state(), envelopes, hashes)
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert any("observable-map hash" in item for item in result["reasons"])


def test_pion_statistical_adapter_abstains_on_incomplete_source_hash_map():
    envelopes, hashes = _envelopes()
    bad_hashes = copy.deepcopy(hashes)
    del bad_hashes["anchor"]["common_bias_receipt"]
    result = _apply(_ready_state(), envelopes, bad_hashes)
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert any("common_bias_receipt" in item for item in result["reasons"])

    result = apply_pion_statistical_evidence_gate(
        _ready_state(),
        envelopes,
        _receipt(),
        anchor_scalar_map_id=ANCHOR_MAP_ID,
        anchor_scalar_map_sha256=ANCHOR_MAP_HASH,
        common_bias_bound_over_R=0.01,
        required_source_hashes_by_envelope={"score": hashes["score"]},
    )
    assert result["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert any("exactly score and anchor routes" in item for item in result["reasons"])


def test_pion_source_byte_gate_is_required_and_verified_bundle_stops_at_draft(tmp_path):
    envelopes, hashes = _envelopes()
    missing_paths_state = _ready_state()
    missing_paths = _apply(missing_paths_state, envelopes, hashes,
                           require_source_bytes=True)
    assert missing_paths["status"] == "ABSTAIN_PION_STATISTICAL_ADAPTER"
    assert missing_paths_state.status["statistical"] == "abstained"
    assert any("requires both source_root" in item for item in missing_paths["reasons"])

    envelopes, hashes = _envelopes()
    paths, anchor_map_hash = _materialize_source_bytes(tmp_path, envelopes, hashes)
    state = _ready_state()
    result = _apply(
        state,
        envelopes,
        hashes,
        anchor_map_hash=anchor_map_hash,
        source_root=tmp_path,
        source_paths_by_envelope=paths,
        require_source_bytes=True,
    )

    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_byte_identity_verified"] is True
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file",
        "planned": 16,
        "valid": 16,
        "failed": 0,
        "abstained": 0,
        "not_run": 0,
    }
    assert result["common_bias_receipt_verification"]["status"] == "internally_consistent_claim"
    assert result["common_bias_receipt_verification"]["denominator"] == {
        "unit": "captured_common_bias_receipt_claim", "planned": 1,
        "valid": 1, "failed": 0, "abstained": 0, "not_run": 0,
    }
    assert result["run_state"]["status"]["statistical"] == "draft"
    assert result["physical_score"] is None


def test_pion_source_snapshot_bundle_survives_path_changes_after_reader_capture(tmp_path):
    envelopes, hashes = _envelopes()
    paths, anchor_map_hash = _materialize_source_bytes(tmp_path, envelopes, hashes)
    snapshots = {
        route: capture_source_snapshot(tmp_path, hashes[route], paths[route])
        for route in ("score", "anchor")
    }
    score_input = snapshots["score"].read_bytes("score_map")
    (tmp_path / paths["anchor"]["common_bias_receipt"]).write_bytes(
        b"changed after the reader captured its input bytes"
    )

    result = _apply(
        _ready_state(), envelopes, hashes,
        anchor_map_hash=anchor_map_hash,
        source_snapshots_by_envelope=snapshots,
        require_source_bytes=True,
    )

    assert hashlib.sha256(score_input).hexdigest() == hashes["score"]["score_map"]
    assert result["status"] == "DRAFT_PENDING_TASK_VALIDATION"
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 16, "valid": 16,
        "failed": 0, "abstained": 0, "not_run": 0,
    }
    assert result["common_bias_receipt_verification"]["denominator"]["valid"] == 1
    assert result["physical_score"] is None


def test_pion_source_byte_gate_abstains_if_one_receipt_file_is_tampered(tmp_path):
    envelopes, hashes = _envelopes()
    paths, anchor_map_hash = _materialize_source_bytes(tmp_path, envelopes, hashes)
    tampered_path = tmp_path / paths["anchor"]["common_bias_receipt"]
    tampered_path.write_bytes(b"tampered after receipt creation")

    state = _ready_state()
    result = _apply(
        state,
        envelopes,
        hashes,
        anchor_map_hash=anchor_map_hash,
        source_root=tmp_path,
        source_paths_by_envelope=paths,
        require_source_bytes=True,
    )

    assert result["status"] == "ABSTAIN_SOURCE_BYTES"
    assert result["source_verification"]["denominator"]["planned"] == 16
    assert result["source_verification"]["denominator"]["valid"] == 15
    assert result["source_verification"]["denominator"]["failed"] == 1
    assert state.status["statistical"] == "abstained"


def test_snapshot_common_bias_receipt_is_the_caller_c_over_r_source(tmp_path):
    envelopes, hashes = _envelopes()
    paths, anchor_map_hash = _materialize_source_bytes(tmp_path, envelopes, hashes)
    receipt_path = tmp_path / paths["anchor"]["common_bias_receipt"]
    receipt_path.write_bytes(_common_bias_receipt_bytes(anchor_map_hash, bound=0.005))
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    hashes["anchor"]["common_bias_receipt"] = digest
    envelopes["anchor"]["source_hashes"]["common_bias_receipt"] = digest
    snapshots = {
        route: capture_source_snapshot(tmp_path, hashes[route], paths[route])
        for route in ("score", "anchor")
    }

    result = _apply(
        _ready_state(), envelopes, hashes,
        common_bias=0.01, anchor_map_hash=anchor_map_hash,
        source_snapshots_by_envelope=snapshots, require_source_bytes=True,
    )

    assert result["status"] == "ABSTAIN_PION_COMMON_BIAS_RECEIPT"
    assert any("caller C/R differs" in reason for reason in result["reasons"])
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 16, "valid": 16,
        "failed": 0, "abstained": 0, "not_run": 0,
    }
    assert result["common_bias_receipt_verification"]["denominator"] == {
        "unit": "captured_common_bias_receipt_claim", "planned": 1,
        "valid": 0, "failed": 1, "abstained": 0, "not_run": 0,
    }
    assert result["physical_score"] is None


def test_source_root_and_snapshot_routes_abstain_on_the_same_bias_claim(tmp_path):
    envelopes, hashes = _envelopes()
    paths, anchor_map_hash = _materialize_source_bytes(tmp_path, envelopes, hashes)
    receipt_path = tmp_path / paths["anchor"]["common_bias_receipt"]
    receipt_path.write_bytes(_common_bias_receipt_bytes(anchor_map_hash, bound=0.005))
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    hashes["anchor"]["common_bias_receipt"] = digest
    envelopes["anchor"]["source_hashes"]["common_bias_receipt"] = digest
    snapshots = {
        route: capture_source_snapshot(tmp_path, hashes[route], paths[route])
        for route in ("score", "anchor")
    }

    source_root_result = _apply(
        _ready_state(), copy.deepcopy(envelopes), copy.deepcopy(hashes),
        common_bias=0.01, anchor_map_hash=anchor_map_hash,
        source_root=tmp_path, source_paths_by_envelope=paths, require_source_bytes=True,
    )
    snapshot_result = _apply(
        _ready_state(), copy.deepcopy(envelopes), copy.deepcopy(hashes),
        common_bias=0.01, anchor_map_hash=anchor_map_hash,
        source_snapshots_by_envelope=snapshots, require_source_bytes=True,
    )

    for result in (source_root_result, snapshot_result):
        assert result["status"] == "ABSTAIN_PION_COMMON_BIAS_RECEIPT"
        assert any("caller C/R differs" in reason for reason in result["reasons"])
        assert result["source_verification"]["denominator"] == {
            "unit": "source_file", "planned": 16, "valid": 16,
            "failed": 0, "abstained": 0, "not_run": 0,
        }
        assert result["common_bias_receipt_verification"]["denominator"] == {
            "unit": "captured_common_bias_receipt_claim", "planned": 1,
            "valid": 0, "failed": 1, "abstained": 0, "not_run": 0,
        }
        assert result["run_state"]["status"]["statistical"] == "abstained"
        assert result["physical_score"] is None


def test_snapshot_common_bias_receipt_must_bind_validated_reference(tmp_path):
    envelopes, hashes = _envelopes()
    paths, anchor_map_hash = _materialize_source_bytes(tmp_path, envelopes, hashes)
    receipt_path = tmp_path / paths["anchor"]["common_bias_receipt"]
    receipt_path.write_bytes(
        _common_bias_receipt_bytes(anchor_map_hash, target_hash=H[13]),
    )
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    hashes["anchor"]["common_bias_receipt"] = digest
    envelopes["anchor"]["source_hashes"]["common_bias_receipt"] = digest
    snapshots = {
        route: capture_source_snapshot(tmp_path, hashes[route], paths[route])
        for route in ("score", "anchor")
    }

    result = _apply(
        _ready_state(), envelopes, hashes,
        anchor_map_hash=anchor_map_hash,
        source_snapshots_by_envelope=snapshots, require_source_bytes=True,
    )

    assert result["status"] == "ABSTAIN_PION_COMMON_BIAS_RECEIPT"
    assert any("validated reference stage" in reason for reason in result["reasons"])
    assert result["common_bias_receipt_verification"]["denominator"]["failed"] == 1


def test_snapshot_common_bias_receipt_rejects_duplicate_json_keys(tmp_path):
    envelopes, hashes = _envelopes()
    paths, anchor_map_hash = _materialize_source_bytes(tmp_path, envelopes, hashes)
    receipt_path = tmp_path / paths["anchor"]["common_bias_receipt"]
    payload = _common_bias_receipt_bytes(anchor_map_hash).decode()
    payload = payload.replace('"bound_over_R":0.01', '"bound_over_R":0.01,"bound_over_R":0.0')
    receipt_path.write_text(payload)
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    hashes["anchor"]["common_bias_receipt"] = digest
    envelopes["anchor"]["source_hashes"]["common_bias_receipt"] = digest
    snapshots = {
        route: capture_source_snapshot(tmp_path, hashes[route], paths[route])
        for route in ("score", "anchor")
    }

    result = _apply(
        _ready_state(), envelopes, hashes,
        anchor_map_hash=anchor_map_hash,
        source_snapshots_by_envelope=snapshots, require_source_bytes=True,
    )

    assert result["status"] == "ABSTAIN_PION_COMMON_BIAS_RECEIPT"
    assert any("duplicate JSON key: bound_over_R" in reason for reason in result["reasons"])
    assert result["source_verification"]["denominator"] == {
        "unit": "source_file", "planned": 16, "valid": 16,
        "failed": 0, "abstained": 0, "not_run": 0,
    }
    assert result["common_bias_receipt_verification"]["denominator"]["failed"] == 1
    assert result["physical_score"] is None
