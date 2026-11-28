import csv
import hashlib
import json

from tasks.wilson_loop.lane_contract import (
    CONTRACT_PATH,
    _synthetic_contract,
    build,
    empty_contract,
    inspect_contract,
    selected_protocol,
)


def test_null_draft_stays_not_ready_with_two_separate_estimands():
    result = inspect_contract(empty_contract())
    assert result["schema_valid"]
    assert result["status"] == "DRAFT_NOT_READY"
    assert result["physical_score"] is None
    assert result["certified"] is False
    assert set(result["lanes"]) == {
        "finite_T_Creutz", "ground_state_discrete_static_force",
    }
    assert all(row["status"] == "NOT_READY" for row in result["lanes"].values())
    protocol = empty_contract()["selected_protocol"]
    assert protocol == selected_protocol()
    assert protocol["finite_T_Creutz"]["R"] == 2
    assert protocol["finite_T_Creutz"]["T"] == 5
    assert protocol["finite_T_Creutz"]["data_receipt_sha256"] is None
    assert protocol["ground_state_discrete_static_force"]["midpoint_R"] == 2.5
    assert protocol["ground_state_discrete_static_force"]["evidence_receipts"][
        "thermal_h_Nt32"] is None
    spectral = protocol["ground_state_discrete_static_force"]["spectral_evidence_plan"]
    q_contract = spectral["q_overlap_contract"]
    assert q_contract["definition_id"] == "bare_basis_squared_overlap_ratio_v1"
    assert q_contract["kernel_to_basis_operator"] == {
        "identity": "identity",
        "ape_alpha0.4_n5": "APE",
        "spatial_stout_rho0.10_n5": "stout",
    }
    assert "must not be copied" in q_contract["gevp_role"]
    trace = q_contract["finite_Nt_trace_decomposition"]
    assert trace["definition_id"] == "two_sector_transfer_trace_qhx_v1"
    assert trace["component_sectors"] == {
        "q": "m=0,n>0: vacuum ground state with excited static-source states",
        "h": "m>0,n=0: excited vacuum-sector states with static-source ground state",
        "x": "m>0,n>0: excited states in both transfer-matrix sectors",
    }
    assert trace["status"] == "PROPOSED_TRACE_INTERPRETATION_REQUIRES_INDEPENDENT_REVIEW"
    assert "production exporter time origin" in trace["endpoint_convention"]
    contamination = spectral["contamination_stencil"]
    assert contamination["primary_effective_energy_T"] == 5
    assert contamination["loop_time_extents"] == [5, 6]
    assert set(contamination["loop_time_extents"]).issubset(
        range(*[spectral["fit_window"][0], spectral["fit_window"][1] + 1])
    )
    assert contamination["status"] == (
        "PROPOSED_METHOD_DEFAULT_REQUIRES_INDEPENDENT_REVIEW"
    )
    assert protocol["gates"]["ground_state_force_certified"] is False


def test_selected_protocol_json_and_runtime_contract_share_contamination_stencil():
    frozen = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    runtime = selected_protocol()
    draft = empty_contract()
    assert frozen["selected_protocol"] == runtime
    assert frozen["lanes"]["ground_state_discrete_static_force"][
        "spectral_evidence_plan"
    ] == draft["lanes"]["ground_state_discrete_static_force"][
        "spectral_evidence_plan"
    ]


def test_selected_lattice_force_draft_does_not_require_physical_scale_conversion():
    contract = empty_contract()
    result = inspect_contract(contract)
    assert contract["common"]["lattice_spacing"]["scale_status"] == (
        "not_required_for_dimensionless_lattice_estimands"
    )
    assert not any("common.lattice_spacing" in issue
                   for issue in result["common_issues"])
    assert result["physical_score"] is None
    assert not result["lanes"]["ground_state_discrete_static_force"][
        "structurally_complete"
    ]


def test_force_error_budget_requires_model_truncation_instead_of_extrapolations():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    force = contract["lanes"]["ground_state_discrete_static_force"]
    names = {row["component"] for row in force["systematic_bounds"]}
    assert names == {
        "excited_state_force", "finite_T_thermal", "mixed",
        "spectral_model_truncation",
    }

    force["systematic_bounds"] = [
        row for row in force["systematic_bounds"]
        if row["component"] != "spectral_model_truncation"
    ]
    force["systematic_bounds"].append({
        "component": "finite_a", "bound": 0.0,
        "units": force["units"], "evidence_sha256": "c" * 64,
        "scope": "SYNTHETIC_ONLY outside fixed-lattice estimand",
        "independent_review_receipt": "SYNTHETIC_ONLY_review",
    })
    result = inspect_contract(contract)
    issues = result["lanes"]["ground_state_discrete_static_force"]["issues"]
    assert any("spectral_model_truncation: missing required evidence" in issue
               for issue in issues)
    assert not any("finite_a: missing required evidence" in issue for issue in issues)


def test_qhx_caps_are_post_propagation_and_bind_the_four_stout_corner_formula():
    force = empty_contract()["lanes"]["ground_state_discrete_static_force"]
    propagation = force["force_bound_propagation"]
    assert force["estimand"] == "absolute_ground_state_discrete_static_force"
    assert force["formula"] == "U0_stout(3)-U0_stout(2)"
    assert force["reference_formula"] == "U0_identity(3)-U0_identity(2)"
    assert "not the primary estimand" in force["reference_role"]
    assert len(force["force_stencil"]) == 2
    assert len(force["reference_force_stencil"]) == 2
    assert propagation["corner_count"] == 4
    assert len(propagation["corner_layout"]) == 4
    assert propagation["uniform_raw_loop_cap"] == (
        "expm1(force_fraction*positive_force_scale_lower/4)"
    )
    for row in force["spectral_bounds"]:
        assert "post-propagation absolute stout-force component budget" in row["scope"]
        assert row["uniform_raw_loop_cap_formula"].endswith("/4)")


def test_identity_force_stencil_is_a_separate_consistency_check():
    contract = empty_contract()
    force = contract["lanes"]["ground_state_discrete_static_force"]
    force["force_stencil"][1]["kernel_id"] = "identity"
    result = inspect_contract(contract)
    issues = result["lanes"]["ground_state_discrete_static_force"]["issues"]
    assert any("force_stencil: terms do not match its declared kernel" in issue
               for issue in issues)


def test_selected_lane_alpha_allocation_closes_the_familywise_budget():
    protocol = selected_protocol()
    allocation = protocol["familywise_alpha_allocation"]
    assert allocation["alpha_creutz_lane"] + allocation["alpha_force_lane"] == 0.05
    assert (allocation["alpha_force_statistics"]
            + allocation["alpha_force_systematics"]
            == allocation["alpha_force_lane"])
    assert (allocation["alpha_force_primary_interval"]
            + allocation["alpha_positive_force_scale_lower_bound"]
            == allocation["alpha_force_statistics"])
    budget = protocol["ground_state_discrete_static_force"]["relative_error_budget"]
    n_cells = budget["q_h_x_cells"] + budget["model_truncation_cells"]
    assert n_cells == allocation["force_systematic_bound_cells"]
    assert (n_cells * allocation["alpha_per_force_systematic_cell"]
            == allocation["alpha_force_systematics"])


def test_changed_protocol_defaults_fail_closed_without_certifying_evidence():
    contract = empty_contract()
    contract["selected_protocol"]["finite_T_Creutz"]["precision_halfwidth_target"] = 0.5
    result = inspect_contract(contract)
    assert result["status"] == "DRAFT_NOT_READY"
    assert all(row["status"] == "NOT_READY" for row in result["lanes"].values())
    assert any("selected_protocol: differs" in message
               for message in result["common_issues"])
    assert result["physical_score"] is None


def test_c_complete_does_not_make_f_ready_and_complete_fixture_is_not_certified():
    c_only = inspect_contract(_synthetic_contract(
        complete_creutz=True, complete_force=False,
    ))
    assert c_only["lanes"]["finite_T_Creutz"]["structurally_complete"]
    assert c_only["lanes"]["finite_T_Creutz"]["status"] == (
        "CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW"
    )
    assert c_only["lanes"]["ground_state_discrete_static_force"]["status"] == "NOT_READY"
    both = inspect_contract(_synthetic_contract(
        complete_creutz=True, complete_force=True,
    ))
    for row in both["lanes"].values():
        assert row["status"] == "CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW"
        assert row["certified"] is False
        assert row["physical_score"] is None


def test_global_sign_reversal_is_rejected_for_both_physical_conventions():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    creutz = contract["lanes"]["finite_T_Creutz"]
    for row in creutz["stencil"]:
        row["coefficient"] *= -1
    result = inspect_contract(contract)
    c_issues = result["lanes"]["finite_T_Creutz"]["issues"]
    assert any("prescribed signed four-point convention" in message
               for message in c_issues)
    assert not result["lanes"]["finite_T_Creutz"]["structurally_complete"]

    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    force = contract["lanes"]["ground_state_discrete_static_force"]
    for row in force["force_stencil"]:
        row["coefficient"] *= -1
    result = inspect_contract(contract)
    f_issues = result["lanes"]["ground_state_discrete_static_force"]["issues"]
    assert any("coefficients must encode R+1 minus R" in message
               for message in f_issues)
    assert not result["lanes"]["ground_state_discrete_static_force"]["structurally_complete"]


def test_invalid_temporal_extent_boolean_integer_and_stencil_are_rejected():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    contract["common"]["geometry"]["temporal_extent"] = 3
    contract["lanes"]["finite_T_Creutz"]["primary"]["R"] = True
    result = inspect_contract(contract)
    c = result["lanes"]["finite_T_Creutz"]
    assert not c["structurally_complete"]
    assert any("primary.T: T+1 exceeds" in message for message in c["issues"])
    assert any("primary.R: expected a positive integer" in message
               for message in c["issues"])


def test_missing_qhx_receipt_alpha_overflow_and_budget_overrun_are_rejected():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    force = contract["lanes"]["ground_state_discrete_static_force"]
    force["spectral_bounds"] = [row for row in force["spectral_bounds"]
                                 if row["component"] != "q"]
    force["systematic_bounds"][0]["bound"] = 0.02
    force["systematic_bounds"][0]["independent_review_receipt"] = ""
    force["inference"]["alpha_allocations"] = {"first": 0.04, "second": 0.02}
    result = inspect_contract(contract)
    issues = result["lanes"]["ground_state_discrete_static_force"]["issues"]
    assert any("spectral_bounds.q: missing required evidence" in message
               for message in issues)
    assert any("independent_review_receipt: missing" in message for message in issues)
    assert any("alpha_allocations: sum exceeds alpha_total" in message
               for message in issues)
    assert any("force-level systematic plus statistical budgets exceed total_tolerance" in message
               for message in issues)


def test_qhx_inputs_are_dimensionless_and_force_budget_is_in_force_units():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    force = contract["lanes"]["ground_state_discrete_static_force"]
    assert force["spectral_bounds"][0]["units"] == "dimensionless"
    assert all(row["units"] == force["units"] for row in force["systematic_bounds"])
    force["spectral_bounds"][0]["units"] = force["units"]
    result = inspect_contract(contract)
    issues = result["lanes"]["ground_state_discrete_static_force"]["issues"]
    assert any("spectral_bounds.q.units: missing or inconsistent" in message
               for message in issues)


def test_force_precision_with_wrong_units_is_rejected_and_not_summed_as_force():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    force = contract["lanes"]["ground_state_discrete_static_force"]
    force["inference"]["statistical_precision_target"] = 100.0
    force["inference"]["precision_units"] = "dimensionless"
    result = inspect_contract(contract)
    issues = result["lanes"]["ground_state_discrete_static_force"]["issues"]
    assert any("precision_units: expected 'SYNTHETIC_ONLY_GeV'" in message
               for message in issues)
    assert not any("systematic plus statistical budgets exceed total_tolerance" in message
                   for message in issues)


def test_creutz_precision_must_be_dimensionless():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    creutz = contract["lanes"]["finite_T_Creutz"]
    assert creutz["inference"]["precision_units"] == "dimensionless"
    assert inspect_contract(contract)["lanes"]["finite_T_Creutz"][
        "structurally_complete"
    ]

    creutz["inference"]["precision_units"] = "SYNTHETIC_ONLY_GeV"
    result = inspect_contract(contract)
    lane = result["lanes"]["finite_T_Creutz"]
    assert not lane["structurally_complete"]
    assert any("precision_units: expected 'dimensionless'" in message
               for message in lane["issues"])
    assert result["physical_score"] is None


def test_null_common_kernel_list_is_reported_without_crashing():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    contract["common"]["kernel_ids"] = None
    result = inspect_contract(contract)
    assert any("common.kernel_ids: expected unique nonempty identifiers" in message
               for message in result["common_issues"])
    assert all(not row["structurally_complete"]
               for row in result["lanes"].values())


def test_schema_mismatch_cannot_return_structural_completeness():
    contract = _synthetic_contract(complete_creutz=True, complete_force=True)
    contract["schema"] = "future-or-unknown-schema"
    result = inspect_contract(contract)
    assert result["schema_valid"] is False
    assert all(not row["structurally_complete"] for row in result["lanes"].values())
    assert all(any("contract.schema" in message for message in row["issues"])
               for row in result["lanes"].values())


def test_build_report_snapshots_source_and_hashes_every_artifact(tmp_path):
    output = tmp_path / "wilson-report"
    summary = build(output)
    manifest = json.loads((output / "artifact_hashes.json").read_text())
    assert "snapshot_lane_contract.py" in manifest
    assert "lane_contract_v2.json" in manifest
    assert "snapshot_core_receipts.py" in manifest
    assert "snapshot_core_state.py" in manifest
    assert "run_receipt.json" in manifest
    assert summary["physical_score"] is None
    assert summary["run_state"]["physical_ready"] is False
    assert summary["run_state"]["score"] is None
    assert set(summary["run_state"]["status"].values()) == {"abstained"}
    receipt = json.loads((output / "run_receipt.json").read_text())
    assert receipt["schema"] == "latticearena-run-receipt-v1"
    assert receipt["status"] == "complete"
    assert len(receipt["denominators"]) == 17
    assert sum(row["planned"] for row in receipt["denominators"].values()) == 21
    assert sum(row["valid"] for row in receipt["denominators"].values()) == 7
    assert sum(row["failed"] for row in receipt["denominators"].values()) == 9
    assert sum(row["abstained"] for row in receipt["denominators"].values()) == 5
    assert receipt["denominators"]["physical_run_state_gates"] == {
        "unit": "one physical certification stage in core.RunState",
        "planned": 5, "valid": 0, "failed": 0, "abstained": 5, "not_run": 0,
    }
    assert sum(row["count"] for row in receipt["reason_rows"]
               if row["state"] == "failed") == 9
    assert sum(row["count"] for row in receipt["reason_rows"]
               if row["state"] == "abstained") == 5
    assert "not scientific success" in receipt["scenario"]["description"] or (
        "not scientific success" in receipt["text_flow"]
    )
    report = (output / "README.md").read_text()
    assert "|fixture|lane|planned|valid|failed|abstained|status|" in report
    assert "Current no-data RunState projection" in report
    assert "planned/valid/failed/abstained=`5/0/0/5`" in report
    assert "complete` means all planned units are accounted for" in report
    assert "## Next work after the selected protocol defaults" in report
    assert "Keep the selected C estimand fixed" in report
    assert "freeze C's primary rectangle" not in report
    with (output / "lane_summary.csv").open(newline="") as stream:
        denominator_rows = list(csv.DictReader(stream))
    assert sum(int(row["planned"]) for row in denominator_rows) == 16
    assert sum(int(row["valid"]) for row in denominator_rows) == 7
    assert sum(int(row["failed"]) for row in denominator_rows) == 9
    assert sum(int(row["abstained"]) for row in denominator_rows) == 0
    assert all(int(row["planned"]) == int(row["valid"]) + int(row["failed"])
               + int(row["abstained"]) for row in denominator_rows)
    c_reverse = summary["fixtures"]["synthetic_C_reversed_sign"]["lanes"][
        "finite_T_Creutz"]
    assert not c_reverse["structurally_complete"]
    f_reverse = summary["fixtures"]["synthetic_F_reversed_sign"]["lanes"][
        "ground_state_discrete_static_force"]
    assert not f_reverse["structurally_complete"]
    f_units = summary["fixtures"]["synthetic_F_wrong_precision_units"]["lanes"][
        "ground_state_discrete_static_force"]
    assert not f_units["structurally_complete"]
    assert any("precision_units: expected" in message for message in f_units["issues"])
    c_units = summary["fixtures"]["synthetic_C_wrong_precision_units"]["lanes"][
        "finite_T_Creutz"]
    assert not c_units["structurally_complete"]
    assert any("precision_units: expected 'dimensionless'" in message
               for message in c_units["issues"])
    null_kernels = summary["fixtures"]["synthetic_null_common_kernel_ids"]["lanes"]
    assert all(not row["structurally_complete"] for row in null_kernels.values())
    for name, expected in manifest.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    for name, expected in receipt["artifact_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert "core/state.py" in receipt["scenario"]["source_hashes"]


def test_saved_lane_contract_matches_selected_protocol_template():
    from pathlib import Path

    path = Path(__file__).parents[1] / "tasks/wilson_loop/lane_contract_v2.json"
    assert json.loads(path.read_text()) == empty_contract()
