"""Versioned scenario and run receipts for reproducible task diagnostics.

Receipts record counts and identities; they do not authenticate independence,
measurement provenance, or scientific validity.
"""
import re


SCENARIO_SCHEMA = "latticearena-scenario-v1"
RECEIPT_SCHEMA = "latticearena-run-receipt-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _hash_map(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping of names to SHA-256 hashes")
    for name, digest in value.items():
        _text(name, f"{label} name")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"{label}[{name!r}] must be a lowercase SHA-256 hash")
    return dict(value)


def scenario_spec(name, unit, description, *, assumptions=(), seed=None,
                  source_hashes=None):
    """Create a validated scenario descriptor without inferring independence."""
    _text(name, "scenario name")
    _text(unit, "scenario unit")
    _text(description, "scenario description")
    if isinstance(assumptions, str) or not isinstance(assumptions, (list, tuple)):
        raise ValueError("assumptions must be a sequence of nonempty strings")
    assumptions = [_text(item, "assumption") for item in assumptions]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    return {
        "schema": SCENARIO_SCHEMA,
        "name": name,
        "unit": unit,
        "description": description,
        "assumptions": assumptions,
        "seed": seed,
        "source_hashes": _hash_map(
            {} if source_hashes is None else source_hashes, "source_hashes"
        ),
    }


def denominator(unit, *, planned, valid=0, failed=0, abstained=0):
    """Return explicit counts and derive, but never hide, the not-run remainder."""
    _text(unit, "denominator unit")
    counts = {"planned": planned, "valid": valid, "failed": failed,
              "abstained": abstained}
    for name, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    completed = valid + failed + abstained
    if completed > planned:
        raise ValueError("valid + failed + abstained cannot exceed planned")
    return {"unit": unit, **counts, "not_run": planned - completed}


def run_receipt(scenario, denominators, *, artifact_hashes, text_flow,
                reason_rows=()):
    """Bind a scenario to stage-specific denominators and output identities.

    ``artifact_hashes`` should cover the report payloads. The receipt file itself
    may be covered by an external manifest to avoid a self-hash cycle.
    """
    if not isinstance(scenario, dict) or scenario.get("schema") != SCENARIO_SCHEMA:
        raise ValueError("scenario must be a latticearena-scenario-v1 object")
    scenario_spec(
        scenario.get("name"), scenario.get("unit"), scenario.get("description"),
        assumptions=scenario.get("assumptions"), seed=scenario.get("seed"),
        source_hashes=scenario.get("source_hashes"),
    )
    if not isinstance(denominators, dict) or not denominators:
        raise ValueError("denominators must be a nonempty mapping")
    normalized = {}
    for name, counts in denominators.items():
        _text(name, "denominator name")
        if not isinstance(counts, dict):
            raise ValueError(f"denominator {name!r} must be a mapping")
        result = denominator(
            counts.get("unit"), planned=counts.get("planned"),
            valid=counts.get("valid"), failed=counts.get("failed"),
            abstained=counts.get("abstained"),
        )
        if "not_run" in counts:
            not_run = counts["not_run"]
            if isinstance(not_run, bool) or not isinstance(not_run, int) or not_run < 0:
                raise ValueError(f"denominator {name!r} not_run must be a nonnegative integer")
            if not_run != result["not_run"]:
                raise ValueError(f"denominator {name!r} has an inconsistent not_run count")
        normalized[name] = result
    reason_rows = _validate_reasons(reason_rows, normalized)
    artifacts = _hash_map(artifact_hashes, "artifact_hashes")
    _text(text_flow, "text_flow")
    status = "complete" if all(item["not_run"] == 0 for item in normalized.values()) else "partial"
    return {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "scenario": scenario,
        "denominators": normalized,
        "reason_rows": reason_rows,
        "artifact_hashes": artifacts,
        "text_flow": text_flow,
    }


def _validate_reasons(rows, denominators):
    if isinstance(rows, str) or not isinstance(rows, (list, tuple)):
        raise ValueError("reason_rows must be a sequence of reason records")
    totals = {name: {state: 0 for state in ("failed", "abstained", "not_run")}
              for name in denominators}
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each reason row must be a mapping")
        unit = _text(row.get("unit"), "reason unit")
        state = row.get("state")
        if unit not in denominators or state not in ("failed", "abstained", "not_run"):
            raise ValueError("reason row must identify a known unit and failed/abstained/not_run state")
        count = row.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("reason count must be a positive integer")
        reason = _text(row.get("reason"), "reason")
        totals[unit][state] += count
        normalized.append({"unit": unit, "state": state, "count": count,
                           "reason": reason})
    for unit, counts in denominators.items():
        for state in ("failed", "abstained", "not_run"):
            if totals[unit][state] != counts[state]:
                raise ValueError(
                    f"reason rows for {unit!r}/{state} total {totals[unit][state]}, "
                    f"expected {counts[state]}"
                )
    return normalized
