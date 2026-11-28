"""Explain the selected bare-basis meaning of Wilson F-lane q cells.

The diagnostic uses the existing exact four-state toy only. It contrasts each
bare smearing operator with one fixed GEVP-composite operator to show why the
kernel-indexed q cells must name their overlap object explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .gevp_overlap_object_diagnostic import (
    CONTAMINATION_TIMES,
    T0,
    principal_gevp_vector,
    projected_state_overlaps,
    spectral_q,
)
from .gevp_t0_truncation_diagnostic import (
    OVERLAPS_4X3,
    TOY_ENERGIES_4,
    correlation_matrix,
)
from .lane_contract import CONTRACT_PATH, q_overlap_contract


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TASK_DIR.parents[1] / "results/wilson_loop/q_cell_semantics_diagnostic_20261126_v1"
TD = 2
KERNELS = (
    ("identity", "identity", 0),
    ("ape_alpha0.4_n5", "APE (descriptive)", 1),
    ("spatial_stout_rho0.10_n5", "stout", 2),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calculate() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    energies = np.asarray(TOY_ENERGIES_4, dtype=float)
    overlaps = np.asarray(OVERLAPS_4X3, dtype=complex)
    matrices = {time: correlation_matrix(energies, overlaps, time)
                for time in range(T0, TD + 1)}
    vector, eigenvalue = principal_gevp_vector(matrices[TD], matrices[T0])
    projected = projected_state_overlaps(overlaps, vector)
    rows: list[dict[str, Any]] = []
    for kernel_id, basis_operator_id, column in KERNELS:
        for time_extent in CONTAMINATION_TIMES:
            rows.append({
                "object_kind": "bare_basis_operator",
                "kernel_id": kernel_id,
                "basis_operator_id": basis_operator_id,
                "gevp_td": "",
                "T": time_extent,
                "q_T": spectral_q(energies, overlaps[:, column], time_extent),
                "use": "selected_force_q" if kernel_id != "ape_alpha0.4_n5" else "descriptive_only",
            })
    for time_extent in CONTAMINATION_TIMES:
        rows.append({
            "object_kind": "gevp_composite_operator",
            "kernel_id": "",
            "basis_operator_id": "principal_vector_composite",
            "gevp_td": TD,
            "T": time_extent,
            "q_T": spectral_q(energies, projected, time_extent),
            "use": "separate_diagnostic_not_kernel_q",
        })
    summary = {
        "status": "METHOD_MAPPING_DIAGNOSTIC_ONLY",
        "overlap_definition_id": q_overlap_contract()["definition_id"],
        "selected_bare_q_cells": 4,
        "descriptive_bare_basis_cells": 2,
        "separate_gevp_composite_cells": 2,
        "physical_measurements": 0,
        "certification_gate": {"planned": 1, "valid": 0, "failed": 0, "abstained": 1},
        "physical_score": None,
        "certified": False,
        "gevp_principal_eigenvalue": float(eigenvalue),
        "gevp_metric_norm": float(np.real(vector.conj() @ matrices[T0] @ vector)),
    }
    return rows, summary


def build(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    rows, summary = _calculate()
    output_dir.mkdir(parents=True)

    with (output_dir / "q_objects.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {"identity": "#3973ac", "APE (descriptive)": "#929292",
              "stout": "#d46a28", "GEVP composite": "#714c9f"}
    for axis, time_extent in zip(axes, CONTAMINATION_TIMES):
        selected = [row for row in rows if row["T"] == time_extent]
        labels, values, bar_colors = [], [], []
        for row in selected:
            label = ("GEVP composite" if row["object_kind"] == "gevp_composite_operator"
                     else row["basis_operator_id"])
            labels.append(label)
            values.append(row["q_T"])
            bar_colors.append(colors[label])
        bars = axis.bar(np.arange(len(values)), values, color=bar_colors, width=0.72)
        axis.set_yscale("log")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=22, ha="right")
        axis.set_title(f"positive forward q(T={time_extent})")
        axis.grid(axis="y", alpha=0.25, which="both")
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.3g}", (bar.get_x() + bar.get_width()/2, value),
                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_ylabel("q(T), logarithmic scale")
    fig.suptitle("Kernel-indexed q uses each bare smearing operator")
    fig.tight_layout()
    figure_path = output_dir / "q_bare_basis_vs_gevp.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    contract = q_overlap_contract()
    source_paths = {
        "q_cell_semantics_diagnostic.py": Path(__file__),
        "gevp_overlap_object_diagnostic.py": Path(__file__).with_name("gevp_overlap_object_diagnostic.py"),
        "gevp_t0_truncation_diagnostic.py": Path(__file__).with_name("gevp_t0_truncation_diagnostic.py"),
        "lane_contract.py": Path(__file__).with_name("lane_contract.py"),
        "lane_contract_v2.json": CONTRACT_PATH,
    }
    source_hashes = {name: _sha(path) for name, path in source_paths.items()}
    protocol = {
        "schema": "wilson-f-q-object-semantics-diagnostic-v1",
        "classification": "exact synthetic forward-spectrum; method mapping only",
        "selected_q_overlap_contract": contract,
        "synthetic_input": {
            "energies": np.asarray(TOY_ENERGIES_4, dtype=float).tolist(),
            "overlaps_state_by_operator": np.asarray(OVERLAPS_4X3, dtype=complex).real.tolist(),
            "basis_order": [entry[1] for entry in KERNELS],
            "t0": T0,
            "td": TD,
            "T_values": list(CONTAMINATION_TIMES),
        },
        "source_sha256": source_hashes,
        "limitations": [
            "does not use raw Wilson measurements or estimate physical q bounds",
            "does not establish positive transfer-matrix assumptions for production data",
            "does not fit energies/overlaps, calibrate interval coverage, or quantify finite-Nt effects",
            "td=2 GEVP composite is a separate diagnostic and must not populate kernel-indexed q cells",
            "independent review and authorized raw cross-correlator measurements remain required",
        ],
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["denominators"] = {
        "selected_bare_operator_T_cells": {"planned": 4, "valid": 4, "failed": 0, "abstained": 0},
        "descriptive_APE_T_cells": {"planned": 2, "valid": 2, "failed": 0, "abstained": 0},
        "GEVP_composite_T_cells": {"planned": 2, "valid": 2, "failed": 0, "abstained": 0},
        "physical_measurements": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0},
        "certification_gate": summary["certification_gate"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# F-lane q-cell object mapping diagnostic

**Decision proposed in the task contract:** each kernel-indexed q cell uses that kernel's bare smearing basis operator. For `C_ij(R,t)=Σ_n Z_ni(R) Z*_nj(R) exp[-E_n(R)t]`, the fit field `overlap_ratio_upper[n]` is a one-sided upper bound on `|Z_nk/Z_0k|²`, and

```text
q_k(R,T) = Σ_(n>0) |Z_nk(R)/Z_0k(R)|² exp[-(E_n(R)-E_0(R)) T].
```

The joint fit uses all measured 3×3 `C_ij` entries, while the q cell maps to the row/column for its fixed bare operator: `identity → identity`, `ape_alpha0.4_n5 → APE`, `spatial_stout_rho0.10_n5 → stout`. APE remains descriptive. The single GEVP vector at `td={TD}` creates a different composite operator; it is retained for state identification and sensitivity checks, never copied into each kernel's q cell. This keeps the object aligned with the selected kernel-specific correlators. The GEVP framework explicitly constructs linear combinations of the input fields, so the composite and bare-basis overlaps are distinct objects ([Blossier et al., 2009](https://arxiv.org/abs/0902.1265)).

![Bare-basis and composite q values](q_bare_basis_vs_gevp.png)

The logarithmic y scale makes clear that the exact toy's composite q can be much larger than every bare-basis q. This is a method-mapping example only; it does not imply that either object is the production contamination bound. The kernel-indexed choice is a project-level proposal requiring independent physics review, and production must fit the basis overlaps from raw cross-correlators under the reviewed finite-`N_t` model.

Text flow: `raw 3×3 matrix per R → joint spectral fit → per-basis squared overlap ratios + gap bounds → map kernel ID to its bare operator → q envelope over fit variants and T={5,6}; GEVP composite remains separate → independent review and production receipts still required`.

Denominators (planned/valid/failed/abstained): selected bare operator×T cells `4/4/0/0`; descriptive APE cells `2/2/0/0`; separate GEVP composite×T cells `2/2/0/0`; physical measurement files `0/0/0/0`; certification gate `1/0/0/1`. `physical_score=null`, `GROUND_STATE_FORCE_UNCERTIFIED`, and `NOT_READY/UNCERTIFIED` remain.

`q_objects.csv`, `protocol.json`, and `summary.json` contain the exact toy values and source hashes. All scientific inputs are deterministic synthetic data; no sealed split or GPU was accessed.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "q_cell_semantics_diagnostic.py").write_bytes(Path(__file__).read_bytes())
    artifact_hashes = {
        path.name: _sha(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps({"schema": "latticearena-artifact-hashes-v1", "files": artifact_hashes},
                   indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
