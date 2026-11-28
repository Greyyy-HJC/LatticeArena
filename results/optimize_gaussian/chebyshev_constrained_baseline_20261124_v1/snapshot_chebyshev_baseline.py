"""CPU contract draft for a positive degree-4 spectral smear baseline.

This is a proposal for D11 review, not a frozen production method. It uses a
Bernstein control-point parameterization of a degree-at-most-four polynomial;
the resulting polynomial is converted to Chebyshev coefficients for Clenshaw
application to the scaled covariant hopping operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .kernels import apply_kernel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/chebyshev_constrained_baseline_20261124_v1"
DEGREE = 4
LAPLACIAN_MAX = 12.0


def validate_control_points(control_points) -> np.ndarray:
    """Require monotone Bernstein controls 0<=b0<=...<=b4=1.

    The constraints imply that the scalar filter is a positive contraction
    and monotonically decreases with Laplacian eigenvalue. Values are rejected
    rather than clipped so a solver cannot silently cross the frozen domain.
    """
    raw = np.asarray(control_points)
    if (raw.shape != (DEGREE + 1,) or np.iscomplexobj(raw)
            or not np.issubdtype(raw.dtype, np.number)):
        raise ValueError("control_points must be five real numeric Bernstein coefficients")
    values = raw.astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("control_points must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("control_points must lie in [0,1]")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("control_points must be nondecreasing")
    if values[-1] != 1.0:
        raise ValueError("control_points[4] is fixed to 1 for low-mode normalization")
    return values


def evaluate_scaled_laplacian(control_points, lambda_fraction):
    """Evaluate P4(1-lambda/12) for lambda/12 in [0,1]."""
    b = validate_control_points(control_points)
    fraction = np.asarray(lambda_fraction, dtype=float)
    if not np.isfinite(fraction).all() or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("lambda_fraction must be finite and in [0,1]")
    y = 1.0 - fraction
    value = np.zeros_like(y, dtype=float)
    for i, coefficient in enumerate(b):
        value += coefficient * _comb4(i) * y**i * (1.0 - y)**(DEGREE - i)
    return value


def _comb4(k):
    # Small fixed-degree binomial coefficients avoid a SciPy runtime dependency.
    return (1, 4, 6, 4, 1)[k]


def bernstein_to_chebyshev(control_points) -> np.ndarray:
    """Return a[0:5] such that P4(x)=sum_k a[k] T_k(x), x in [-1,1]."""
    b = validate_control_points(control_points)
    nodes = np.cos(np.pi * (np.arange(DEGREE + 1) + 0.5) / (DEGREE + 1))
    laplacian_fraction = (1.0 - nodes) / 2.0
    values = evaluate_scaled_laplacian(b, laplacian_fraction)
    vandermonde = np.polynomial.chebyshev.chebvander(nodes, DEGREE)
    return np.linalg.solve(vandermonde, values)


def apply_filter(control_points, hopping_operator, field, *, hermitian_tolerance=1e-10,
                 spectral_tolerance=1e-9):
    """Apply the proposed P4 to a small dense Hermitian hopping matrix.

    The CPU matrix-free candidate below uses the spatial hopping kernel.
    This dense routine exists only for small contract tests and applies
    Clenshaw recurrence in x=I+H/6, where the proposed contract requires
    eig(H) subset [-12,0].
    """
    coefficients = bernstein_to_chebyshev(control_points)
    h = np.asarray(hopping_operator)
    psi = np.asarray(field)
    if (h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0
            or np.iscomplexobj(h) and not np.isfinite(h).all()
            or not np.isfinite(h).all()):
        raise ValueError("hopping_operator must be a finite square matrix")
    if not np.allclose(h, h.conj().T, rtol=0.0, atol=hermitian_tolerance):
        raise ValueError("hopping_operator must be Hermitian")
    if (psi.ndim not in (1, 2) or psi.shape[0] != h.shape[0]
            or not np.isfinite(psi).all()):
        raise ValueError("field must be finite with leading dimension matching H")
    eigenvalues = np.linalg.eigvalsh(h)
    if (eigenvalues[0] < -LAPLACIAN_MAX - spectral_tolerance
            or eigenvalues[-1] > spectral_tolerance):
        raise ValueError("spectrum of H is outside the declared [-12,0] interval")

    dtype = np.result_type(h.dtype, psi.dtype, np.float64)
    h = h.astype(dtype, copy=False)
    psi = psi.astype(dtype, copy=False)
    x_apply = lambda vector: vector + (h @ vector) / 6.0
    return _clenshaw_apply(coefficients, x_apply, psi)


def _clenshaw_apply(coefficients, x_apply, field):
    """Apply a fixed Chebyshev series through an operator callback."""
    psi = np.asarray(field)
    b_kplus1 = np.zeros_like(psi)
    b_kplus2 = np.zeros_like(psi)
    for k in range(DEGREE, 0, -1):
        b_k = 2.0 * x_apply(b_kplus1) - b_kplus2 + coefficients[k] * psi
        b_kplus2, b_kplus1 = b_kplus1, b_k
    return x_apply(b_kplus1) - b_kplus2 + coefficients[0] * psi


def _prepare_hopping_cpu(links, field, unitary_tolerance):
    if (type(links).__module__.startswith("cupy")
            or type(field).__module__.startswith("cupy")):
        raise TypeError("CPU hopping candidates accept NumPy arrays only")
    u = np.asarray(links)
    psi = np.asarray(field)
    if (psi.ndim != 5 or any(size < 1 for size in psi.shape[:3])
            or psi.shape[-2] != 3 or psi.shape[-1] < 1
            or not np.isfinite(psi).all()):
        raise ValueError("field must be finite with layout (z,y,x,3,nrhs)")
    expected_links = (3, *psi.shape[:3], 3, 3)
    if u.shape != expected_links or not np.isfinite(u).all():
        raise ValueError(f"links must be finite with shape {expected_links}")
    if (not np.isfinite(unitary_tolerance) or unitary_tolerance <= 0):
        raise ValueError("unitary_tolerance must be finite and positive")
    identity = np.eye(3)
    link_gram = u.conj().swapaxes(-1, -2) @ u
    if not np.allclose(link_gram, identity, rtol=0.0, atol=unitary_tolerance):
        raise ValueError("links must be unitary within unitary_tolerance")
    dtype = np.result_type(u.dtype, psi.dtype, np.float64)
    u = u.astype(dtype, copy=False)
    psi = psi.astype(dtype, copy=False)
    return u, psi


def _apply_hopping_filter_prepared(control_points, links, field):
    b = validate_control_points(control_points)
    u, psi = links, field
    unit_coefficients = np.ones((3, *psi.shape[:3]), dtype=float)

    # apply_kernel(I) = I + sum_i(T_i + T_i^dag); hence
    # X = I + H/6 = (apply_kernel(I) - I)/6 for the declared H.
    def x_apply(vector):
        return (apply_kernel(vector, u, unit_coefficients) - vector) / 6.0

    return _clenshaw_apply(bernstein_to_chebyshev(b), x_apply, psi)


def apply_hopping_filter_cpu(control_points, links, field, *,
                             unitary_tolerance=1e-10):
    """Apply P4(I+H/12) with matrix-free periodic CPU hopping matvecs.

    This is a review candidate, not production integration. It assumes three
    spatial directions, periodic shifts, unitary 3x3 links, and the field
    layout (z, y, x, color, rhs). No eigensolver is used; the declared
    spectrum [-12, 0] follows from the checked unitarity and periodic shifts.
    """
    b = validate_control_points(control_points)
    u, psi = _prepare_hopping_cpu(links, field, unitary_tolerance)
    return _apply_hopping_filter_prepared(b, u, psi)


def apply_gaussian_sandwich_cpu(control_points, links, field, *, alpha, steps,
                                unitary_tolerance=1e-10):
    """Apply the current unnormalized G^m P4(H) G^m CPU candidate.

    This mirrors ``train.py::gaussian`` on both sides, followed by the spectral
    filter. Each step is G=I+alpha*sum_i(T_i+T_i^dagger); no zero-mode or
    per-step normalization is inserted. It remains a CPU contract helper and
    does not select the production objective or normalization convention.
    """
    b = validate_control_points(control_points)
    u, psi = _prepare_hopping_cpu(links, field, unitary_tolerance)
    if (isinstance(steps, (bool, np.bool_))
            or not isinstance(steps, (int, np.integer)) or steps < 0):
        raise ValueError("steps must be a nonnegative integer")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("alpha must be a finite real scalar") from exc
    if not np.isfinite(alpha_value):
        raise ValueError("alpha must be a finite real scalar")

    coefficients = np.full((3, *psi.shape[:3]), alpha_value, dtype=float)
    base = psi
    for _ in range(steps):
        base = apply_kernel(base, u, coefficients)
    result = _apply_hopping_filter_prepared(b, u, base)
    for _ in range(steps):
        result = apply_kernel(result, u, coefficients)
    return result


def _contract() -> dict:
    return {
        "schema": "optimize-degree4-chebyshev-baseline-review-v1",
        "status": "REVIEW_DRAFT_NOT_FROZEN",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "degree_cap": 4,
        "hopping_operator": "H psi(x)=sum_i[U_i(x)psi(x+i)+U_i(x-i)^dagger psi(x-i)-2psi(x)] over three spatial directions",
        "assumptions": ["periodic spatial boundaries", "unitary SU(3) links", "color-position inner product"],
        "spectrum": "H is Hermitian with spectrum in [-12,0]; X=I+H/6 has spectrum in [-1,1]; y=I+H/12 has spectrum in [0,1]",
        "polynomial": "P4(y)=sum_i b_i C(4,i)y^i(1-y)^(4-i); equivalently sum_k a_k T_k(2y-1)",
        "coefficient_domain": "0<=b0<=b1<=b2<=b3<=b4=1; four free control parameters",
        "properties": ["0<=P4(y)<=1", "P4 is nondecreasing in y", "P4(1)=1", "positive Hermitian contraction on H"],
        "application": "five Clenshaw operator matvecs; CPU matrix-free P4 and unnormalized G^m P4(H) G^m candidates use the periodic hopping kernel without an eigensolver; GPU/train.main integration is not implemented",
        "comparison_contract_proposal": "Use the same outer Gaussian sandwich, objective/reference, fit/development IDs, precision, failure accounting, and paired evaluation as trainable-W; fit the four fixed-basis controls on fit IDs only; charge basis/application, solver calls, walltime, setup, and selection cost to this arm.",
        "not_frozen": ["D04 reference", "D06 r_star or Delta_star", "alpha/sample-size/family-wise procedure", "whether this positivity/monotonicity constraint is the accepted strong baseline", "outer Gaussian sandwich power and normalization", "calls and walltime budgets"],
    }


def _make_plot(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fraction = np.linspace(0.0, 1.0, 401)
    examples = {
        "identity: b=(1,1,1,1,1)": np.ones(5),
        "linear: b=(0,.25,.5,.75,1)": np.linspace(0.0, 1.0, 5),
        "strong cutoff: b=(0,0,0,0,1)": np.array([0., 0., 0., 0., 1.]),
    }
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for label, controls in examples.items():
        ax.plot(fraction, evaluate_scaled_laplacian(controls, fraction), lw=2,
                label=label)
    ax.set(xlim=(0, 1), ylim=(-0.03, 1.03),
           xlabel=r"normalized spatial Laplacian eigenvalue $\lambda/12$",
           ylabel=r"spectral response $P_4(1-\lambda/12)$",
           title="Admissible degree-4 spectral response examples (not fitted)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build(output: str | Path = DEFAULT_OUT) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    contract = _contract()
    source_path = Path(__file__).resolve()
    kernel_path = ROOT / "tasks/optimize_gaussian/kernels.py"
    source_bytes = source_path.read_bytes()
    kernel_bytes = kernel_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    contract["basis_reference_source"] = "tasks/optimize_gaussian/kernels.py"
    contract["basis_reference_source_sha256"] = hashlib.sha256(kernel_bytes).hexdigest()
    contract["contract_source_sha256"] = source_hash
    examples = {
        "identity": np.ones(5),
        "linear": np.linspace(0.0, 1.0, 5),
        "strong_cutoff": np.array([0., 0., 0., 0., 1.]),
    }
    contract["illustrative_control_points"] = {
        name: {"bernstein": b.tolist(),
               "chebyshev": bernstein_to_chebyshev(b).tolist(),
               "note": "algebraic example only; no fitting or physical data"}
        for name, b in examples.items()
    }
    contract["illustration_denominators"] = {
        "planned": len(examples), "valid": len(examples),
        "failed": 0, "abstained": 0,
        "unit": "deterministic response examples, not fit runs or physical samples",
    }
    (out / "snapshot_chebyshev_baseline.py").write_bytes(source_bytes)
    (out / "snapshot_kernels.py").write_bytes(kernel_bytes)
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    _make_plot(out / "spectral_response.png")
    lines = [
        "# Optimize D11: constrained degree-4 spectral baseline proposal",
        "",
        "**Status: `REVIEW_DRAFT_NOT_FROZEN`; CPU algebra only; no physics objective, reference, production input, GPU, or sealed split. `physical_score=null`.**",
        "",
        "## Scope",
        "",
        "- Goal: make the user's chosen trainable-W vs strong spectral baseline contrast executable at the operator-contract level.",
        "- Inputs: the current `tasks/optimize_gaussian/kernels.py` hopping kernel, interpreted under the explicitly stated periodic three-dimensional operator contract, and the user-selected F=2 comparison family.",
        "- Deliverable: a mathematically constrained degree-at-most-4 spectral family, dense and matrix-free CPU candidates including the unnormalized Gaussian sandwich, tests, and response plot.",
        "- Acceptance: explicit spectrum map, normalization, positivity/monotonicity, solver domain, and a hard boundary that leaves D04/D06 and production authorization unresolved.",
        "",
        "## Operator and spectrum map",
        "",
        "For unitary SU(3) spatial links and periodic spatial boundaries, use `H psi(x)=sum_i[U_i(x)psi(x+i)+U_i(x-i)^dagger psi(x-i)-2psi(x)]`. Its quadratic form obeys `0 <= -H <= 12 I`: `-H` is the sum over directions of link-covariant nearest-neighbor differences, and each difference norm is bounded by twice the two endpoint norms. Thus `H` is Hermitian with spectrum in `[-12,0]`; `X=I+H/6` lies in `[-1,1]`; and `y=I+H/12` lies in `[0,1]`.",
        "",
        "## Constrained degree-4 polynomial",
        "",
        "Define `P4(y)=sum_{i=0}^4 b_i * binom(4,i) * y^i * (1-y)^(4-i)`, with `0<=b0<=b1<=b2<=b3<=b4=1`. Since Bernstein basis functions are nonnegative and sum to one, `P4` is in `[0,1]`; since `b_i` are nondecreasing, `P4'(y)=4*sum_{i=0}^3 (b_{i+1}-b_i) B_{i,3}(y)>=0`. At zero Laplacian `y=1`, so `P4(1)=b4=1`. In `x=2y-1=I+H/6`, the same unique polynomial is `sum_{k=0}^4 a_k T_k(x)`. The Bernstein controls supply simple linear constraints; the degree-4 Chebyshev coefficients are obtained by an exact-degree basis conversion and applied by Clenshaw.",
        "",
        "![Admissible degree-4 spectral response examples](spectral_response.png)",
        "",
        "The identity, linear, and fourth-power cutoff curves are hand-chosen valid control vectors to show the allowed response shape. They are not fit results or selected physics kernels. A degree cap of four is fixed; it is not changed after development or blind inspection. The response plot's horizontal axis is the normalized Laplacian eigenvalue: zero is the low mode normalized to one, one is the maximum eigenvalue where suppression is strongest.",
        "",
        "The three illustrative curves are the only planned items in this report: planned/valid/failed/abstained = `3/3/0/0`. These counts do not represent optimizer fits, ensemble measurements, or physical samples.",
        "",
        "## Proposed comparison plumbing",
        "",
        "```text",
        "same frozen spatial links / source / outer Gaussian sandwich",
        "  ├─ learned arm: train W,v,c on fit IDs",
        "  └─ spectral arm: fit b0..b3 with 0<=b0<=...<=b3<=1; fix b4=1",
        "       both use the same frozen reference/objective, fit/development split, precision and paired test IDs",
        "       both report solver success/residual, all planned failures and abstentions",
        "       account separately for operator construction, each matvec, setup, fitting, inference, solve and total walltime",
        "       apply the same objective-call cap and a separate matched total walltime cap",
        "```",
        "",
        "The spectral arm has four free controls, while the current hidden-width-2 trainable-W prototype has seven W/v/c scalars. They are different function classes, so equal parameter count is not required; both contrasts use a common objective and report parameter count, range, and full cost. The matrix-free helper is checked against an explicitly assembled small periodic lattice operator, including the uniform Gaussian-step affine identity, Hermiticity, and dense-versus-matrix-free Gaussian-sandwich action. The sandwich helper preserves the raw zero-mode factor `(1+6 alpha)^(2m)`. The polynomial's positivity and monotonicity are mathematical guarantees under the stated H spectrum, not evidence of physical superiority. The Gaussian exponent and normalization remain to be frozen with the objective.",
        "",
        "This provides a concrete D11 candidate for review, not a certified strongest baseline. The current heat-kernel operator toy used unconstrained least-squares Chebyshev coefficients and cannot validate this constrained family. Do not reuse its performance numbers as baseline evidence. `train.main` remains fixed-W; this module is not integrated into production training. D04 reference, D06 minimum effect, alpha/sample size, family-wise method and both budgets remain unresolved. Keep the two user-selected contrasts F=2, `NOT_READY/UNCERTIFIED`, `physical_score=null`, and all sealed data closed.",
        "",
        "This CPU helper is not wired into `train.main` and has no GPU/CuPy path. The report includes snapshots of this helper and the periodic hopping kernel; both are SHA-256 listed in its manifest. Rebuild the deterministic response artifact into a fresh directory with `python3 -B -m tasks.optimize_gaussian.chebyshev_baseline --output NEW_DIRECTORY`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (out / "contract.json", out / "spectral_response.png", out / "README.md",
                     out / "snapshot_chebyshev_baseline.py", out / "snapshot_kernels.py")
    }
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({"output": str(args.output), "degree_cap": result["degree_cap"],
                      "status": result["status"]}))


if __name__ == "__main__":
    main()
