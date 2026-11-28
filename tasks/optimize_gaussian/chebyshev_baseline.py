"""CPU candidate for the selected positive degree-4 spectral smear baseline.

The method defaults are recorded in ``research_contract_v1.json``; this module
does not certify production links, connect a physical objective, or close the
independent review and authorization gates. It uses a Bernstein control-point
parameterization and a fixed dyadic rational transform to Chebyshev form.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from core.numerics import as_finite_real, as_real_float_array

from .kernels import apply_kernel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/optimize_gaussian/chebyshev_constrained_baseline_20261125_v4"
DEGREE = 4
LAPLACIAN_MAX = 12.0

# Rows are Chebyshev coefficients a_k; columns are Bernstein controls b_i.
# This fixed rational map follows from y=(x+1)/2 and
# P(y)=sum_i b_i * C(4,i) * y**i * (1-y)**(4-i).
# The denominator is a power of two, so each entry is exactly representable
# in float64 and no node selection or numerical linear solve is involved.
_BERNSTEIN_TO_CHEBYSHEV = np.asarray([
    [35, 20, 18, 20, 35],
    [-56, -16, 0, 16, 56],
    [28, -16, -24, -16, 28],
    [-8, 16, 0, -16, 8],
    [1, -4, 6, -4, 1],
], dtype=np.float64) / 128.0


def validate_control_points(control_points) -> np.ndarray:
    """Require monotone Bernstein controls 0<=b0<=...<=b4=1.

    The constraints imply that the scalar filter is a positive contraction
    and monotonically decreases with Laplacian eigenvalue. Values are rejected
    rather than clipped so a solver cannot silently cross the frozen domain.
    """
    try:
        values = as_real_float_array(control_points, name="control_points")
    except ValueError as exc:
        raise ValueError("control_points must be five real numeric Bernstein coefficients") from exc
    if values.shape != (DEGREE + 1,):
        raise ValueError("control_points must be five real numeric Bernstein coefficients")
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
    fraction = as_real_float_array(lambda_fraction, name="lambda_fraction")
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
    """Apply the fixed rational degree-4 basis map in float64.

    The symbolic transform is exact; this implementation performs the final
    5-by-5 matrix-vector product in floating point. Its entries are dyadic
    rationals exactly representable by float64.
    """
    b = validate_control_points(control_points)
    return _BERNSTEIN_TO_CHEBYSHEV @ b


def apply_filter(control_points, hopping_operator, field, *, hermitian_tolerance=1e-10,
                 spectral_tolerance=1e-9):
    """Apply the proposed P4 to a small dense Hermitian hopping matrix.

    The CPU matrix-free candidate below uses the spatial hopping kernel.
    This dense routine exists only for small contract tests and applies
    Clenshaw recurrence in x=I+H/6, where the proposed contract requires
    eig(H) subset [-12,0].
    """
    coefficients = bernstein_to_chebyshev(control_points)
    hermitian_tolerance = as_finite_real(
        hermitian_tolerance, name="hermitian_tolerance",
    )
    spectral_tolerance = as_finite_real(
        spectral_tolerance, name="spectral_tolerance",
    )
    if hermitian_tolerance <= 0 or spectral_tolerance < 0:
        raise ValueError("Hermitian tolerance must be positive and spectral tolerance nonnegative")
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
    unitary_tolerance = as_finite_real(unitary_tolerance, name="unitary_tolerance")
    if unitary_tolerance <= 0:
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
    alpha_value = as_finite_real(alpha, name="alpha")

    coefficients = np.full((3, *psi.shape[:3]), alpha_value, dtype=float)
    base = psi
    for _ in range(steps):
        base = apply_kernel(base, u, coefficients)
    result = _apply_hopping_filter_prepared(b, u, base)
    for _ in range(steps):
        result = apply_kernel(result, u, coefficients)
    return result




class _ObjectiveCallBudgetExceeded(RuntimeError):
    pass


def fit_control_points_slsqp(objective, *, max_objective_calls, max_iterations,
                             initial_free_controls=None, ftol=1e-9):
    """Fit four Bernstein controls to a fixed scalar objective with SLSQP.

    The callback receives all five controls, including the fixed endpoint b4=1.
    The optimizer is constrained to 0<=b0<=b1<=b2<=b3<=1. Objective calls are
    hard-capped; exhaustion returns the best feasible point evaluated so far
    with a non-success status. The callback should be deterministic for fixed
    controls and fit IDs. This helper does not choose or certify the objective
    and is intended only as CPU plumbing for a later frozen D04.
    """
    from scipy.optimize import minimize

    if not callable(objective):
        raise TypeError("objective must be callable")
    for name, value in (("max_objective_calls", max_objective_calls),
                        ("max_iterations", max_iterations)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    ftol_value = as_finite_real(ftol, name="ftol")
    if ftol_value <= 0:
        raise ValueError("ftol must be finite and positive")

    if initial_free_controls is None:
        initial = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
    else:
        raw_initial = as_real_float_array(
            initial_free_controls, name="initial_free_controls",
        )
        if raw_initial.shape != (DEGREE,) or not np.isfinite(raw_initial).all():
            raise ValueError("initial_free_controls must be four finite real values")
        initial = raw_initial.copy()
    validate_control_points(np.concatenate((initial, [1.0])))

    objective_calls = 0
    iterations = 0
    best_controls = None
    best_value = None
    best_residual = None

    def constraint_residual(free_controls):
        controls = np.concatenate((np.asarray(free_controls, dtype=float), [1.0]))
        violations = np.maximum(0.0, -controls)
        upper_violations = np.maximum(0.0, controls - 1.0)
        monotonicity_violations = np.maximum(0.0, -np.diff(controls))
        return float(max(violations.max(), upper_violations.max(),
                         monotonicity_violations.max()))

    def wrapped_objective(free_controls):
        nonlocal objective_calls, best_controls, best_value, best_residual
        if objective_calls >= max_objective_calls:
            raise _ObjectiveCallBudgetExceeded
        controls = np.concatenate((np.asarray(free_controls, dtype=float), [1.0]))
        if controls.shape != (DEGREE + 1,) or not np.isfinite(controls).all():
            raise ValueError("SLSQP proposed nonfinite or malformed controls")
        objective_calls += 1
        raw_value = np.asarray(objective(controls.copy()))
        if (raw_value.ndim != 0 or np.iscomplexobj(raw_value)
                or not np.issubdtype(raw_value.dtype, np.number)
                or not np.isfinite(raw_value)):
            raise ValueError("objective must return one finite real scalar")
        value = float(raw_value)
        residual = constraint_residual(free_controls)
        if (residual == 0.0 and (best_value is None or value < best_value)):
            best_controls = controls.copy()
            best_value = value
            best_residual = residual
        return value

    def record_iteration(_free_controls):
        nonlocal iterations
        iterations += 1

    constraints = [{
        "type": "ineq",
        "fun": lambda free: np.diff(np.concatenate((np.asarray(free, dtype=float), [1.0]))),
    }]
    try:
        result = minimize(
            wrapped_objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * DEGREE,
            constraints=constraints,
            callback=record_iteration,
            options={"maxiter": int(max_iterations), "ftol": ftol_value, "disp": False},
        )
    except _ObjectiveCallBudgetExceeded:
        return {
            "status": "OBJECTIVE_CALL_BUDGET_EXHAUSTED",
            "solver_success": False,
            "message": "objective-call budget exhausted; best feasible evaluated point retained",
            "control_points": None if best_controls is None else best_controls.tolist(),
            "objective": best_value,
            "objective_calls": objective_calls,
            "iterations": iterations,
            "feasibility_residual": best_residual,
        }

    final_x = np.asarray(result.x, dtype=float)
    final_residual = (constraint_residual(final_x)
                      if final_x.shape == (DEGREE,) and np.isfinite(final_x).all()
                      else float("inf"))
    solver_success = bool(result.success and final_residual == 0.0)
    status = "SLSQP_SUCCESS" if solver_success else "SLSQP_NON_SUCCESS"
    return {
        "status": status,
        "solver_success": solver_success,
        "message": str(result.message),
        "control_points": None if best_controls is None else best_controls.tolist(),
        "objective": best_value,
        "objective_calls": objective_calls,
        "iterations": iterations,
        "feasibility_residual": best_residual,
        "final_constraint_residual": final_residual,
    }

def _contract() -> dict:
    research_path = ROOT / "tasks/optimize_gaussian/research_contract_v1.json"
    research = json.loads(research_path.read_text())
    return {
        "schema": "optimize-degree4-chebyshev-baseline-review-v2",
        "status": "METHOD_DEFAULTS_SELECTED_EVIDENCE_NOT_READY",
        "physical_score": None,
        "certified": False,
        "data_access": "none",
        "selected_method_defaults": {
            key: research[key]
            for key in ("D04_reference", "D06_effect_threshold", "D11_strong_spectral_operator")
        },
        "selected_fair_budget_defaults": research["fair_budget"],
        "research_contract_source": "tasks/optimize_gaussian/research_contract_v1.json",
        "research_contract_source_sha256": hashlib.sha256(research_path.read_bytes()).hexdigest(),
        "degree_cap": 4,
        "hopping_operator": "H psi(x)=sum_i[U_i(x)psi(x+i)+U_i(x-i)^dagger psi(x-i)-2psi(x)] over three spatial directions",
        "assumptions": ["periodic spatial boundaries", "unitary SU(3) links", "color-position inner product"],
        "spectrum": "H is Hermitian with spectrum in [-12,0]; X=I+H/6 has spectrum in [-1,1]; y=I+H/12 has spectrum in [0,1]",
        "polynomial": "P4(y)=sum_i b_i C(4,i)y^i(1-y)^(4-i); equivalently sum_k a_k T_k(2y-1)",
        "coefficient_domain": "0<=b0<=b1<=b2<=b3<=b4=1; four free control parameters",
        "properties": ["0<=P4(y)<=1", "P4 is nondecreasing in y", "P4(1)=1", "positive Hermitian contraction on H"],
        "application": "five Clenshaw operator matvecs; CPU matrix-free P4 and unnormalized G^m P4(H) G^m candidates use the periodic hopping kernel without an eigensolver; GPU/train.main integration is not implemented",
        "coefficient_fit_helper": "fit_control_points_slsqp accepts a fixed scalar objective callback, constrains four free Bernstein controls with b4=1, enforces a hard objective-call cap, and returns solver status and best feasible controls; it has no D04 objective wired in",
        "coefficient_fit_denominators": {"planned": 0, "valid": 0, "failed": 0, "abstained": 0, "unit": "no D04 callback supplied"},
        "comparison_contract": "Use the selected common outer G^2 P4(H) G^2 sandwich, candidate-blind D04 risk/reference, common fit IDs and precision, full failure accounting, and paired evaluation; fit the four Bernstein controls on fit IDs only; charge basis/application, all solver calls, walltime, setup and selection cost to this arm.",
        "open_gates": [
            "independent E-chain D04 reference data and receipt",
            "fit/evaluation ID and hash receipts",
            "empirical paired-risk and full-cost receipts for the two selected comparisons",
            "independent production-link/operator review",
            "data-owner and GPU/QUDA authorization for production runs",
        ],
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
    numerics_path = ROOT / "core/numerics.py"
    source_bytes = source_path.read_bytes()
    kernel_bytes = kernel_path.read_bytes()
    numerics_bytes = numerics_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    contract["basis_reference_source"] = "tasks/optimize_gaussian/kernels.py"
    contract["basis_reference_source_sha256"] = hashlib.sha256(kernel_bytes).hexdigest()
    contract["contract_source_sha256"] = source_hash
    contract["numeric_validation_source"] = "core/numerics.py"
    contract["numeric_validation_source_sha256"] = hashlib.sha256(numerics_bytes).hexdigest()
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
    (out / "snapshot_core_numerics.py").write_bytes(numerics_bytes)
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    _make_plot(out / "spectral_response.png")
    lines = [
        "# Optimize D11: selected constrained degree-4 spectral baseline",
        "",
        "**Status: `METHOD_DEFAULTS_SELECTED_EVIDENCE_NOT_READY`; CPU algebra only; production receipts/review/authorization are open. `physical_score=null`; sealed data remain closed.**",
        "",
        "## Scope",
        "",
        "- Goal: make the user's chosen trainable-W vs strong spectral baseline contrast executable at the operator-contract level.",
        "- Inputs: the current `tasks/optimize_gaussian/kernels.py` hopping kernel, interpreted under the explicitly stated periodic three-dimensional operator contract, and the user-selected F=2 comparison family.",
        "- Deliverable: a mathematically constrained degree-at-most-4 spectral family, dense and matrix-free CPU candidates including the unnormalized Gaussian sandwich, a bounded-call SLSQP coefficient-fitting helper, tests, and response plot.",
        "- Acceptance: D04/D06/D11 method defaults are recorded; this report checks the CPU algebra and leaves independent measurements, production review, and authorization open.",
        "",
        "## Operator and spectrum map",
        "",
        "For unitary 3x3 links and periodic spatial boundaries, use `H psi(x)=sum_i[U_i(x)psi(x+i)+U_i(x-i)^dagger psi(x-i)-2psi(x)]`. U(3) unitarity is sufficient for this mathematical bound; physical QCD production still requires SU(3) link provenance, which the CPU helper does not check. The quadratic form obeys `0 <= -H <= 12 I`: `-H` is the sum over directions of link-covariant nearest-neighbor differences, and each difference norm is bounded by twice the two endpoint norms. Thus `H` is Hermitian with spectrum in `[-12,0]`; `X=I+H/6` lies in `[-1,1]`; and `y=I+H/12` lies in `[0,1]`.",
        "",
        "## Constrained degree-4 polynomial",
        "",
        "Define `P4(y)=sum_{i=0}^4 b_i * binom(4,i) * y^i * (1-y)^(4-i)`, with `0<=b0<=b1<=b2<=b3<=b4=1`. Since Bernstein basis functions are nonnegative and sum to one, `P4` is in `[0,1]`; since `b_i` are nondecreasing, `P4'(y)=4*sum_{i=0}^3 (b_{i+1}-b_i) B_{i,3}(y)>=0`. At the upper spectral endpoint `y=1`, `P4(1)=b4=1`; this does not assert that every gauge field has a zero eigenmode. In `x=2y-1=I+H/6`, the same unique polynomial is `sum_{k=0}^4 a_k T_k(x)`. The exact symbolic basis transform has dyadic rational entries; its final matrix-vector product is evaluated in float64 and applied by Clenshaw.",
        "",
        "![Admissible degree-4 spectral response examples](spectral_response.png)",
        "",
        "The identity, linear, and fourth-power cutoff curves are hand-chosen valid control vectors to show the allowed response shape. They are not fit results or selected physics kernels. A degree cap of four is fixed; it is not changed after development or blind inspection. The response plot's horizontal axis is the normalized Laplacian spectral coordinate: endpoint zero maps to y=1 and response one, but a general gauge field need not have an H=0 eigenmode; endpoint one is the maximum eigenvalue where suppression is strongest.",
        "",
        "The three illustrative curves are the only planned items in this report: planned/valid/failed/abstained = `3/3/0/0`. These counts do not represent optimizer fits, ensemble measurements, or physical samples.",
        "",
        "## Bounded-call coefficient fitting helper",
        "",
        "fit_control_points_slsqp accepts a fixed deterministic scalar objective callback and optimizes only b0..b3 with box and monotonicity constraints while fixing b4=1. It counts objective evaluations and stops before exceeding the caller-supplied cap, returning the best feasible point evaluated with OBJECTIVE_CALL_BUDGET_EXHAUSTED. Nonfinite or nonscalar objectives fail closed. No reference or physics objective is supplied or evaluated, so fitted-run denominator = 0/0/0/0.",
        "",
        "Text flow: fixed objective callback and fit IDs -> bounded SLSQP over four controls -> objective-call cap, status and feasibility residual -> retain fit failure or abstention -> later paired confirmation only after D04/D06 approval.",
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
        "The spectral arm has four free controls, while the current hidden-width-2 trainable-W prototype has seven W/v/c scalars. They are different function classes, so equal parameter count is not required; both contrasts use a common objective and report parameter count, range, and full cost. The matrix-free helper is checked against an explicitly assembled small periodic lattice operator, including the uniform Gaussian-step affine identity, Hermiticity, and dense-versus-matrix-free Gaussian-sandwich action. Since `G=1.6I+0.1H`, `G` and `P4(H)` commute and the unnormalized sandwich has eigenresponse `(1.6+0.1lambda)^4 P4(1+lambda/12)`. Its `6.5536=1.6^4` value applies at `lambda=0` only if `H` actually has a zero eigenmode; it is not a general gauge-field normalization. The polynomial's positivity and monotonicity follow from its Bernstein constraints under the stated H spectrum; this is not evidence of physical superiority or a rounding-level PSD certificate for production links.",
        "",
        "This implements the selected D11 method defaults, not a certified strongest production baseline. The current heat-kernel operator toy used unconstrained least-squares Chebyshev coefficients and cannot validate this constrained family. Do not reuse its performance numbers as baseline evidence. `train.main` remains fixed-W; this module is not integrated into production training. D04 E-chain measurements/receipts, D06 empirical risk and cost receipts, independent D11 production-link review, and data/GPU authorization remain open. Keep both selected F=2 contrasts, `NOT_READY/UNCERTIFIED`, `physical_score=null`, and all sealed data closed.",
        "",
        "This CPU helper is not wired into `train.main` and has no GPU/CuPy path. The report includes snapshots of this helper, the periodic hopping kernel and the shared numeric validator; all are SHA-256 listed in its manifest. Rebuild the deterministic response artifact into a fresh directory with `python3 -B -m tasks.optimize_gaussian.chebyshev_baseline --output NEW_DIRECTORY`.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (out / "contract.json", out / "spectral_response.png", out / "README.md",
                     out / "snapshot_chebyshev_baseline.py", out / "snapshot_kernels.py",
                     out / "snapshot_core_numerics.py")
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
