# D11 operator contract source audit

**Source and algebra review only. No gauge fields, propagators, sealed splits, GPU, or QUDA inputs were read. `physical_score=null`; this is not an independent physics sign-off.**

## Finding

The degree-4 Bernstein/Chebyshev operator contract is algebraically consistent with the current CPU hopping and matrix-free filter implementation under periodic spatial shifts and unitary 3x3 links. Seven source/algebra obligations pass, none contradicts the contract, and two production obligations abstain. The full nine-row ledger and SHA-256 bindings are in `audit_checks.csv` and `protocol.json`.

## Derivation and implementation mapping

For each spatial direction define `(T_i psi)(x)=U_i(x) psi(x+i)`. With periodic shifts and unitary links, `T_i` is unitary. Thus `H_i=T_i+T_i†-2I=-(T_i-I)†(T_i-I)`, so `-4I<=H_i<=0`; summing three directions gives `-12I<=H<=0`. The stored axes are `(z,y,x)`, while direction indices map `(x,y,z)` to storage axes `(2,1,0)`. `kernels.py` implements the forward edge and the shifted adjoint edge, and the coefficient field is real, so the operator is Hermitian.

The map `y=I+H/12` lies in `[0,1]`; `x=2y-I=I+H/6` lies in `[-1,1]` and is the coordinate used by the Clenshaw recurrence. Degree-4 Bernstein controls `0<=b0<=...<=b4=1` make `P4(y)` a positive contraction with `P4'(y)>=0` and `P4(1)=1`. This fixes the response at the upper spectral endpoint; a general gauge field need not have an `H=0` eigenmode.

Since `sum_i(T_i+T_i†)=H+6I`, the selected `G=I+0.10 sum_i(T_i+T_i†)=1.6I+0.1H`, with spectrum in `[0.4,1.6]`. `G` commutes with `P4(H)`. The common unnormalized `G² P4(H) G²` therefore has eigenresponse `(1.6+0.1 lambda)^4 P4(1+lambda/12)`, which is nonnegative and bounded above by `1.6^4=6.5536`. That value is the response at `lambda=0` only if zero belongs to the actual spectrum; it is not a per-configuration normalization guarantee.

## Review ledger

Denominator: planned/valid/failed/abstained = `9/7/0/2`. `valid` means the source and stated algebra agree under the declared assumptions; it does not mean a production measurement passed.

|check|status|evidence|
|---|---|---|
|spatial_layout_and_periodic_shift|valid|features.py declares storage axes (z,y,x), direction order x/y/z, and np.roll periodic shifts|
|paired_forward_adjoint_hop|valid|kernels.py pairs each real forward coefficient with its shifted adjoint coefficient|
|H_spectrum_under_unitary_links|valid|each unitary covariant shift T_i gives H_i=T_i+T_i†-2I with 0<=-H_i<=4I; three directions give 0<=-H<=12I|
|chebyshev_coordinate_matches_H|valid|the code applies x=I+H/6=2(I+H/12)-I, so x is in [-1,1] for the declared H spectrum|
|Bernstein_constraints_and_endpoint|valid|nonnegative Bernstein basis plus monotone controls gives 0<=P4<=1 and P4'(y)>=0; b4=1 gives P4(y=1)=1|
|Gaussian_affine_spectrum|valid|G=I+0.1 sum_i(T_i+T_i†)=1.6I+0.1H, hence spec(G) is contained in [0.4,1.6]|
|common_unnormalized_sandwich_and_conditional_factor|valid|the helper applies G^2 on each side without per-step normalization; G and P4(H) commute, giving response G(lambda)^4 P4(y); 6.5536 is conditional on lambda=0 being an eigenvalue|
|physical_SU3_link_receipt|abstained|the CPU helper checks U†U≈I but does not check det(U)=1; production SU(3) provenance or determinant-unitarity evidence is still absent|
|production_objective_and_RunSpec_integration|abstained|trainable_w.py identifies itself as an isolated CPU parameterization with no production physics objective; D04 data and production evaluator receipts remain absent|

## Remaining handoff

The matrix helper currently checks `U†U≈I`, which is sufficient for the spectral bound but does not establish `det(U)=1`; production QCD inputs need an independent SU(3) provenance or determinant-unitarity receipt. The production objective/evaluator and current D04 reference are also absent. Keep D11 `NOT_READY/UNCERTIFIED` until those source, data, and reviewer gates are independently satisfied. The analytic endpoint and raw sandwich factor have been corrected in the machine contract and candidate adapter.

## Sources

The audit binds the active `research_contract_v1.json`, `contract.py`, `candidate_contract.py`, `features.py`, `kernels.py`, `chebyshev_baseline.py`, and `trainable_w.py` by SHA-256 in `protocol.json`. The audit script itself is covered by `artifact_hashes.json`.
