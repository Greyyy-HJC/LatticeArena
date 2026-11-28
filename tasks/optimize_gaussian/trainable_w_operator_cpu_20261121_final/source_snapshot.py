"""CPU operator-loss bridge, with synthetic heat-kernel targets and no physics score.

Run into a new directory. All targets are defined independently of trainable W.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
from . import trainable_w as model
from .trainable_w_diagnostic import su3
from .features import directional_features
from .kernels import apply_kernel


def instance(rng, size=3, tau=.18):
    u = su3(rng, (3, size, size, size))
    n = 3 * size**3
    identity = np.eye(n).reshape(size, size, size, 3, n).astype(complex)
    h = apply_kernel(identity, u, np.ones((3, size, size, size))).reshape(n, n) - np.eye(n)
    eigenvalues, vectors = np.linalg.eigh(h)
    weights = np.exp(tau * eigenvalues)
    target = (vectors * weights) @ vectors.conj().T / weights.mean()
    # A coefficient multiplies both oriented blocks of one undirected edge.
    # L>=3 keeps all these block pairs distinct, so each tangent has norm^2=6.
    linear = np.empty((3, size, size, size))
    for d in range(3):
        for z,y,x in np.ndindex((size,size,size)):
            xyz = [x,y,z]; xyz[d] = (xyz[d]+1) % size
            j = ((xyz[2]*size + xyz[1])*size+xyz[0])*3
            i = ((z*size+y)*size+x)*3
            linear[d,z,y,x] = -2*np.vdot(u[d,z,y,x], target[i:i+3,j:j+3]).real
    return dict(links=u, z=directional_features(u), target=target, h=h,
                target_checks=dict(hermiticity_relative=float(np.linalg.norm(target-target.conj().T)/np.linalg.norm(target)),trace_minus_dim=float(np.trace(target).real-n),minimum_eigenvalue=float(weights.min()/weights.mean())),
                linear=linear, constant=float(np.linalg.norm(np.eye(n)-target)**2),
                denominator=float(np.linalg.norm(target)**2))


def operator_loss(parameters, data, hidden=2):
    W,v,c = model.unpack(parameters,hidden)
    z = np.stack([r['z'] for r in data])
    a,phi,centered = model._forward(z,W,v,c)
    linear=np.stack([r['linear'] for r in data])
    denominators=np.array([r['denominator'] for r in data])
    values=(np.array([r['constant'] for r in data]) +
            (2*linear*a+6*a*a).sum(axis=(1,2,3,4)))/denominators
    ga=(2*linear+12*a)/denominators[:,None,None,None,None]/len(data)
    projected=ga-ga.mean(axis=(2,3,4),keepdims=True)
    gw=np.einsum('ndzyxh,ndzyxj->hj',projected[...,None]*v*(1-phi**2),z)
    gv=np.einsum('ndzyx,ndzyxh->h',ga,centered)
    return float(values.mean()),np.r_[gw.ravel(),gv,ga.sum()]


def direct_loss(parameters, data):
    W,v,c=model.unpack(parameters,(len(parameters)-1)//3)
    values=[]
    for r in data:
        a=model.coefficient_field(r['z'][None],W,v,c)[0]
        n=len(r['h']); size=r['links'].shape[1]
        ident=np.eye(n).reshape(size,size,size,3,n).astype(complex)
        matrix=apply_kernel(ident,r['links'],a).reshape(n,n)
        values.append(float(np.linalg.norm(matrix-r['target'])**2/r['denominator']))
    return float(np.mean(values))


class Cap(Exception):
    pass


def solve(initial, fit, heldout, method, budget, joint):
    hidden=(len(initial)-1)//3
    active=np.arange(len(initial)) if joint else np.arange(2*hidden,len(initial))
    start=initial[active]; trace=[]; best=initial.copy(); bestval=float('inf')
    began=time.perf_counter()
    def objective(x):
        nonlocal best,bestval
        if len(trace)>=budget: raise Cap
        p=initial.copy();p[active]=x
        value,grad=operator_loss(p,fit,hidden)
        trace.append(dict(call=len(trace)+1,seconds=time.perf_counter()-began,loss=value,
                          gradient_norm=float(np.linalg.norm(grad[active]))))
        if value<bestval: bestval=value;best=p.copy()
        return value,grad[active]
    success=False; exhausted=False; status=None;message=''
    try:
        result=minimize(objective,start,method=method,jac=True,bounds=[(-3,3)]*len(active),
                        options=dict(maxiter=2000,ftol=1e-13,**({'gtol':1e-9} if method=='L-BFGS-B' else {})))
        success=bool(result.success);status=int(result.status);message=str(result.message)
    except Cap:
        exhausted=True;message='objective_call_cap'
    elapsed=time.perf_counter()-began
    value,grad=operator_loss(best,fit,hidden)
    pg=best[active]-np.clip(best[active]-grad[active],-3,3)
    return dict(method=method,joint_W=joint,budget=budget,calls=len(trace),budget_exhausted=exhausted,
                solver_success=success,status=status,message=message,optimization_seconds=elapsed,
                initial_parameters=initial.tolist(),parameters=best.tolist(),fit_loss=value,
                hidden=hidden,optimized_parameters=len(active),heldout_loss=operator_loss(best,heldout,hidden)[0],projected_gradient_norm=float(np.linalg.norm(pg)),
                W_update_norm=float(np.linalg.norm(best[:2*hidden]-initial[:2*hidden])),trace=trace)


def spectral(fit,heldout,degree):
    def arrays(rows):
        xx=[];yy=[]
        for r in rows:
            h=r['h']/6; b=[np.eye(len(h)),h]
            for k in range(2,degree+1): b.append(2*h@b[-1]-b[-2])
            scale=np.sqrt(r['denominator'])
            x=np.stack([a.ravel() for a in b[:degree+1]],axis=-1)/scale
            y=r['target'].ravel()/scale
            xx.append(np.r_[x.real,x.imag]); yy.append(np.r_[y.real,y.imag])
        return np.concatenate(xx),np.concatenate(yy)
    began=time.perf_counter();x,y=arrays(fit)
    coefficients=np.linalg.lstsq(x,y,rcond=None)[0];fitloss=float(np.mean((x@coefficients-y)**2)*len(y)/len(fit))
    elapsed=time.perf_counter()-began;xt,yt=arrays(heldout)
    return dict(degree=degree,coefficients=coefficients.tolist(),fit_loss=fitloss,
                heldout_loss=float(np.sum((xt@coefficients-yt)**2)/len(heldout)),
                setup_and_fit_seconds=elapsed,rank=int(np.linalg.matrix_rank(x)))


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    spec=dict(classification='diagnostic_only',physical_score=None,size=3,tau=.18,hidden=2,
              panel_seeds=[2026112201,2026112202,2026112203],fit_per_panel=3,heldout_per_panel=3,
              budgets=[40,160],bounds=[-3,3],methods=['SLSQP','L-BFGS-B'],spectral_degrees=[1,4],
              target='exp(tau H) / (trace exp(tau H) / dim), H=sum(U shift + adjoint)',
              loss='mean configuration relative squared Frobenius operator error; full identity basis, no noisy probes',
              selection='best fitting loss observed within cap; heldout never used for selection',
              source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ['trainable_w_operator_diagnostic.py','trainable_w.py','features.py','kernels.py']},
              versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
              limitations='3 independent synthetic panels only; no physics risk/production freeze; heat-kernel target favors spectral family; no multiplicity-adjusted inference')
    spec['arms']=['fixed W width2:3 fitted scalars','joint W width2:7 fitted scalars','fixed W width6:7 fitted scalars']
    (out/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    panels=[];inputs={}
    for seed in spec['panel_seeds']:
        rng=np.random.default_rng(seed)
        fit=[instance(rng) for _ in range(3)];heldout=[instance(rng) for _ in range(3)]
        initial=model.pack(rng.normal(size=(2,2))*.4,rng.normal(size=2)*.2,.05)
        value,grad=operator_loss(initial,fit)
        numeric=[]
        for i in range(7):
            delta=np.zeros(7);delta[i]=1e-6
            numeric.append((operator_loss(initial+delta,fit)[0]-operator_loss(initial-delta,fit)[0])/2e-6)
        check=dict(gradient_max_absolute_error=float(np.max(abs(grad-numeric))),
                   compressed_minus_direct_loss=float(value-direct_loss(initial,fit)),
                   analytic_gradient=grad.tolist(),finite_difference=numeric)
        assert check['gradient_max_absolute_error']<1e-8 and abs(check['compressed_minus_direct_loss'])<1e-12
        check['heldout_compressed_minus_direct_loss']=operator_loss(initial,heldout)[0]-direct_loss(initial,heldout)
        vh,gh=operator_loss(initial,heldout)
        nh=[]
        for i in range(7):
            delta=np.zeros(7);delta[i]=1e-6
            nh.append((operator_loss(initial+delta,heldout)[0]-operator_loss(initial-delta,heldout)[0])/2e-6)
        check['heldout_gradient_max_absolute_error']=float(np.max(abs(gh-nh)))
        assert abs(check['heldout_compressed_minus_direct_loss'])<1e-12 and check['heldout_gradient_max_absolute_error']<1e-8
        rows=[solve(initial,fit,heldout,m,b,j) for b in spec['budgets'] for j in [False,True] for m in spec['methods']]
        initial6=model.pack(np.r_[initial[:4].reshape(2,2),rng.normal(size=(4,2))*.4], np.r_[initial[4:6],rng.normal(size=4)*.2],initial[-1])
        rows += [solve(initial6,fit,heldout,m,b,False) for b in spec['budgets'] for m in spec['methods']]
        strong=[spectral(fit,heldout,k) for k in spec['spectral_degrees']]
        panels.append(dict(seed=seed,check=check,rows=rows,spectral=strong,target_checks=[r['target_checks'] for r in fit+heldout]))
        for population,ds in [('fit',fit),('heldout',heldout)]:
            inputs[f'{seed}_{population}_links']=np.stack([r['links'] for r in ds])
            inputs[f'{seed}_{population}_target']=np.stack([r['target'] for r in ds])
    np.savez_compressed(out/'synthetic_inputs.npz',**inputs)
    raw=dict(spec=spec,panels=panels,denominators=dict(planned_solver_runs=36,returned_solver_runs=36,
              independent_panels=3,fit_links=9,heldout_links=9,excluded=0,
              budget_exhausted=sum(r['budget_exhausted'] for p in panels for r in p['rows']),
              solver_non_success=sum(not r['solver_success'] for p in panels for r in p['rows'])))
    (out/'raw.json').write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for joint in [False,True]:
        for method in spec['methods']:
            label=('joint W' if joint else 'fixed W')+' / '+method
            for panel in panels:
                rows=[r for r in panel['rows'] if r['joint_W']==joint and r['method']==method and r['hidden']==2]
                axes[0].plot(spec['budgets'],[r['heldout_loss'] for r in rows],marker='o',alpha=.4,
                             label=label if panel is panels[0] else None)
    for method in spec['methods']:
        for panel in panels:
            a=next(r for r in panel['rows'] if r['budget']==160 and r['joint_W'] and r['method']==method)
            b=next(r for r in panel['rows'] if r['budget']==160 and r['hidden']==6 and r['method']==method)
            axes[2].plot([0,1],[b['heldout_loss'],a['heldout_loss']],marker='o',label=method if panel is panels[0] else None)
    axes[2].set_xticks([0,1],['fixed W width6','joint W width2'],rotation=15)
    axes[2].set(title='7 fitted scalars each; cap 160',ylabel='heldout squared operator error')
    for degree in spec['spectral_degrees']:
        mean=np.mean([p['spectral'][spec['spectral_degrees'].index(degree)]['heldout_loss'] for p in panels])
        axes[0].axhline(mean,ls='--',color='black' if degree==4 else 'gray',label=f'Chebyshev degree {degree} mean')
    for r in panels[0]['rows']:
        if r['budget']==160 and r['hidden']==2:
            axes[1].semilogy([t['call'] for t in r['trace']],np.minimum.accumulate([t['loss'] for t in r['trace']]),
                             label=('joint' if r['joint_W'] else 'fixed')+' / '+r['method'])
    axes[0].set(xlabel='objective+gradient call cap',ylabel='heldout relative squared operator error',yscale='log',title='Three independent panels; no confidence claim')
    axes[1].set(xlabel='objective+gradient calls',ylabel='best fitting relative squared error',title='First panel; all methods share initial W/v/c')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'operator_comparison.png',dpi=170);plt.close(fig)
    lines=['# Trainable W operator objective CPU bridge','','Status: **COMPLETED CPU DIAGNOSTIC; production and physical comparison NOT_RUN**. `physical_score=null`.','','The target is a trace-normalized gauge-covariant heat kernel G=exp(0.18 H)/(tr exp(0.18 H)/dim), independent of the learned parameterization. H is the actual periodic SU(3) nearest-neighbor hopping operator. Each link is fresh synthetic Haar-like SU(3), with shape (directions x/y/z, z/y/x, color/color), L=3. Full color/site identity probes give an exact Frobenius loss. No temporal/gamma/source/sink or real gauge ensemble is modeled. Trace normalization is a fixture convention, not physical smearing normalization.','','![Operator comparison](operator_comparison.png)','','## Design and what this resolves','','Each of 3 independent panels uses 3 fitting and 3 heldout links. All four solver/model arms share their initial W/v/c and box [-3,3], and SLSQP/L-BFGS-B receive the same analytic gradient and 40/160 objective+gradient call caps. Fixed W changes v/c only; joint W changes all 7 variables. The selected point minimizes observed fitting loss inside the cap. Final loss/KKT and heldout evaluations are diagnostic overhead outside that cap and are not used for selection. Time shown is optimizer time only, excluding link/target preparation and postchecks. This is a call-budget comparison, not matched total walltime or training amortization.','','Parameter-budget control: an additional fixed-W width6 arm fits 7 v/c scalars, matching the 7 optimized W/v/c scalars of joint width2. It shares the first 2 random W rows and v entries, but its four additional random features change the function class and initialization. This is an equal fitted-parameter count sensitivity, not a perfect identical-class ablation. The fixed width2 arm has 3 fitted scalars and isolates W freezing; do not treat its comparison as equal capacity. All caps apply per complete fit, without uncounted restarts.','','Chebyshev degree 1/4 coefficients are fit by unconstrained real least squares on the same fitting targets, then evaluated without refitting heldout. They are strong comparator families with independently adjustable identity term, different parameter domains, and direct linear algebra rather than matched calls. The target is a spectral function by definition, so these comparisons probe whether one-hop learned coefficients offer any reason to pursue this target; they cannot establish universal spectral superiority.','','|panel|cap|W|solver|calls|budget stop|solver success|fit loss|heldout loss|projected grad|W update|optimizer seconds|','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for p in panels:
        for r in p['rows']:
            lines.append(f"|{p['seed']}|{r['budget']}|{('joint' if r['joint_W'] else 'fixed')+str(r['hidden'])}|{r['method']}|{r['calls']}|{r['budget_exhausted']}|{r['solver_success']}|{r['fit_loss']:.6g}|{r['heldout_loss']:.6g}|{r['projected_gradient_norm']:.3g}|{r['W_update_norm']:.3g}|{r['optimization_seconds']:.4g}|")
    lines+=['','|panel|Chebyshev degree|fit loss|heldout loss|setup+fit seconds|','|---|---|---|---|---|']
    for p in panels:
        for r in p['spectral']:lines.append(f"|{p['seed']}|{r['degree']}|{r['fit_loss']:.6g}|{r['heldout_loss']:.6g}|{r['setup_and_fit_seconds']:.4g}|")
    lines+=['','## Diagnostics and interpretation','','The compressed quadratic in edge coefficients was checked against dense apply_kernel Frobenius loss at each panel initial point on both fitting and independent heldout links (absolute discrepancy below 1e-12). All 7 joint-W analytic parameter derivatives were checked on fitting and independent heldout links by centered finite differences (max error below 1e-8). Raw JSON retains values, gradients, all traces, selected parameters, solver non-success and cap stops; NPZ retains all actual links and targets. `protocol.json` was written before data generation and binds source hashes and seeds. All 36 planned solver runs returned; no samples were excluded. Budget stops are valid finite-budget observations and are not convergence claims. Projected-gradient residuals are continuous diagnostics with no production acceptance threshold.','','Observed result: at cap160, joint-W heldout squared error is 0.00600509–0.00614590 for SLSQP and 0.00600509–0.00614590 for L-BFGS-B (one L-BFGS-B panel is 0.00601241). Fixed-width2 differs only at roughly 1e-7 absolute or below. Equal-count fixed-width6 is slightly better in one panel and slightly worse in two; no stable learned-W advantage is established. Degree4 spectral heldout error is 3.34e-8–2.40e-7; degree1 is approximately 0.0060. Thus improving the one-hop parameterization/optimizer has very little headroom on this target, whereas changing operator range has a large effect. This is a structural negative diagnostic conditional on this spectral target.','','Target checks for all18 links retain Hermiticity, trace-minus-dimension and positive minimum eigenvalue in raw.json. Candidate K has trace dim exactly at L3 because hopping has no diagonal. Candidate Hermiticity follows from real paired-edge coefficients, but positivity is not imposed. These trace/spatial/boundary conventions match for this objective; production normalization is still open.','','This bridge establishes that trainable W can optimize a target defined at the operator-action level, beyond coefficient-label recovery. More calls to this same heat-kernel fixture cannot determine the physical learned-smear question. Read paired solver differences and family differences separately; three panels provide no calibrated population interval. Positivity, Gaussian sandwich, physical risk, independent-ensemble generalization, GPU time, production integration, and superiority to SLSQP remain unproven. Next production work still requires a justified physical reference/objective, frozen comparison/normalization and budgets, then an authorized GPU development run before sealed evaluation.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(raw['denominators']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    run(parser.parse_args().out)
