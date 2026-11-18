"""Ask an LLM to propose a signed mixture of diffusion kernels and score it on development."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from agents.llm import chat
from core.candidates import validate_weights
from core.search import Budget, search
from core.io import fingerprint, write_json
from tasks.pion_projection.common import protocol
from tasks.pion_projection.experiment import late_metric
from core.reference_provider import make_provider, provide
from tasks.pion_projection.reference import reference_kind, verify_reference
from tasks.pion_projection.statistics import blocks, bootstrap_indices, fold, risk

PROMPT = """You are optimizing the sink kernel of a quenched pion two-point function.

K = sum_k u_k H^p_k / A_k,  H = (I + {alpha} * spatial_hopping) / (1 + 6*{alpha})
powers p_k = {powers}, A_k are the measured amplitudes of each channel.
u is the normalized coefficient vector: sum(u) = 1, sum(|u|) <= 8.
u = e_0 (the first power) is the local sink. Signed u are allowed.

Score (lower is better), with m0 fixed by the evaluator:
  q(t) = C(t) / (A f0(t)),  f0(t) = exp(-m0 t) + exp(-m0 ({T}-t))
  risk = mean over t in {score_window} of [ Var_boot(q(t)) + (q(t)-1)^2 ]
The departure term dominates, so a good kernel makes q(t) flat over that window.

Channels, as (power, amplitude A_k, risk of u=e_k):
{baselines}

Previous attempts, as (u, risk):
{history}

Return JSON only: {{"candidates": [[{n_powers} u values], [{n_powers} u values], [{n_powers} u values]]}}
Propose three structurally different vectors each round.
"""


def render_prompt(p, baselines, history):
    return PROMPT.format(alpha=p['alpha'], powers=list(p['powers']), T=p['geometry'][3],
                         score_window=list(p['score_window']), baselines=baselines,
                         history=json.dumps(history[-6:]), n_powers=len(p['powers']))


def run(output, model, rounds, allow_validation=True):
    output = Path(output)
    run_protocol = json.loads((output / 'development_provenance.json').read_text())['protocol']
    if run_protocol != protocol():
        print(f'WARNING: this run embeds its own protocol ({output}), not tasks/pion_projection/protocol.json')
    npz = np.load(output / 'development.npz')
    assert str(npz['protocol_hash']) == fingerprint(run_protocol)
    p = run_protocol
    data = blocks(fold(npz['correlators']), p['block_length'])
    frozen = json.loads((output / 'submission.json').read_text()) if (output / 'submission.json').exists() else {}
    spec = frozen.get('reference')
    kind = verify_reference(spec) if spec else reference_kind()
    provider = make_provider('pion_projection', kind,
                             protocol=p, window=p['reference_windows'][p['reference_nominal_window']])
    index = bootstrap_indices(len(data), p['bootstrap_samples'], p['seed'])
    provided = provide(provider, data, index=index)
    ref, masses = provided['fit'], provided['masses']
    basis = np.eye(len(p['powers']))
    scores = [risk(data, w, ref['mass'], masses, index, p) for w in basis]
    scales = np.array([s['amplitude'] for s in scores])
    baselines = [[p['powers'][i], round(float(scales[i]), 4), round(s['risk'], 6)] for i, s in enumerate(scores)]
    print('reference mass %.6f' % ref['mass'])
    print('basis (n, amplitude, risk)', baselines)

    worst = [s['risk'] for s in scores]
    history, rejections, llm_best = [], [], None
    best = {'risk': min(worst), 'weights': basis[int(np.argmin(worst))].tolist(), 'round': -1}
    for r in range(rounds):
        prompt = render_prompt(p, baselines, history)
        reply = chat(prompt, model)
        match = re.search(r'\{.*\}', reply, re.S)
        if match is None:
            history.append({'error': 'no json', 'reply': reply[:200]})
            continue
        cfg = json.loads(match.group(0))
        proposals = [(f'{r}.{i}', raw) for i, raw in enumerate(cfg.get('candidates') or [cfg['weights']])]

        def evaluate(raw):
            u = np.asarray(raw, dtype=float)
            return risk(data, u / scales, ref['mass'], masses, index, p)['risk']

        found = search(proposals, evaluate,
                       validate=lambda raw: validate_weights(np.asarray(raw, dtype=float),
                                                             len(p['powers']), p['optimizer_l1_bound']),
                       budget=Budget(len(proposals)))
        for item in found.rejected:
            candidate = item['candidate']
            safe = ([None if not isinstance(x, (int, float)) or not np.isfinite(x) else float(x)
                     for x in candidate] if isinstance(candidate, list) else str(candidate))
            history.append([safe, item['reason']])
            rejections.append({'round': item['source'], 'u': safe, 'reason': item['reason']})
        for item in found.accepted:
            u = np.asarray(item['candidate'], dtype=float)
            weights = u / scales
            history.append([u.tolist(), round(item['risk'], 6)])
            write_json(output / 'agent' / f"round{item['source'].replace('.', '_')}.json",
                       {'model': model, 'u': u.tolist(), 'weights': weights.tolist(),
                        'risk': item['risk'],
                        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                        'reply_sha256': hashlib.sha256(reply.encode()).hexdigest()})
            print(f"round {item['source']}: risk {item['risk']:.6g}  u {np.round(u, 3).tolist()}")
            if llm_best is None or item['risk'] < llm_best['risk']:
                llm_best = {'risk': item['risk'], 'u': u.tolist(), 'weights': weights.tolist(),
                            'round': item['source']}
            if item['risk'] < best['risk']:
                best = {'risk': item['risk'], 'weights': weights.tolist(), 'u': u.tolist(),
                        'round': item['source']}

    best['llm_best'] = llm_best
    write_json(output / 'agent' / 'rejections.json', rejections)
    write_json(output / 'agent' / 'search_trace.json',
               {'model': model, 'rounds': rounds, 'seed': int(p['seed']),
                'budget_per_round': 'len(proposals)', 'history': history,
                'rejections': rejections,
                'best': {'risk': best['risk'], 'round': best['round']},
                'validation_status': 'not_read' if not allow_validation else 'post_hoc_diagnostic'})
    if not allow_validation:
        best.update({'model': model, 'rounds': rounds, 'protocol_hash': fingerprint(p),
                     'validation_status': 'not_read', 'validation_risk': None,
                     'note': 'search only; validation was not opened'})
        write_json(output / 'agent' / 'best.json', best)
        print('BEST (search only)', json.dumps(best, indent=2))
        return best
    weights = np.asarray((llm_best or best)['weights'])
    provenance = json.loads((output / 'validation_provenance.json').read_text())
    if str(np.load(output / 'validation.npz')['protocol_hash']) != fingerprint(p):
        raise ValueError('validation protocol hash does not match the frozen protocol')
    if provenance.get('protocol') != p:
        raise ValueError('validation provenance protocol differs from the run protocol')
    validation = blocks(fold(np.load(output / 'validation.npz')['correlators']), p['block_length'])
    vindex = bootstrap_indices(len(validation), p['bootstrap_samples'], p['seed'] + 1)
    local = risk(validation, basis[0], ref['mass'], masses, vindex, p,
                 late_inverse=late_metric(data, basis[0], p))['risk']
    try:
        value = risk(validation, weights, ref['mass'], masses, vindex, p,
                     late_inverse=late_metric(data, weights, p))
        best.update({'validation_risk': value['risk'],
                     'validation_ratio_vs_local': (local + p['score_epsilon']) / (value['risk'] + p['score_epsilon']),
                     'validation_risk_of': (llm_best or best)['round']})
    except ValueError as exc:
        best['validation_invalid'] = str(exc)
    best.update({'model': model, 'rounds': rounds, 'protocol_hash': fingerprint(p),
                 'validation_status': 'post_hoc_diagnostic',
                 'note': 'post-hoc diagnostic, not an official submission'})
    write_json(output / 'agent' / 'best.json', best)
    print('BEST', json.dumps(best, indent=2))
