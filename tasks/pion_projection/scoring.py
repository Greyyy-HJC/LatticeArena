"""Conservative improvement certificate with an enforced contract (todo P0.4/P0.5).

The function works on per-model bootstrap risk draws. It is a *conditional*
summariser: it checks the caller's spec and draw shapes, but it cannot certify
a reference by itself. A separate gate must establish that the accepted models
and the joint draws are trustworthy.
"""
import numpy as np

from core.comparator import check_draws, log_lower_bound

SCHEMA = 'pion-score-v1'
REQUIRED = ('schema', 'version', 'estimand', 'baseline_id', 'accepted_models',
            'candidate_ids', 'analysis_version', 'alpha', 'comparison_family',
            'min_baseline_risk', 'invalid_tolerance', 'min_improvement')


def validate_spec(spec):
    missing = [key for key in REQUIRED if key not in spec]
    if missing:
        raise ValueError(f'score spec is missing {missing}')
    if spec['schema'] != SCHEMA:
        raise ValueError(f'unsupported score schema {spec["schema"]!r}')
    if not 0 < spec['alpha'] < 1:
        raise ValueError('alpha must be in (0, 1)')
    if not spec['accepted_models']:
        raise ValueError('accepted_models must be non-empty; an empty model set must abstain')
    if not spec['candidate_ids']:
        raise ValueError('candidate_ids must be non-empty')
    if len(set(spec['accepted_models'])) != len(spec['accepted_models']):
        raise ValueError('accepted_models contains duplicates')
    family = spec['comparison_family']
    if (not isinstance(family, dict) or family.get('candidates') != len(spec['candidate_ids'])
            or family.get('models') != len(spec['accepted_models'])):
        raise ValueError('comparison_family must match the declared candidate and model counts')
    if spec['min_baseline_risk'] < 0 or float(spec.get('risk_floor', 0.0)) < 0:
        raise ValueError('risk_floor and min_baseline_risk must be non-negative')
    if not 0 <= spec['invalid_tolerance'] < 1:
        raise ValueError('invalid_tolerance must be in [0, 1)')
    if spec['min_improvement'] <= 0:
        raise ValueError('min_improvement must be positive')
    return spec


def _check_draws(risks, spec, draw_ids):
    models = list(risks)
    if set(models) != set(spec['accepted_models']):
        raise ValueError(f'risk models {sorted(models)} are not the accepted set {sorted(spec["accepted_models"])}')
    baseline = spec['baseline_id']
    for model in models:
        if baseline not in risks[model]:
            raise ValueError(f'baseline {baseline!r} missing for model {model!r}')
        for name, values in risks[model].items():
            array = np.asarray(values)
            if array.ndim != 1:
                raise ValueError(f'{model}/{name}: risk draws must be one-dimensional, got shape {array.shape}')
    candidates = [name for name in risks[models[0]] if name != baseline]
    if set(candidates) != set(spec['candidate_ids']):
        raise ValueError(f'candidate ids {sorted(candidates)} are not the declared set {sorted(spec["candidate_ids"])}')
    flat = {f'{model}/{name}': np.asarray(values) for model in models for name, values in risks[model].items()}
    n_draws, _ = check_draws(flat, draw_ids, finite=False)
    return models, candidates, n_draws


def improvement_certificate(risks, spec, draw_ids=None, draw_ids_by_name=None):
    """Certify (or refuse) a simultaneous improvement over the baseline."""
    spec = validate_spec(spec)
    if spec.get('require_draw_ids') and draw_ids is None:
        raise ValueError('this certificate requires explicit draw_ids')
    if draw_ids_by_name is not None:
        if draw_ids is None:
            raise ValueError('draw_ids_by_name requires the shared draw_ids')
        for name, ids in draw_ids_by_name.items():
            if not np.array_equal(np.asarray(ids), np.asarray(draw_ids)):
                raise ValueError(f'{name}: draw ids are reordered relative to the shared draw axis')
    models, candidates, n_draws = _check_draws(risks, spec, draw_ids)
    if draw_ids is not None and 'draw_ids_hash' in spec:
        import hashlib
        digest = hashlib.sha256(np.asarray(draw_ids, dtype=np.int64).tobytes()).hexdigest()
        if digest != spec['draw_ids_hash']:
            raise ValueError('draw_ids do not match the frozen draw_ids_hash')
    baseline = spec['baseline_id']
    floor = float(spec.get('risk_floor', 0.0))
    level = spec['alpha'] / (len(candidates) * len(models))
    out = {'spec_version': spec['version'], 'analysis_version': spec['analysis_version'],
           'level': level, 'models': models, 'baseline_id': baseline,
           'candidates': {}, 'abstain': False, 'failure_reasons': []}
    for name in candidates:
        row = {'per_model_lower_bound': {}, 'per_model_median_improvement': {},
               'invalid_draws': 0, 'certified': False}
        bounds = []
        for model in models:
            raw_base = np.asarray(risks[model][baseline], dtype=float)
            raw_cand = np.asarray(risks[model][name], dtype=float)
            valid = np.isfinite(raw_base) & np.isfinite(raw_cand) & (raw_base > 0) & (raw_cand > 0)
            row['invalid_draws'] += int((~valid).sum())
            low, _ = log_lower_bound(raw_base, raw_cand, level, floor)
            if low is None:
                row['per_model_lower_bound'][model] = None
                bounds.append(-np.inf)
                continue
            row['per_model_lower_bound'][model] = float(low)
            row['per_model_median_improvement'][model] = float(np.median(np.log(raw_base[valid]) - np.log(raw_cand[valid])))
            bounds.append(float(low))
        row['lower_bound'] = float(min(bounds))
        row['improvement_lower_bound'] = float(np.exp(min(bounds))) if np.isfinite(min(bounds)) else 0.0
        out['candidates'][name] = row
    baseline_medians = []
    for m in models:
        values = np.asarray(risks[m][baseline], dtype=float)
        finite = values[np.isfinite(values)]
        baseline_medians.append(float(np.median(finite)) if finite.size else np.nan)
    if min(baseline_medians) < spec['min_baseline_risk']:
        out['abstain'] = True
        out['failure_reasons'].append('baseline_unresolved: median baseline risk below min_baseline_risk')
    invalid = sum(row['invalid_draws'] for row in out['candidates'].values())
    if invalid / (n_draws * len(candidates)) > spec['invalid_tolerance']:
        out['abstain'] = True
        out['failure_reasons'].append('invalid draw fraction above invalid_tolerance')
    threshold = float(spec['min_improvement'])
    for row in out['candidates'].values():
        if not out['abstain'] and np.isfinite(row['lower_bound']) and row['lower_bound'] > np.log(threshold):
            row['certified'] = True
    out['status'] = ('abstain' if out['abstain'] else
                     'certified' if any(r['certified'] for r in out['candidates'].values()) else 'not_certified')
    return out
