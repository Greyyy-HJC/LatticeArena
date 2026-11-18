"""Audit of the historical optimize_gaussian runs (todo P0.3).

Classification uses each run's manifest: v9 was trained with the certified
link-centred A1+ features, earlier runs with the legacy forward-anchored ones.
Writes to an explicit output path and refuses to overwrite an existing file.
"""
import argparse
import json
from pathlib import Path

ROOT = Path('results/optimize_gaussian')
CURRENT_FEATURES = 'a1plus-link-centred-v3'


def feature_status(name):
    manifest = ROOT / name / 'manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get('features') == CURRENT_FEATURES:
            return 'current_operator_diagnostic', CURRENT_FEATURES
        return 'historical', data.get('features', 'unknown')
    return 'historical', 'legacy'


def load(name):
    for filename in ('results.json', 'v3_results.json', 'v2_results.json', 'smoke_results.json'):
        path = ROOT / name / filename
        if path.exists():
            return json.loads(path.read_text())
    return None


def dev_val_of(r):
    if 'best' in r and 'dev_val' in r['best']:
        return r['best']['dev_val'], 'best.dev_val'
    if 'dev_val' in r:
        return r['dev_val'], 'dev_val'
    return None, 'missing'


def summarize(name, r):
    status, features = feature_status(name)
    if r is None:
        return {'version': name, 'status': 'missing', 'feature_version': features}
    dev_val, source = dev_val_of(r)
    if 'models' in r:  # v5
        best = r['best_mode']
        return {'version': name, 'status': status, 'feature_version': features,
                'dev_val': r['models'][best]['dev_val'], 'dev_val_source': f'models[{best}].dev_val',
                'test': r['models'][best]['test'], 'base_test': r['base']['test'],
                'ratio': r['base']['test'] / r['models'][best]['test'],
                'note': f'mode={best}'}
    if 'test_risk_learned' in r:
        return {'version': name, 'status': status, 'feature_version': features,
                'dev_val': dev_val, 'dev_val_source': source,
                'test': r['test_risk_learned'], 'base_test': r['test_risk_base'],
                'ratio': r.get('test_ratio', r['test_risk_base'] / r['test_risk_learned']),
                'base_steps': r.get('best', {}).get('base_steps', r.get('base_steps'))}
    if 'final_loss' in r:  # smoke runs: shape loss, not the full risk
        return {'version': name, 'status': 'historical-smoke', 'feature_version': features,
                'initial_loss': r['initial_loss'], 'final_loss': r['final_loss'],
                'note': 'shape-loss objective, not the full risk'}
    if 'grid' in r and 'base' in r:  # v4
        return {'version': name, 'status': status, 'feature_version': features,
                'dev_val': min(g['dev_val_risk'] for g in r['grid']), 'dev_val_source': 'grid min',
                'test': r['test_risk_learned'], 'base_test': r['test_risk_base'],
                'ratio': r['test_ratio']}
    return {'version': name, 'status': 'unknown', 'keys': sorted(r)[:8]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/optimize_gaussian/audit_v2/audit.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError(f'{out} exists; choose a new audit path instead of overwriting')
    rows = []
    for path in sorted(ROOT.iterdir()):
        if path.is_dir() and path.name.startswith(('v', 'smoke')):
            rows.append(summarize(path.name, load(path.name)))
    candidate = Path('results/pion_projection/s16t16_v1/candidates/f686a72d9c868716')
    compare = {}
    if (candidate / 'compare.json').exists():
        compare = json.loads((candidate / 'compare.json').read_text())
        compare = {k: v for k, v in compare.items() if k in ('point_risk', 'ratio_a1p_over_slsqp', 'ratio_95')}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'runs': rows, 'candidate_v8': compare,
                               'classification': 'manifest-based (features field); v9 is '
                                                 'current_operator_diagnostic, earlier runs historical',
                               'note': 'Legacy risks are diagnostics only; this audit did not execute a new GPU measurement (the v9 archive remains separate).'},
                              indent=2) + '\n')
    for row in rows:
        print(row.get('version'), row.get('status'), row.get('feature_version'), row.get('ratio'))


if __name__ == '__main__':
    main()
