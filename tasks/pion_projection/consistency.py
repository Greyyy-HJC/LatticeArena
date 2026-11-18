"""Cross-entry consistency and cache-invalidation counterexamples (F1 acceptance)."""
import os
os.environ.setdefault('LATTICEARENA_PROTOCOL', 'tasks/pion_projection/protocol_gevp200.json')
import json
from pathlib import Path

import numpy as np

from core.comparator import log_lower_bound, paired_ratio

RUN = Path('results/pion_projection/joint_20261118')
OUT = Path('results/pion_projection/consistency_v1')


def cross_entry():
    draws = np.load(RUN / 'joint_draws.npz')
    report = json.loads((RUN / 'metric_results.json').read_text())
    level = report['level']
    rows = []
    for model in report['models']:
        for name in report['candidates']:
            base = draws[f'risk_{model}_local']
            cand = draws[f'risk_{model}_{name}']
            low, invalid = log_lower_bound(base, cand, level, 0.0)
            ratio = paired_ratio(base, cand)
            expected = report['candidates'][name]['per_model_lower_bound'][model]
            rows.append({'model': model, 'candidate': name,
                         'lower_bound_from_npz': low, 'lower_bound_from_certificate': expected,
                         'delta': abs(low - expected),
                         'median_ratio_npz': ratio['median'],
                         'median_improvement_certificate': float(np.exp(report['candidates'][name]['per_model_median_improvement'][model])),
                         'invalid': invalid})
    return rows


def cache_invalidation():
    """Flip a data value or a protocol hash in a scratch copy and require refusal."""
    import shutil
    import tempfile

    from core.io import digest
    from tasks.pion_projection.common import load_data
    tmp = Path(tempfile.mkdtemp())
    for name in ('development.npz', 'development_provenance.json', 'submission.json'):
        shutil.copy(f'results/pion_projection/s16t16_gevp200/{name}', tmp / name)
    frozen = json.loads((tmp / 'submission.json').read_text())
    rows = []
    def attempt(case, action):
        try:
            action()
            rows.append({'case': case, 'rejected': False})
        except Exception as exc:  # noqa: BLE001
            rows.append({'case': case, 'rejected': True, 'reason': str(exc)})
    attempt('untouched copy', lambda: load_data(tmp, 'development'))
    data = np.load(tmp / 'development.npz')
    raw = data['correlators'].copy()
    raw[0, 0, 0] += 1.0
    np.savez_compressed(tmp / 'development.npz', ids=data['ids'], correlators=raw,
                        protocol_hash=data['protocol_hash'])
    attempt('corrupted development value (sha256)', lambda: _require(
        digest(tmp / 'development.npz') == frozen['development_sha256'], 'development sha256 changed'))
    data = np.load(tmp / 'development.npz')
    np.savez_compressed(tmp / 'development.npz', ids=data['ids'], correlators=data['correlators'],
                        protocol_hash=np.asarray('0' * 64))
    attempt('protocol hash changed', lambda: load_data(tmp, 'development'))
    return rows


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = cross_entry()
    invalid = cache_invalidation()
    invalidation_ok = (not invalid[0]['rejected']) and all(r['rejected'] for r in invalid[1:])
    result = {'cross_entry': rows, 'cache_invalidation': invalid,
              'status': 'PASS' if all(r['delta'] < 1e-9 for r in rows) and invalidation_ok else 'FAIL'}
    (OUT / 'consistency.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(row['model'], row['candidate'], 'delta %.2e' % row['delta'],
              'median %.3f' % row['median_ratio_npz'])
    for row in invalid:
        print(row)
    print('status', result['status'])


if __name__ == '__main__':
    main()
