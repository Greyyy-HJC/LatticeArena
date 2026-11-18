"""Pre-registration schema for a new pion measurement (todo §7)."""
import json
from pathlib import Path

REQUIRED = ('schema', 'ensemble', 'geometry', 'sources', 'sink_powers', 'splits',
            'identity', 'checks', 'statistics', 'freeze', 'budget', 'stop_rules', 'status')


def validate_plan(plan):
    missing = [key for key in REQUIRED if key not in plan]
    if missing:
        raise ValueError(f'measurement plan is missing {missing}')
    if plan['schema'] != 'pion-measurement-plan-v1':
        raise ValueError(f'unsupported plan schema {plan["schema"]!r}')
    if not plan['sources'] or not plan['sink_powers']:
        raise ValueError('sources and sink_powers must be non-empty')
    for key in ('development', 'validation'):
        if key not in plan['splits']:
            raise ValueError(f'splits must declare {key}')
    for key in ('gauge_sha256', 'propagator_sha256', 'quda', 'device', 'precision',
                'trajectory_ids', 'generation'):
        if key not in plan['identity']:
            raise ValueError(f'identity must include {key}')
    if plan['status'] not in ('NOT_RUN', 'RUNNING', 'DONE'):
        raise ValueError(f'unsupported status {plan["status"]!r}')
    return plan


def load_plan(path):
    return validate_plan(json.loads(Path(path).read_text()))
