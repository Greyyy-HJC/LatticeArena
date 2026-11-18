"""Shared run-identity verification for joint/evaluate/candidate (A1)."""
import json
from pathlib import Path

from .common import digest, fingerprint


def verify_run_identity(output, models=None):
    """Return (frozen_kind, reasons); reasons empty means the identity is valid."""
    output = Path(output)
    p = json.loads((output / 'development_provenance.json').read_text())['protocol']
    sub = json.loads((output / 'submission.json').read_text())
    report_path = output / 'validation_results.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    reasons = []
    kind = sub.get('reference', {}).get('kind')
    if kind is None:
        return None, ['submission has no frozen reference kind']
    wanted = tuple(models) if models is not None else (kind,)
    if wanted != (kind,):
        reasons.append(f'requested models {wanted} differ from the frozen kind {kind!r}')
    if sub.get('protocol_hash') != fingerprint(p):
        reasons.append('submission protocol hash does not match the run protocol')
    if sub.get('development_sha256') != digest(output / 'development.npz'):
        reasons.append('development data hash does not match the submission')
    if report is not None:
        if report.get('reference_kind') not in (None, kind):
            reasons.append('validation report reference kind differs from the frozen submission')
        if report.get('protocol', {}).get('version') != p['version']:
            reasons.append('validation report protocol differs from the run protocol')
    return kind, reasons
