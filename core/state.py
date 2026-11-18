"""Explicit run state separate from the score (todo F3).

The score can be None; validation observation is permanent. Transitions follow
a small table so a validated claim cannot silently regress or skip a stage.
"""
STATES = ('operator', 'reference', 'numerical', 'statistical', 'comparison')
STATUS = ('not_run', 'draft', 'validated', 'failed')
ALLOWED = {('not_run', 'draft'), ('not_run', 'failed'), ('draft', 'validated'),
           ('draft', 'failed'), ('validated', 'failed')}


class RunState:
    def __init__(self):
        self.status = {name: 'not_run' for name in STATES}
        self.evidence = {name: [] for name in STATES}
        self.evidence_hash = {name: None for name in STATES}
        self.validation_observed = False
        self.score = None

    def set(self, name, status, evidence=None, evidence_hash=None):
        if name not in STATES:
            raise ValueError(f'unknown state {name!r}')
        if status not in STATUS:
            raise ValueError(f'unknown status {status!r}')
        if (self.status[name], status) not in ALLOWED:
            raise ValueError(f'illegal transition {name}: {self.status[name]} -> {status}')
        if status == 'validated' and not evidence_hash:
            raise ValueError(f'{name}: validated requires a bound evidence hash, not a label')
        if evidence_hash is not None and (not isinstance(evidence_hash, str) or len(evidence_hash) < 8):
            raise ValueError('evidence_hash must be a hash string')
        self.status[name] = status
        if evidence:
            self.evidence[name].append(evidence)
        if evidence_hash:
            self.evidence_hash[name] = evidence_hash
        return self

    def physical_ready(self):
        return all(self.status[name] == 'validated' and self.evidence_hash[name]
                   for name in STATES)

    def observe_validation(self, reference='unknown'):
        """Permanent record; a second observation does not reset the leak flag."""
        self.validation_observed = True
        self.evidence['comparison'].append(f'validation_observed:{reference}')
        return self

    def score_or_null(self):
        """A score requires all five hash-bound states to be validated."""
        return self.score if self.physical_ready() else None

    def summary(self):
        return {'status': dict(self.status), 'validation_observed': self.validation_observed,
                'score': self.score_or_null(), 'evidence': self.evidence,
                'evidence_hash': dict(self.evidence_hash), 'physical_ready': self.physical_ready()}
