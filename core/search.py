"""One search path for optimizer and LLM proposals (todo F1, slice 5).

Proposals are (source, candidate) pairs. The adapter validates, evaluates under
a budget and records every rejection with its reason; it never silently drops a
candidate.
"""
import math
from dataclasses import dataclass, field


@dataclass
class Budget:
    max_evaluations: int
    used: int = 0

    def spend(self):
        if self.used >= self.max_evaluations:
            return False
        self.used += 1
        return True


@dataclass
class SearchResult:
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    evaluations: int = 0
    budget_exhausted: bool = False

    def best(self, key='risk'):
        if not self.accepted:
            return None
        return min(self.accepted, key=lambda item: item[key])


def search(proposals, evaluate, validate=None, budget=None, key='risk'):
    """Run proposals through validate -> evaluate -> record.

    proposals: iterable of (source, candidate)
    validate(candidate) -> (ok, reason)
    evaluate(candidate) -> float
    """
    result = SearchResult()
    for source, candidate in proposals:
        if budget is not None and not budget.spend():
            result.budget_exhausted = True
            break
        if validate is not None:
            ok, reason = validate(candidate)
            if not ok:
                result.rejected.append({'source': source, 'candidate': candidate, 'reason': reason})
                continue
        try:
            value = evaluate(candidate)
        except Exception as exc:  # noqa: BLE001 - evaluation failures are recorded, not hidden
            result.rejected.append({'source': source, 'candidate': candidate, 'reason': f'evaluation failed: {exc}'})
            continue
        if value is None or not math.isfinite(float(value)):
            result.rejected.append({'source': source, 'candidate': candidate,
                                    'reason': 'non-finite evaluation result'})
            continue
        result.evaluations += 1
        result.accepted.append({'source': source, 'candidate': candidate, key: float(value)})
    return result
