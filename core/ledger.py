"""Cost ledger with explicit phases and environment (todo F3)."""
PHASES = ('setup', 'solve', 'residual', 'kernel', 'io', 'analysis')


class Ledger:
    def __init__(self, device='cpu', precision='complex128', parallel=1):
        self.device = device
        self.precision = precision
        self.parallel = parallel
        self.phases = {phase: 0.0 for phase in PHASES}
        self.cache = {'cold': 0.0, 'warm': 0.0}

    def add(self, phase, seconds, cache='cold'):
        if phase not in PHASES:
            raise ValueError(f'unknown phase {phase!r}')
        if cache not in ('cold', 'warm'):
            raise ValueError(f'unknown cache state {cache!r}')
        self.phases[phase] += float(seconds)
        self.cache[cache] += float(seconds)
        return self

    def merge(self, other):
        if (other.device, other.precision, other.parallel) != (self.device, self.precision, self.parallel):
            raise ValueError('refusing to merge ledgers from different devices/precision/parallelism')
        for phase in PHASES:
            self.phases[phase] += other.phases[phase]
        for key in self.cache:
            self.cache[key] += other.cache[key]
        return self

    def summary(self):
        total = sum(self.phases.values())
        return {'device': self.device, 'precision': self.precision, 'parallel': self.parallel,
                'phases': dict(self.phases), 'total_seconds': total,
                'cache_seconds': dict(self.cache),
                'unaccounted_seconds': max(total - sum(self.cache.values()), 0.0)}
