# LatticeArena

LatticeArena collects research problems in lattice gauge theory and turns them
into well-defined optimization tasks for AI agents. Each task isolates a physical
or computational objective, specifies what an agent may change, and evaluates
proposals through a fixed, reproducible measurement and scoring procedure.

The goal is to test frontier models' ability to do research: identify useful
structure, propose an optimization, and demonstrate an improvement over strong
existing methods. A new state-of-the-art solution to one of these problems could
support a research paper. Beating a local baseline alone does not establish a
literature-wide SOTA result or a publishable discovery; those claims require
appropriate comparisons, uncertainty estimates, and independent validation.

Each task should define:

- The scientific background, target observable, and fixed problem parameters.
- The agent's search space and submission interface.
- A trusted evaluation pipeline, data protocol, and computational budget.
- A score with interpretable components, reference methods, and failure rules.

## First task: pion sink projection

[Pion projection](doc/pion_projection.md) asks an agent to design the spatial
shape and weights of a smeared pion sink on a pure-gauge ensemble, with a point
source fixed at the spacetime origin. The objective combines high signal-to-noise
with small excited-state contamination. Gauge-covariant, gauge-dependent sink
designs are allowed; inversion, contraction, and scoring belong to the evaluator.

The initial dataset is the local, unfixed `16^3 x 16` ensemble. The
[executable pilot](tasks/pion_projection/README.md) implements measurement,
propagator caching, baseline search, frozen kernel submissions, and statistical
evaluation. The [first agent experiment](doc/pion_projection_results.md) evaluates
a signed multiscale sink on 15 configurations held out from 30 development
configurations. It improves the provisional loss point estimate, but reference
checks fail and the comparison remains statistically inconclusive: no official
scores or SOTA claims are issued.

See [environment setup and verification](doc/environment.md) for the project
`.venv`, local QUDA/MPI requirements, and the single-configuration smoke test.

## Archived implementation

The previous implementation, including all tasks, datasets, benchmarks, tests,
and documentation, is preserved on the `legacy` branch with its Git history.

To explore the archived implementation:

```bash
git switch legacy
```
