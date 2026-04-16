# PyQUDA Notes

This note records the PyQUDA details that matter for LatticeArena tasks so we
do not need to re-derive them from package internals each time.

Upstream project:

- [CLQCD/PyQUDA](https://github.com/CLQCD/PyQUDA)

## What we use (currently from `pion_2pt`)

The fixed workflow in `tasks/pion_2pt/scripts/measure.py` relies on the
following PyQUDA / `pyquda-utils` pieces:

- `pyquda_utils.core.init(...)`
  - initializes the runtime and resource cache
- `pyquda_utils.core.LatticeInfo(...)`
  - defines the global lattice size and boundary-condition metadata
- `pyquda_utils.core.getClover(...)`
  - builds the Clover Dirac operator used for inversions
- `pyquda_utils.io.readNERSCGauge(path)`
  - loads quenched gauge configurations in NERSC format
- `pyquda_utils.core.invert(...)`
  - the direct point-source solve path used by the `plain`-style baseline
- `pyquda_utils.source.source(...)`
  - builds fermion sources, including `colorvector`
- `pyquda_utils.core.invertPropagator(...)`
  - solves the Dirac equation for a prebuilt propagator source when a
    submission requests a non-point profile source

## Source types worth remembering

From the installed `pyquda-utils` helpers:

- `point`
  - source localized at one spacetime site
- `wall`
  - source filling one timeslice
- `volume`
  - source filling the full lattice
- `momentum`
  - wall or volume source with an externally supplied phase
- `colorvector`
  - arbitrary spatial profile supplied as a color-vector field

For `pion_2pt`, the important split is:

- `point`
  - mirrors `pion_disp.py` for the reference point-source workflow
- `colorvector`
  - lets a submission provide an arbitrary normalized 3D source profile while
    the fixed workflow keeps control of the inversion and contraction

## Momentum phases

`pyquda_utils.phase.MomentumPhase` provides:

- `getPhase(mom_mode, x0=[0, 0, 0, 0])`
- `getPhases(mom_mode_list, x0=[0, 0, 0, 0])`

The helper interprets momentum as lattice integer modes, not physical GeV.
That matches the current quenched local test ensemble, which has no scale
setting yet. For the present benchmark target we use `(3, 3, 3)`.

## Smearing-related takeaways

PyQUDA itself supports gauge and fermion smearing operations, but for the
current task integration the simplest robust interface is:

- let submissions describe source and sink designs explicitly
- map point sources onto `core.invert(..., "point", ...)`
- realize arbitrary source profiles through `colorvector` sources
- keep the fixed benchmark workflow in control of the solve and contraction

This gives us a clean path for:

- point-like baselines
- Gaussian or momentum-smeared profiles
- future source variants inspired by PyQUDA utilities

without forcing the submission API to expose low-level QUDA objects.

## Practical constraints

- Importing PyQUDA can trigger MPI/runtime setup immediately, so benchmark code
should keep PyQUDA imports inside functions when possible.
- The current task implementation assumes local smoke-test runs and does not
depend on a tracked public ensemble.
- `cupy` must be available alongside `pyquda` / `pyquda-utils` for the
contraction path used by `pion_2pt`.
- The current dataset contract assumes NERSC gauge files plus an
`ensemble.json` file that carries Clover and source-time metadata.

## Recommended mental model

For this task, think of the stack as:

1. submission chooses source design, sink design, and `gamma_matrix`
2. fixed workflow dispatches the source to either `point` or `colorvector`
3. PyQUDA performs the inversion
4. fixed workflow constructs the sink projection and contracts the solved propagator
5. benchmark metrics score signal-to-noise and excited-state suppression
