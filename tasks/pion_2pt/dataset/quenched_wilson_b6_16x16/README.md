## Quenched Wilson \(\beta = 6\) smoke-test ensemble

This directory contains a **local-only** quenched gauge ensemble for `pion_2pt`
development and smoke tests.

- **Gauge action**: Wilson plaquette action
- **Coupling**: \(\beta = 6\)
- **Lattice volume**: \(16^3 \times 16\)

### Contents

- `ensemble.json`: metadata consumed by the fixed measurement workflow
- `wilson_b6.*`: gauge configurations in NERSC / PyQUDA-readable format
- `.cache/`: optional QUDA/PyQUDA cache files created at runtime

### Notes

The concrete gauge files are intentionally treated as local data. The repository
may exclude large datasets from version control; if you do not have the gauge
files locally, provide your own dataset directory (with an `ensemble.json`) via
`--dataset-path`.
