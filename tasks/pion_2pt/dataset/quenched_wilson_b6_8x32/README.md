## Quenched Wilson \(\beta = 6\) default local ensemble

This directory contains the default **local-only** quenched gauge ensemble for
`pion_2pt` development, script comparison, and smoke-test benchmarking.

- **Gauge action**: Wilson plaquette action
- **Coupling**: \(\beta = 6\)
- **Lattice volume**: \(8^3 \times 32\)
- **Temporal source positions**: \(t = 0\) and \(t = 16\)

### Contents

- `ensemble.json`: metadata consumed by the fixed measurement workflow
- `wilson_b6.*`: gauge configurations in NERSC / PyQUDA-readable format
- `.cache/`: optional QUDA/PyQUDA cache files created at runtime

### Notes

The concrete gauge files are intentionally treated as local data and are not
tracked by git. The lattice dimensions were verified from the NERSC headers in
the imported `S8T32` gauge files. If you want to benchmark against another
ensemble, pass `--dataset-path` explicitly.
