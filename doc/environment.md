# Local PyQUDA environment

The project uses an independent `.venv` at the repository root. These instructions
describe the environment verified on 2026-11-09 on the current host. They assume
the local QUDA build, OpenMPI installation, reference checkout, and ensemble
paths below. Gauge data and binary dependencies are not distributed by this repo.
The environment now also includes pinned Matplotlib and pytest dependencies for
the [executable ensemble pilot](../tasks/pion_projection/README.md). Its broader
validation and results are recorded in [the experiment report](pion_projection_results.md).

## Activate an existing environment

Run from the repository root:

```bash
source /home/jinchen/env/pyquda.sh
source .venv/bin/activate
export PATH=/home/jinchen/software/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/home/jinchen/software/openmpi/lib:${LD_LIBRARY_PATH:-}
export CUDA_PATH=/usr
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
mkdir -p .cache/quda
```

The supplied `pyquda.sh` sets `QUDA_PATH=/home/jinchen/software/quda/build`.
It does not activate Python or select MPI. QUDA links to the OpenMPI installation
above; mpi4py must load that same MPI. On this host the CUDA toolkit is installed
under `/usr`, and CuPy 14 needs `CUDA_PATH=/usr` to find its headers. The driver's
reported CUDA 13.2 capability is not the toolkit version: the installed QUDA
library uses CUDA 12.0. On a different host, use its actual matching paths.

## Rebuild the environment

Use a uv-managed Python to obtain Python 3.12 and its development headers without
depending on the system Python's missing `ensurepip` or development package:

```bash
uv venv --python 3.12.13 --managed-python .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Apply the activation settings above. Build native PyQUDA against the local QUDA
headers/library, rather than assuming an unrelated binary wheel is compatible.
The reference source is `temp/PyQUDA` at commit
`138d8ad6e4d6e0805ecc7ec0a59a45febe0482ce` (`v0.10.54-1-g138d8ad`).
If that local checkout is unavailable, obtain the same revision from
[CLQCD/PyQUDA](https://github.com/CLQCD/PyQUDA/tree/138d8ad6e4d6e0805ecc7ec0a59a45febe0482ce).

For a fresh build directory, export the pinned source and fetch its exact
`pycparser` submodule. An ordinary `git archive` does not include submodules;
the reference checkout originally had this submodule uninitialized. All build
products and generated wrapper sources stay in `.cache`, leaving the reference
checkout unchanged.

```bash
mkdir -p .cache/build/pyquda .cache/wheels
git -C temp/PyQUDA archive 138d8ad6e4d6e0805ecc7ec0a59a45febe0482ce |
  tar -x -C .cache/build/pyquda
git clone --no-checkout https://github.com/eliben/pycparser.git \
  .cache/build/pyquda/pyquda_core/pycparser
git -C .cache/build/pyquda/pyquda_core/pycparser checkout --detach \
  5b60167602e7914139b3e277266ccdff5c24dbf2
python - <<'PY'
from pathlib import Path
# An archive lacks the Git metadata used to generate the utilities version.
Path('.cache/build/pyquda/pyquda_utils/_version.py').write_text(
    '__version__ = "0.10.54.post1"\n'
)
PY
GIT_CEILING_DIRECTORIES="$PWD/.cache/build" python -m pip wheel \
  --no-deps --no-build-isolation --wheel-dir .cache/wheels \
  ./.cache/build/pyquda/pyquda_core ./.cache/build/pyquda
uv pip install --python .venv/bin/python --no-deps \
  .cache/wheels/pyquda-*.whl .cache/wheels/pyquda_utils-*.whl
python -m pip check
```

The Git ceiling prevents the archive's version discovery from accidentally
using tags in the enclosing LatticeArena repository. This recipe assumes a
fresh build directory and wheel directory; do not mix wheels from other revisions
or Python versions. `requirements.txt` pins the ordinary runtime and build
dependencies; PyQUDA's two locally built wheels are installed separately.

## Verified versions and results

| Component | Verified value |
| --- | --- |
| Python | CPython 3.12.13, uv-managed |
| PyQUDA / PyQUDA-Utils | 0.10.54 / 0.10.54.post1, source commit above |
| QUDA | 1.1.0, runtime build tag `1.1.0-f4af6b974-sm_86` |
| QUDA checkout | `f4af6b974aab71ea525e9d7e5319cfd0403d2a80` |
| CUDA toolkit / driver | 12.0.140 / 595.91.07 |
| GPU | NVIDIA GeForce RTX 3060, 12 GiB |
| MPI / mpi4py | Open MPI 5.0.10 / 4.1.2, one rank |
| CuPy / NumPy / SciPy | 14.2.0 / 2.5.4 / 1.18.1 |

The smoke test used `S16T16/wilson_b6.0`, whose SHA-256 is
`a5014e41446fe7ac50556de0b7ff57d8f9b5d9fe7759e7a8d44b24779890a07f`.
Default NERSC checksum, link-trace, and plaquette checks passed. The independently
computed QUDA plaquette was `0.5952301540550844`, consistent with the header value
`0.5952301541`.

All 12 point-source spin-color columns were inverted at `kappa = 0.14` with
antiperiodic temporal boundary conditions. Explicitly applying the Dirac matrix
to each solution gave a maximum relative residual of `8.7056e-11`, below `1e-10`.
The local pion correlator was finite and positive at all 16 times. The gamma5
contraction and `sum |S|^2` agreed to a maximum relative difference of `1.49e-16`.
Example values are `C(1) = 0.0979855744361` and `C(8) = 3.00352178335e-5`.
`pip check` also passed.

This checks local loading, GPU/MPI linkage, inversion, and contraction. It does
not measure ensemble SNR, establish a ground-state mass or plateau, validate
the proposed scoring windows, test multi-GPU execution, or demonstrate an
optimized sink. No ensemble score has been produced.

## Reproduce the single-configuration smoke test

After activation, run the following Python body from the repository root, either
with `python -` or saved as `.cache/smoke_pion.py`. It writes the local report to
`.cache/smoke_pion.json`; a full execution log can be captured with
`python .cache/smoke_pion.py > .cache/smoke_pion.log 2>&1` when saved as a file.
The code is an environment check, not an evaluator or a submission API.

```python
"""Local environment smoke test; not an ensemble evaluator."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cupy as cp
import numpy as np
from mpi4py import MPI
from pyquda_utils import core, gamma, io, source

config = Path('/data/jinchen/ensembles/S16T16/wilson_b6.0')
start = perf_counter()
core.init([1, 1, 1, 1], [16, 16, 16, 16], resource_path='.cache/quda')
assert MPI.COMM_WORLD.Get_size() == 1
gauge = io.readNERSCGauge(str(config))
assert gauge.latt_info.global_size == [16, 16, 16, 16]
plaquette = np.asarray(gauge.plaquette())
assert np.isclose(plaquette[0], 0.5952301541, rtol=1e-9, atol=0)
latt = core.LatticeInfo([16, 16, 16, 16], -1, 1.0)
dirac = core.getWilson(latt, 1 / (2 * 0.14) - 4, 1e-10, 10000)
residuals = []
with dirac.useGauge(gauge):
    prop = core.invert(dirac, 'point', [0, 0, 0, 0])
    for spin in range(4):
        for color in range(3):
            b = source.source(latt, 'point', [0, 0, 0, 0], spin, color)
            r = dirac.mat(prop.getFermion(spin, color)) - b
            residuals.append(float((r.norm2() / b.norm2()) ** 0.5))
assert max(residuals) <= 1e-10, residuals
p = prop.data
local_density = cp.sum(cp.abs(p) ** 2, axis=(-4, -3, -2, -1))
correlator = cp.sum(local_density, axis=(0, 2, 3, 4)).get()
g5 = cp.asarray(gamma.gamma(15))
flat = p.reshape(latt.volume, 4, 4, 3, 3)
contracted = cp.einsum('ij,xkjba,kl,xliba->x', g5 @ g5,
                       flat.conj(), g5 @ g5, flat, optimize=True)
contracted = contracted.reshape(2, 16, 16, 16, 8).sum(axis=(0, 2, 3, 4)).get()
assert np.all(np.isfinite(correlator)) and np.all(correlator > 0)
assert np.allclose(contracted, correlator, rtol=1e-12, atol=0)
report = {
    'configuration': str(config),
    'sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
    'mpi': MPI.Get_library_version().strip(),
    'gpu': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
    'plaquette_total_spatial_temporal': plaquette.tolist(),
    'kappa': 0.14, 'residuals': residuals,
    'local_pion_C_t': correlator.tolist(),
    'contraction_relative_max_difference': float(np.max(np.abs(contracted-correlator)/correlator)),
    'elapsed_seconds': perf_counter()-start,
}
Path('.cache/smoke_pion.json').write_text(json.dumps(report, indent=2)+'\n')
print('SMOKE_REPORT', json.dumps(report), flush=True)
```
