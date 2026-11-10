#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/jinchen/env/pyquda.sh
export PATH="/home/jinchen/software/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/home/jinchen/software/openmpi/lib:${LD_LIBRARY_PATH:-}"
export CUDA_PATH=/usr
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec .venv/bin/python -m tasks.pion_projection "$@"
