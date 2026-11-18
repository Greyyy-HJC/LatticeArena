#!/bin/bash
#SBATCH -J la_pion_ms
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 04:00:00
#SBATCH -o /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/pion_ms_%j.out
#SBATCH -e /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/pion_ms_%j.err
set -euo pipefail
source /public/home/acwzoxdwvj/env_pyquda_conda.sh
cd /public/home/acwzoxdwvj/dianjun/LatticeArena
export MPLBACKEND=Agg OMP_NUM_THREADS=1
export LATTICEARENA_PROTOCOL=tasks/pion_projection/protocol_multisource.json
python -m tasks.pion_projection.measure_multisource "$@"
