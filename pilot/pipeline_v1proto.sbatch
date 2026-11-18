#!/bin/bash
#SBATCH -J la_pipev1
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 04:00:00
#SBATCH -o /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/pipev1_%j.out
#SBATCH -e /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/pipev1_%j.err
set -euo pipefail
source /public/home/acwzoxdwvj/env_pyquda_conda.sh
cd /public/home/acwzoxdwvj/dianjun/LatticeArena
export MPLBACKEND=Agg OMP_NUM_THREADS=1
export LATTICEARENA_PROTOCOL=results/pion_projection/s16t16_v1/protocol.json
export LATTICEARENA_LEGACY_REFERENCE=1
python -m tasks.pion_projection "$@"
