#!/bin/bash
#SBATCH -J la_pion_gram
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 04:00:00
#SBATCH -o /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/pion_gram_%j.out
#SBATCH -e /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/pion_gram_%j.err
set -euo pipefail
source /public/home/acwzoxdwvj/env_pyquda_conda.sh
cd /public/home/acwzoxdwvj/dianjun/LatticeArena
export MPLBACKEND=Agg OMP_NUM_THREADS=1
export LATTICEARENA_PROTOCOL=results/pion_projection/s16t16_v1/protocol.json
python -m tasks.pion_projection.measure_gram "$@"
