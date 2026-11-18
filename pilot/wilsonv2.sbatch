#!/bin/bash
#SBATCH -J la_wilsonv2
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 04:00:00
#SBATCH -o /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/wilsonv2_%j.out
#SBATCH -e /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/wilsonv2_%j.err
set -euo pipefail
source /public/home/acwzoxdwvj/env_pyquda_conda.sh
cd /public/home/acwzoxdwvj/dianjun/LatticeArena
export MPLBACKEND=Agg OMP_NUM_THREADS=1
export WILSON_PROTOCOL=tasks/wilson_loop/protocol_v2.json
python -m tasks.wilson_loop "$@"
