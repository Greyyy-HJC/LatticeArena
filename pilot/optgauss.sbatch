#!/bin/bash
#SBATCH -J la_optgauss
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 02:00:00
#SBATCH -o /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/optgauss_%j.out
#SBATCH -e /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/optgauss_%j.err
set -euo pipefail
source /public/home/acwzoxdwvj/env_pyquda_conda.sh
cd /public/home/acwzoxdwvj/dianjun/LatticeArena
export MPLBACKEND=Agg
python -m tasks.optimize_gaussian "$@"
