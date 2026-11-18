#!/bin/bash
#SBATCH -J la_hmc32
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 03:00:00
#SBATCH -o /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/hmc32_%j.out
#SBATCH -e /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/hmc32_%j.err
set -euo pipefail
source /public/home/acwzoxdwvj/env_pyquda_conda.sh
cd /public/home/acwzoxdwvj/dianjun/LatticeArena
export MPLBACKEND=Agg
python pilot/hmc_generate.py --lattice 16,16,16,32 --out data/S16T32 --thermal 400 --trajectories 900 --save-every 5 --max-save 100 --n-steps 120 --tau 0.5
