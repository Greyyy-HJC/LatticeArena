#!/bin/bash
#SBATCH -J la_hmc200c
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 03:00:00
#SBATCH -o /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/hmc200c_%j.out
#SBATCH -e /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/hmc200c_%j.err
set -euo pipefail
source /public/home/acwzoxdwvj/env_pyquda_conda.sh
cd /public/home/acwzoxdwvj/dianjun/LatticeArena
export MPLBACKEND=Agg
python pilot/hmc_generate.py --lattice 16,16,16,16 --out data/S16T16C --thermal 400 --trajectories 1400 --save-every 5 --max-save 200 --n-steps 120 --tau 0.5
