#!/bin/bash
#SBATCH -J la_agent
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 01:00:00
#SBATCH -o /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/agent_%j.out
#SBATCH -e /public/home/acwzoxdwvj/dianjun/LatticeArena/logs/agent_%j.err
set -euo pipefail
source /public/home/acwzoxdwvj/env_pyquda_conda.sh
cd /public/home/acwzoxdwvj/dianjun/LatticeArena
export MPLBACKEND=Agg
export LLM_API_KEY="$(cat .key)"
python -m tasks.pion_projection --output results/pion_projection/s16t16_v1 agent --model deepseek-v4-pro --rounds 3
