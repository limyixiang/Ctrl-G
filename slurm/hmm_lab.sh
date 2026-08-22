#!/bin/bash
#SBATCH -J hmm_lab
#SBATCH -p gpu
#SBATCH -t 30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-47:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH --mail-user=e1121685@u.nus.edu

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs

python ctrlg/hmm_lab.py
