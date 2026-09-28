#!/bin/bash
#SBATCH -J alfworld-lvd
#SBATCH -p gpu-long
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
SAMPLES=${SAMPLES:?Set SAMPLES to the collected samples.jsonl}
OUTPUT=${OUTPUT:-results/alfworld/hmm_data_9b}
LVD_SAMPLES=${LVD_SAMPLES:-10000}
SEED=${SEED:-42}

python ctrlg-alfworld/scripts/build_hmm_data.py \
  --samples "$SAMPLES" \
  --tokenizer "$MODEL" \
  --model "$MODEL" \
  --output_dir "$OUTPUT" \
  --dataset alfworld_actions \
  --lvd_samples "$LVD_SAMPLES" \
  --seed "$SEED" \
  --save_embeddings

# SAMPLES=results/alfworld/9b_rollout/samples.jsonl sbatch -J alfworld-lvd-9b ctrlg-alfworld/slurm/build_hmm_data.sh