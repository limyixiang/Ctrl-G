#!/bin/bash
#SBATCH -J alfworld-lvd
#SBATCH -p gpu-long
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

MODEL=${MODEL:-$WORKDIR/models/Qwen3.5-9B}
SAMPLES=${SAMPLES:?Set SAMPLES to the collected samples.jsonl}
OUTPUT=${OUTPUT:-results/alfworld/hmm_data}
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
