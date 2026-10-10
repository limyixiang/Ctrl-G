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
HISTORY=${HISTORY:-$(dirname "$SAMPLES")/history.jsonl}
HMM_CONTEXT_STEPS=${HMM_CONTEXT_STEPS:-1}
MAX_HMM_CONTEXT_TOKENS=${MAX_HMM_CONTEXT_TOKENS:-1024}
OUTPUT=${OUTPUT:-results/alfworld/hmm_context_data_9b_k${HMM_CONTEXT_STEPS}_b${MAX_HMM_CONTEXT_TOKENS}_v1}
LVD_SAMPLES=${LVD_SAMPLES:-10000}
SEED=${SEED:-42}

python ctrlg-alfworld/scripts/build_hmm_data.py \
  --samples "$SAMPLES" \
  --history "$HISTORY" \
  --hmm_context_steps "$HMM_CONTEXT_STEPS" \
  --max_hmm_context_tokens "$MAX_HMM_CONTEXT_TOKENS" \
  --tokenizer "$MODEL" \
  --model "$MODEL" \
  --output_dir "$OUTPUT" \
  --dataset alfworld_actions \
  --lvd_samples "$LVD_SAMPLES" \
  --lvd_embedding_budget_gib "${LVD_EMBEDDING_BUDGET_GIB:-32}" \
  --seed "$SEED" \
  --save_embeddings

# HMM_CONTEXT_STEPS=3 SAMPLES=results/alfworld/9b_context_rollout_v1/samples.jsonl sbatch -J alfworld-lvd-9b-k3 ctrlg-alfworld/slurm/build_hmm_data.sh
