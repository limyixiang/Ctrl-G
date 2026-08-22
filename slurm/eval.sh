#!/bin/bash
#SBATCH -J bfcl-eval
#SBATCH -p gpu
#SBATCH -t 3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-47:1
# SBATCH --exclude=xgpi13
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH --mail-user=e1121685@u.nus.edu

# In your shell environment
export BFCL_PROJECT_ROOT=./bfcl

set -uo pipefail

TEST_CATEGORIES="non_live"

declare -A models=(
  ["Qwen/Qwen3-8B-FC"]="./models/Qwen3-8B"
  ["meta-llama/Llama-3.1-8B-Instruct-FC"]="models/Llama-3.1-8B-Instruct"
)

for name in "${!models[@]}"; do
  # bfcl generate --model "$name" --test-category "$TEST_CATEGORIES" \
  #   --backend vllm --local-model-path "${models[$name]}" --num-gpus 1 \
  #   --gpu-memory-utilization 0.9
  bfcl evaluate --model "$name" --test-category "$TEST_CATEGORIES"
done
