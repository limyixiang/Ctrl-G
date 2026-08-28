#!/bin/bash
#SBATCH -J jsb-eval
#SBATCH -p gpu
#SBATCH -t 3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-96:1
# SBATCH --exclude=xgpi13
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH --mail-user=e1121685@u.nus.edu

set -uo pipefail

source .venv-jsb/bin/activate

# declare -A models=(
#   ["Qwen/Qwen3-8B-FC"]="models/Qwen3-8B"
#   ["meta-llama/Llama-3.1-8B-Instruct-FC"]="models/Llama-3.1-8B-Instruct"
# )

# for name in "${!models[@]}"; do
#   lm_eval \
#     --model vllm --gen_kwargs max_gen_toks=16384 \
#     --model_args pretrained="${models[$name]}",enable_thinking=True,think_end_token='</think>',tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
#     --tasks jsonschema_bench \
#     --batch_size auto \
#     --apply_chat_template \
#     --fewshot_as_multiturn \
#     --output_path results/jsb/thinking_enabled \
#     --log_samples \
#     # --limit 10 \
# done

lm_eval \
  --model vllm --gen_kwargs max_gen_toks=16384 \
  --model_args pretrained=models/Qwen3-8B,enable_thinking=True,think_end_token='</think>',tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
  --tasks jsonschema_bench \
  --batch_size auto \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path results/jsb/thinking_enabled \
  --log_samples \
  # --limit 10 \

lm_eval \
  --model vllm --gen_kwargs max_gen_toks=4096 \
  --model_args pretrained=models/Llama-3.1-8B-Instruct,enable_thinking=True,think_end_token='</think>',tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
  --tasks jsonschema_bench \
  --batch_size auto \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path results/jsb/thinking_enabled \
  --log_samples \
  # --limit 10 \
