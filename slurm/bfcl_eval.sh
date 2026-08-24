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

export BFCL_PROJECT_ROOT="$SLURM_SUBMIT_DIR/results/bfcl"
mkdir -p "$BFCL_PROJECT_ROOT" logs
touch "$BFCL_PROJECT_ROOT/.env"

source .venv-bfcl/bin/activate
set -uo pipefail

PORT=1053
TEST_CATEGORIES="non_live"

declare -A models=(
  ["Qwen/Qwen3-8B-FC"]="$SLURM_SUBMIT_DIR/models/Qwen3-8B"
  ["meta-llama/Llama-3.1-8B-Instruct-FC"]="$SLURM_SUBMIT_DIR/models/Llama-3.1-8B-Instruct"
)

cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT

for name in "${!models[@]}"; do
  path="${models[$name]}"

  extra=()
  [[ "$name" == Qwen/Qwen3* ]] && \
    extra+=(--default-chat-template-kwargs '{"enable_thinking": false}')

  vllm serve "$path" \
    --served-model-name "$name" "${name%-FC}" "$path" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    "${extra[@]}" &
  VLLM_PID=$!

  ready=0
  for _ in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then ready=1; break; fi
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited during startup"; break; }
    sleep 10
  done

  if [[ $ready -eq 1 ]]; then
    export REMOTE_OPENAI_TOKENIZER_PATH="$path"
    bfcl generate --model "$name" --test-category "$TEST_CATEGORIES" --skip-server-setup
  else
    echo "SKIP $name — server never came up"
  fi

  kill "$VLLM_PID" 2>/dev/null
  wait "$VLLM_PID" 2>/dev/null
  pkill -P "$VLLM_PID" 2>/dev/null
  sleep 15
  unset VLLM_PID

  [[ $ready -eq 1 ]] && bfcl evaluate --model "$name" --test-category "$TEST_CATEGORIES"
done
