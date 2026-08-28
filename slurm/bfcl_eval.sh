#!/bin/bash
#SBATCH -J bfcl-eval
#SBATCH -p gpu-long
#SBATCH -t 10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-47:1
#SBATCH --array=0-1
# SBATCH --exclude=xgpi13
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH --mail-user=e1121685@u.nus.edu

export BFCL_PROJECT_ROOT="$SLURM_SUBMIT_DIR/results/bfcl"
mkdir -p "$BFCL_PROJECT_ROOT" logs
touch "$BFCL_PROJECT_ROOT/.env"

source .venv-bfcl/bin/activate
set -uo pipefail

# One array task per model (keep --array above in sync with this list).
names=(
  "Qwen/Qwen3-8B-FC"
  "meta-llama/Llama-3.1-8B-Instruct-FC"
)
paths=(
  "$SLURM_SUBMIT_DIR/models/Qwen3-8B"
  "$SLURM_SUBMIT_DIR/models/Llama-3.1-8B-Instruct"
)

name="${names[$SLURM_ARRAY_TASK_ID]}"
path="${paths[$SLURM_ARRAY_TASK_ID]}"
PORT=$((1053 + SLURM_ARRAY_TASK_ID))
# TEST_CATEGORIES="non_live"
TEST_CATEGORIES="simple_python,simple_java,simple_javascript,parallel,multiple,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum,format_sensitivity" # all except web search (need api)

cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT

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
