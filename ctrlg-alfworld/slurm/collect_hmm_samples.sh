#!/bin/bash
#SBATCH -J alfworld-sample
#SBATCH -p gpu-long
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# to run a 1-sample test:
#   SHOW_ADMISSIBLE_ACTIONS=1 EPISODES=1 SAMPLES_PER_STATE=1 OUTPUT=results/alfworld/pilot_actions_shown sbatch ctrlg-alfworld/slurm/collect_hmm_samples.sh

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

export ALFWORLD_DATA="${ALFWORLD_DATA:-$WORKDIR/alfworld_data}"
export HF_HOME="${HF_HOME:-$WORKDIR/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-$WORKDIR/models/Qwen3.5-9B}
SERVED_NAME=Qwen/Qwen3.5-9B
EPISODES=${EPISODES:-100}
SAMPLES_PER_STATE=${SAMPLES_PER_STATE:-4}
TEMPERATURE=${TEMPERATURE:-0.7}
OUTPUT=${OUTPUT:-results/alfworld/hmm_samples_${SLURM_JOB_ID}}
PORT=$((8000 + SLURM_JOB_ID % 1000))
PROMPT_ARGS=()
if [[ "${SHOW_ADMISSIBLE_ACTIONS:-0}" == "1" ]]; then
  PROMPT_ARGS+=(--show_admissible_actions)
fi
OUTPUT_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OUTPUT_ARGS+=(--overwrite)
fi

cleanup() {
  [[ -n "${VLLM_PID:-}" ]] || return 0
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  >"logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    break
  fi
  kill -0 "$VLLM_PID"
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null

python ctrlg-alfworld/scripts/run_rollouts.py \
  --backend vllm \
  --model "$SERVED_NAME" \
  --tokenizer "$MODEL" \
  --base_url "http://127.0.0.1:$PORT/v1" \
  --num_episodes "$EPISODES" \
  --samples_per_state "$SAMPLES_PER_STATE" \
  --temperature "$TEMPERATURE" \
  "${PROMPT_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  --out "$OUTPUT"
