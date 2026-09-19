#!/bin/bash
#SBATCH -J alfworld-sample
#SBATCH -p gpu-long
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-47:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

# to run a pilot test:
# MAcdmples.sh

# Example:
# MODEL=Qwen/Qwen3.5-9B OVERWRITE=1 EPISODES=100 OUTPUT=results/alfworld/9b_policy_hmm sbatch ctrlg-alfworld/slurm/collect_hmm_samples.sh

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

export TMPDIR=/tmp
export ALFWORLD_DATA="${ALFWORLD_DATA:-$WORKDIR/alfworld_data}"
export HF_HOME="${HF_HOME:-$WORKDIR/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
SERVED_NAME=${SERVED_NAME:-$MODEL}
EPISODES=${EPISODES:-3553}
TEMPERATURE=${TEMPERATURE:-0.7}
if [[ "${RESUME:-0}" == "1" && -z "${OUTPUT:-}" ]]; then
  echo "RESUME=1 requires OUTPUT to name the existing collection directory" >&2
  exit 2
fi
OUTPUT=${OUTPUT:-results/alfworld/hmm_samples_${SLURM_JOB_ID}}
PORT=$((8000 + SLURM_JOB_ID % 1000))
OUTPUT_ARGS=()
if [[ "${RESUME:-0}" == "1" && "${OVERWRITE:-0}" == "1" ]]; then
  echo "RESUME=1 and OVERWRITE=1 are mutually exclusive" >&2
  exit 2
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  OUTPUT_ARGS+=(--resume)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OUTPUT_ARGS+=(--overwrite)
fi
if [[ "${MAX_STEPS:-50}" -ne 50 ]]; then
  OUTPUT_ARGS+=(--max_steps $MAX_STEPS)
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
  --gdn-prefill-backend triton \
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
  --temperature "$TEMPERATURE" \
  --max_hmm_sequence_tokens 640 \
  --max_thought_tokens 1024 \
  --max_decision_tokens 512 \
  --max_action_tokens 32 \
  "${OUTPUT_ARGS[@]}" \
  --out "$OUTPUT"

jq -s '.' "$OUTPUT/history.jsonl" > "$OUTPUT/history.json"
