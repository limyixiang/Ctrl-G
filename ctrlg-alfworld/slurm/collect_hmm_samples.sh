#!/bin/bash
#SBATCH -J alfworld-sample
#SBATCH -p gpu-long
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

# to run a pilot test:
# MAX_STEPS=50 OVERWRITE=1 SHOW_ADMISSIBLE_ACTIONS=0 EPISODES=1 OUTPUT=results/alfworld/pilot_actions_hidden sbatch -t 3:00:00 -p gpu --gres=gpu:h100-47:1 ctrlg-alfworld/slurm/collect_hmm_samples.sh

# actual runs
# RESUME=1 SHOW_ADMISSIBLE_ACTIONS=0 OUTPUT=results/alfworld/actions_hidden/hmm_samples_h100_96 sbatch --job-name=alfworld-hidden ctrlg-alfworld/slurm/collect_hmm_samples.sh
# RESUME=1 SHOW_ADMISSIBLE_ACTIONS=1 OUTPUT=results/alfworld/actions_shown/hmm_samples_h100_96 sbatch --job-name=alfworld-shown ctrlg-alfworld/slurm/collect_hmm_samples.sh

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
SERVED_NAME=Qwen/Qwen3.5-9B
EPISODES=${EPISODES:-3553}
TEMPERATURE=${TEMPERATURE:-0.7}
if [[ "${RESUME:-0}" == "1" && -z "${OUTPUT:-}" ]]; then
  echo "RESUME=1 requires OUTPUT to name the existing collection directory" >&2
  exit 2
fi
OUTPUT=${OUTPUT:-results/alfworld/hmm_samples_${SLURM_JOB_ID}}
PORT=$((8000 + SLURM_JOB_ID % 1000))
PROMPT_ARGS=()
if [[ "${SHOW_ADMISSIBLE_ACTIONS:-0}" == "1" ]]; then
  PROMPT_ARGS+=(--show_admissible_actions)
fi
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
  --max_hmm_sequence_tokens 256 \
  --max_thought_tokens 1024 \
  --max_decision_tokens 256 \
  "${PROMPT_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  --out "$OUTPUT"

jq '.' results/alfworld/pilot_actions_hidden/history.jsonl > results/alfworld/pilot_actions_hidden/history.json