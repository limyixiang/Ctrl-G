#!/bin/bash
#SBATCH -J alfworld-eval
#SBATCH -p gpu-long
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100-96:1
# SBATCH --exclude=xgpi13
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# SBATCH --mail-type=BEGIN,END,FAIL
# SBATCH --mail-user=e1121685@u.nus.edu

# Serves Qwen/Qwen3.5-9B with vLLM on this node, then points
# ctrlg-alfworld/scripts/run_eval.py at it (--backend vllm).
#
#   sbatch slurm/alfworld_eval.sh                 # 134 eval_out_of_distribution episodes
#   sbatch slurm/alfworld_eval.sh 20              # smoke test
#
# Unconstrained mode only. Constrained mode needs Ctrl-G's HMM x DFA logits
# processor inside the generate() loop, which the server cannot expose --
# run that with `--backend hf` (no vLLM) instead.

set -uo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs

source .venv-alfworld/bin/activate

export TMPDIR=/tmp

# config_tw.yaml holds literal '$ALFWORLD_DATA/...' paths that alfworld expands
# at load time, so this has to be exported before run_eval.py imports the env.
export ALFWORLD_DATA="${ALFWORLD_DATA:-$WORKDIR/alfworld_data}"
export HF_HOME="${HF_HOME:-$WORKDIR/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

NAME="Qwen/Qwen3.5-9B"
# Prefer a pre-downloaded copy -- compute nodes are usually offline.
MODEL="$WORKDIR/models/Qwen3.5-9B"
[[ -d "$MODEL" ]] || MODEL="$NAME"

NUM_EPISODES="${1:-134}"
MAX_STEPS="${2:-50}"
JOB="${SLURM_JOB_ID:-$$}"
PORT=$((8000 + JOB % 1000))
OUT="results/alfworld/unconstrained_$JOB"

echo "model=$MODEL  port=$PORT  episodes=$NUM_EPISODES  out=$OUT"

cleanup() {
  [[ -n "${VLLM_PID:-}" ]] || return 0
  kill "$VLLM_PID" 2>/dev/null
  pkill -P "$VLLM_PID" 2>/dev/null
  wait "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT

served=("$NAME")
[[ "$MODEL" != "$NAME" ]] && served+=("$MODEL")

vllm serve "$MODEL" \
  --served-model-name "${served[@]}" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  > "logs/vllm_${JOB}.log" 2>&1 &
VLLM_PID=$!

# First run downloads ~19GB of weights, so allow 20 min to come up.
ready=0
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=1; break; fi
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited during startup"; break; }
  sleep 10
done

if [[ $ready -ne 1 ]]; then
  echo "server never came up -- see logs/vllm_${JOB}.log"
  tail -n 50 "logs/vllm_${JOB}.log"
  exit 1
fi

python ctrlg-alfworld/scripts/run_eval.py \
  --backend vllm \
  --model "$NAME" \
  --tokenizer "$MODEL" \
  --base_url "http://127.0.0.1:$PORT/v1" \
  --mode unconstrained \
  --num_episodes "$NUM_EPISODES" \
  --max_steps "$MAX_STEPS" \
  --out "$OUT"
