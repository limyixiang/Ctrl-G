#!/bin/bash
#SBATCH -J alfworld-lvd
#SBATCH -p gpu-long
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-47:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash $0 [MODEL_SIZE [HMM_CONTEXT_STEPS]]"
  echo "Example: bash $0 4B 8"
  echo "Defaults: MODEL_SIZE=9B, HMM_CONTEXT_STEPS=1, MAX_HMM_CONTEXT_TOKENS=1024"
  exit 0
fi
if (( $# > 2 )); then
  echo "Usage: bash $0 [MODEL_SIZE [HMM_CONTEXT_STEPS]]" >&2
  exit 2
fi

MODEL_SIZE=${1:-${MODEL_SIZE:-9B}}
MODEL_SIZE=${MODEL_SIZE^^}
if [[ ! "$MODEL_SIZE" =~ ^[0-9]+([.][0-9]+)?B$ ]]; then
  echo "MODEL_SIZE must be a size such as 4B, 9B, or 0.8B" >&2
  exit 2
fi
MODEL=${MODEL:-Qwen/Qwen3.5-${MODEL_SIZE}}
MODEL_TAG=${MODEL##*-}
MODEL_TAG=${MODEL_TAG,,}
HMM_CONTEXT_STEPS=${2:-${HMM_CONTEXT_STEPS:-1}}
MAX_HMM_CONTEXT_TOKENS=${MAX_HMM_CONTEXT_TOKENS:-1024}
if [[ ! "$HMM_CONTEXT_STEPS" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "HMM_CONTEXT_STEPS must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! "$MAX_HMM_CONTEXT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_HMM_CONTEXT_TOKENS must be a positive integer" >&2
  exit 2
fi
SAMPLES=${SAMPLES:-results/alfworld/${MODEL_TAG}_context_rollout_v1/samples.jsonl}
HISTORY=${HISTORY:-$(dirname "$SAMPLES")/history.jsonl}
OUTPUT=${OUTPUT:-results/alfworld/hmm_context_data_${MODEL_TAG}_k${HMM_CONTEXT_STEPS}_b${MAX_HMM_CONTEXT_TOKENS}_v1}
JOB_NAME=${JOB_NAME:-alfworld-lvd-${MODEL_TAG}-k${HMM_CONTEXT_STEPS}-b${MAX_HMM_CONTEXT_TOKENS}}

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs

# Submit from Bash so Slurm receives the resolved job name before opening
# logs/%x_%j.{out,err}; #SBATCH directives cannot expand shell variables.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec sbatch --job-name="$JOB_NAME" "$0" "$MODEL_SIZE" "$HMM_CONTEXT_STEPS"
fi

source .venv-alfworld/bin/activate

LVD_SAMPLES=${LVD_SAMPLES:-10000}
SEED=${SEED:-42}

printf 'Model: %s\nSamples: %s\nHistory: %s\nOutput: %s\n' \
  "$MODEL" "$SAMPLES" "$HISTORY" "$OUTPUT"

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

# bash ctrlg-alfworld/slurm/build_hmm_data.sh 9B 4
# bash ctrlg-alfworld/slurm/build_hmm_data.sh 9B 8
# bash ctrlg-alfworld/slurm/build_hmm_data.sh 4B 4
# bash ctrlg-alfworld/slurm/build_hmm_data.sh 4B 8
