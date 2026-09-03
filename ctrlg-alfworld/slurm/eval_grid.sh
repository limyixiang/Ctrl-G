#!/bin/bash
#SBATCH -J alfworld-pair
#SBATCH -p gpu-long
#SBATCH -t 12:00:00
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

export ALFWORLD_DATA="${ALFWORLD_DATA:-$WORKDIR/alfworld_data}"
export HF_HOME="${HF_HOME:-$WORKDIR/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-$WORKDIR/models/Qwen3.5-9B}
OUTPUT=${OUTPUT:-results/alfworld/pair_${SLURM_ARRAY_JOB_ID}}
EPISODES=${EPISODES:-134}
SEED=${SEED:-42}
CONDITIONS=(decision_dfa decision_dfa_hmm)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
PROMPT_ARGS=()
if [[ "${SHOW_ADMISSIBLE_ACTIONS:-0}" == "1" ]]; then
  PROMPT_ARGS+=(--show_admissible_actions)
fi

HMM_ARGS=()
if [[ "$CONDITION" == "decision_dfa_hmm" ]]; then
  : "${HMM:?Set HMM to the matched decision-format checkpoint directory}"
  HMM_ARGS=(--hmm "$HMM")
fi

python ctrlg-alfworld/scripts/run_eval.py \
  --model "$MODEL" \
  --condition "$CONDITION" \
  "${HMM_ARGS[@]}" \
  "${PROMPT_ARGS[@]}" \
  --num_episodes "$EPISODES" \
  --seed "$SEED" \
  --out "$OUTPUT"
