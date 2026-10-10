#!/bin/bash
#SBATCH -J alfworld-eval
#SBATCH -p gpu-long
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

# Compatibility single-condition launcher. For the matched two-condition
# array use ctrlg-alfworld/slurm/eval_grid.sh.
#
#   sbatch ctrlg-alfworld/slurm/alfworld_eval.sh state_prompt 20 50
#   MODEL='Qwen/Qwen3.5-9B' HMM='results/alfworld/hmm_context_model_9b_v1/checkpoint-112' sbatch ctrlg-alfworld/slurm/alfworld_eval.sh state_ctrlg

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

export TMPDIR=/tmp
export ALFWORLD_DATA="${ALFWORLD_DATA:-$WORKDIR/alfworld_data}"
export HF_HOME="${HF_HOME:-$WORKDIR/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

CONDITION=${1:-state_prompt}
EPISODES=${2:-134}
MAX_STEPS=${3:-50}
BEAM_SIZE=${BEAM_SIZE:-1}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
OUTPUT=${OUTPUT:-results/alfworld/single_${CONDITION}_${SLURM_JOB_ID}}
HMM_ARGS=()
if [[ "$CONDITION" == "state_ctrlg" ]]; then
  : "${HMM:?Set HMM to the matched state-format checkpoint directory}"
  HMM_ARGS=(--hmm "$HMM")
fi

python ctrlg-alfworld/scripts/run_eval.py \
  --model "$MODEL" \
  --condition "$CONDITION" \
  "${HMM_ARGS[@]}" \
  --num_episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" \
  --hmm_context_steps "${HMM_CONTEXT_STEPS:-1}" \
  --max_hmm_context_tokens "${MAX_HMM_CONTEXT_TOKENS:-1024}" \
  --beam_size "$BEAM_SIZE" \
  --out "$OUTPUT"
