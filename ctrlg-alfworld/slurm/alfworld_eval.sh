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

# Compatibility single-condition launcher. For the matched two-condition
# array use ctrlg-alfworld/slurm/eval_grid.sh.
#
#   sbatch ctrlg-alfworld/slurm/alfworld_eval.sh decision_dfa 20 50
#   HMM=results/.../checkpoint-N \
#     sbatch ctrlg-alfworld/slurm/alfworld_eval.sh decision_dfa_hmm 134 50

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

export ALFWORLD_DATA="${ALFWORLD_DATA:-$WORKDIR/alfworld_data}"
export HF_HOME="${HF_HOME:-$WORKDIR/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

CONDITION=${1:-decision_dfa}
EPISODES=${2:-134}
MAX_STEPS=${3:-50}
MODEL=${MODEL:-$WORKDIR/models/Qwen3.5-9B}
OUTPUT=${OUTPUT:-results/alfworld/single_${CONDITION}_${SLURM_JOB_ID}}

HMM_ARGS=()
if [[ "$CONDITION" == "decision_dfa_hmm" ]]; then
  : "${HMM:?Set HMM to the matched decision-format checkpoint directory}"
  HMM_ARGS=(--hmm "$HMM")
fi

python ctrlg-alfworld/scripts/run_eval.py \
  --model "$MODEL" \
  --condition "$CONDITION" \
  "${HMM_ARGS[@]}" \
  --num_episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" \
  --out "$OUTPUT"
