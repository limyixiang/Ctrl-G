#!/bin/bash
#SBATCH -J alfworld-grid
#SBATCH -p gpu-long
#SBATCH -t 72:00:00
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude="xgpi17,xgpi18,xgpi15"
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

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
OUTPUT=${OUTPUT:-results/alfworld/grid_${SLURM_ARRAY_JOB_ID}}
EPISODES=${EPISODES:-134}
BEAM_SIZE=${BEAM_SIZE:-1}
SEED=${SEED:-42}
SKILLS=${SKILLS:-ctrlg-alfworld/templates/SKILLS.md}
SKILLS_ARGS=()
if [[ "${ALLOW_SKILLS_MISMATCH:-0}" == "1" ]]; then
  SKILLS_ARGS=(--allow-skills-mismatch)
fi
CONDITION=state_ctrlg
HMM_ARGS=()
# Each base model shares one prompt-only baseline across both HMM contexts.
case "$SLURM_ARRAY_TASK_ID" in
  0)
    ARM=k8
    : "${HMM_K8:?Set HMM_K8 to the matched k=8 state-format checkpoint directory}"
    HMM_ARGS=(--hmm "$HMM_K8" --hmm_context_steps 8)
    ;;
  1)
    ARM=k4
    : "${HMM_K4:?Set HMM_K4 to the matched k=4 state-format checkpoint directory}"
    HMM_ARGS=(--hmm "$HMM_K4" --hmm_context_steps 4)
    ;;
  2)
    ARM=prompt_only
    CONDITION=state_prompt
    ;;
  *)
    echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID (expected 0, 1, or 2)" >&2
    exit 1
    ;;
esac

python ctrlg-alfworld/scripts/run_eval.py \
  --model "$MODEL" \
  --condition "$CONDITION" \
  --skills "$SKILLS" \
  "${SKILLS_ARGS[@]}" \
  "${HMM_ARGS[@]}" \
  --num_episodes "$EPISODES" \
  --max_hmm_context_tokens "${MAX_HMM_CONTEXT_TOKENS:-1024}" \
  --beam_size "$BEAM_SIZE" \
  --seed "$SEED" \
  --out "$OUTPUT/$ARM"

# Outputs are written under $OUTPUT/{k8,k4,prompt_only}/.
# MODEL='Qwen/Qwen3.5-9B' HMM_K8='results/alfworld/hmm_context_model_9b_k8_v1/checkpoint-824' HMM_K4='results/alfworld/hmm_context_model_9b_k4_v1/checkpoint-824' OUTPUT='results/alfworld/grid_9b' sbatch ctrlg-alfworld/slurm/eval_grid.sh
# MODEL='Qwen/Qwen3.5-4B' HMM_K8='results/alfworld/hmm_context_model_4b_k8_v1/checkpoint-824' HMM_K4='results/alfworld/hmm_context_model_4b_k4_v1/checkpoint-824' OUTPUT='results/alfworld/grid_4b' sbatch ctrlg-alfworld/slurm/eval_grid.sh
