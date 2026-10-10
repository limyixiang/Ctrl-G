#!/bin/bash
#SBATCH -J alfworld-pair
#SBATCH -p gpu-long
#SBATCH -t 72:00:00
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
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
OUTPUT=${OUTPUT:-results/alfworld/pair_${SLURM_ARRAY_JOB_ID}}
EPISODES=${EPISODES:-134}
BEAM_SIZE=${BEAM_SIZE:-1}
SEED=${SEED:-42}
CONDITIONS=(state_prompt state_ctrlg)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
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
  --hmm_context_steps "${HMM_CONTEXT_STEPS:-1}" \
  --max_hmm_context_tokens "${MAX_HMM_CONTEXT_TOKENS:-1024}" \
  --beam_size "$BEAM_SIZE" \
  --seed "$SEED" \
  --out "$OUTPUT"

# MODEL='Qwen/Qwen3.5-9B' HMM='results/alfworld/hmm_context_model_9b_v1/checkpoint-112' OUTPUT='results/alfworld/pair_9b' sbatch ctrlg-alfworld/slurm/eval_grid.sh
# MODEL='Qwen/Qwen3.5-4B' HMM='results/alfworld/hmm_context_model_4b_v1/checkpoint-112' OUTPUT='results/alfworld/pair_4b' sbatch --dependency=afterok:907490 ctrlg-alfworld/slurm/eval_grid.sh
