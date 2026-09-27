#!/bin/bash
#SBATCH -J alfworld-hmm
#SBATCH -p gpu
#SBATCH -t 3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e1121685@u.nus.edu

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
DATA_DIR=${DATA_DIR:?Set DATA_DIR to the build_hmm_data.py output directory}
OUTPUT=${OUTPUT:-results/alfworld/hmm_model_9b}
HIDDEN_STATES=${HIDDEN_STATES:-4096}
TRAIN_CHUNKS=${TRAIN_CHUNKS:-8}
EM_SCHEDULE=${EM_SCHEDULE:-'8,1;4,2;2,4;1,8'}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
SEED=${SEED:-42}
SAVE_PER_STEP=${SAVE_PER_STEP:-8}
DATASET=alfworld_actions
mkdir -p "$OUTPUT"

if [[ ! "$RESUME_CHECKPOINT" =~ ^[0-9]+$ ]]; then
  echo "RESUME_CHECKPOINT must be a nonnegative integer" >&2
  exit 1
fi

if (( RESUME_CHECKPOINT == 0 )); then
  cp "$DATA_DIR/${DATASET}.metadata.json" "$OUTPUT/training_data_metadata.json"

  read -r VOCAB_SIZE EOS_TOKEN_ID < <(
    python -c \
      "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MODEL'); print(len(t), t.eos_token_id)"
  )

  echo "Running distillation/lvd_hmm.py"
  python distillation/lvd_hmm.py \
    --sequences_file "$DATA_DIR/${DATASET}.lvd" \
    --embeddings_file "$DATA_DIR/${DATASET}.lvd.embeddings" \
    --hidden_states "$HIDDEN_STATES" \
    --vocab_size "$VOCAB_SIZE" \
    --eos_token_id "$EOS_TOKEN_ID" \
    --kmeans_iterations 100 \
    --pseudocount 0.001 \
    --seed "$SEED" \
    --output_file "$OUTPUT/checkpoint-0"
else
  if [[ ! -d "$OUTPUT/checkpoint-$RESUME_CHECKPOINT" ]]; then
    echo "resume checkpoint does not exist: $OUTPUT/checkpoint-$RESUME_CHECKPOINT" >&2
    exit 1
  fi
  if [[ ! -f "$OUTPUT/training_data_metadata.json" ]] || \
     ! cmp -s "$DATA_DIR/${DATASET}.metadata.json" "$OUTPUT/training_data_metadata.json"; then
    echo "training data metadata is missing or differs from the original run" >&2
    exit 1
  fi
  echo "Resuming from $OUTPUT/checkpoint-$RESUME_CHECKPOINT"
fi

echo "Running distillation/train_hmm.py"
torchrun --standalone --nproc_per_node=1 distillation/train_hmm.py \
  --model_path "$OUTPUT" \
  --checkpoint "$RESUME_CHECKPOINT" \
  --save_per_step "$SAVE_PER_STEP" \
  --data_path "$DATA_DIR" \
  --dataset "$DATASET" \
  --total_chunks "$TRAIN_CHUNKS" \
  --batch_size "$BATCH_SIZE" \
  --em_schedule "$EM_SCHEDULE" \
  --dropout 0.001 \
  --pseudocount 0.001 \
  --seed "$SEED" \
  --log_file "$OUTPUT/train.log"

SCHEDULED_FINAL_CHECKPOINT=$(
  python -c 'import sys; print(int(sys.argv[1]) + sum(int(a) * int(b) for a, b in (part.split(",") for part in sys.argv[2].split(";") if part)))' "$RESUME_CHECKPOINT" "$EM_SCHEDULE"
)
BEST_CHECKPOINT=$(
  python ctrlg-alfworld/scripts/select_best_hmm_checkpoint.py \
    --log "$OUTPUT/train.log" \
    --model-path "$OUTPUT" \
    --out "$OUTPUT/checkpoint_selection.json"
)
# EVAL_CHECKPOINT is an explicit escape hatch. FINAL_CHECKPOINT remains a
# backwards-compatible alias for existing job submissions.
EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-${FINAL_CHECKPOINT:-$BEST_CHECKPOINT}}
if [[ ! -d "$OUTPUT/checkpoint-$EVAL_CHECKPOINT" ]]; then
  echo "selected checkpoint does not exist: $OUTPUT/checkpoint-$EVAL_CHECKPOINT" >&2
  exit 1
fi
echo "scheduled final checkpoint: $SCHEDULED_FINAL_CHECKPOINT"
echo "best saved checkpoint by dev log-likelihood: $BEST_CHECKPOINT"
echo "evaluating checkpoint: $EVAL_CHECKPOINT"
python ctrlg-alfworld/scripts/evaluate_hmm_fit.py \
  --hmm "$OUTPUT/checkpoint-$EVAL_CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --dataset "$DATASET" \
  --batch-size "$BATCH_SIZE" \
  --out "$OUTPUT/held_out_fit.json"
