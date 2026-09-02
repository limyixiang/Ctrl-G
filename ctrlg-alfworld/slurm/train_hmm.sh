#!/bin/bash
#SBATCH -J alfworld-hmm
#SBATCH -p gpu-long
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100-96:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$WORKDIR"
mkdir -p logs
source .venv-alfworld/bin/activate

MODEL=${MODEL:-$WORKDIR/models/Qwen3.5-9B}
DATA_DIR=${DATA_DIR:?Set DATA_DIR to the build_hmm_data.py output directory}
OUTPUT=${OUTPUT:-results/alfworld/hmm_model}
HIDDEN_STATES=${HIDDEN_STATES:-4096}
TRAIN_CHUNKS=${TRAIN_CHUNKS:-8}
EM_SCHEDULE=${EM_SCHEDULE:-50,8}
BATCH_SIZE=${BATCH_SIZE:-32}
SEED=${SEED:-42}
SAVE_PER_STEP=${SAVE_PER_STEP:-8}
DATASET=alfworld_actions
mkdir -p "$OUTPUT"

read -r VOCAB_SIZE EOS_TOKEN_ID < <(
  python -c \
    "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MODEL'); print(t.vocab_size, t.eos_token_id)"
)

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

torchrun --standalone --nproc_per_node=1 distillation/train_hmm.py \
  --model_path "$OUTPUT" \
  --checkpoint 0 \
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

FINAL_CHECKPOINT=${FINAL_CHECKPOINT:-$(
  python -c 'import sys; print(sum(int(a) * int(b) for a, b in (part.split(",") for part in sys.argv[1].split(";") if part)))' "$EM_SCHEDULE"
)}
python ctrlg-alfworld/scripts/evaluate_hmm_fit.py \
  --hmm "$OUTPUT/checkpoint-$FINAL_CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --dataset "$DATASET" \
  --batch-size "$BATCH_SIZE" \
  --out "$OUTPUT/held_out_fit.json"
