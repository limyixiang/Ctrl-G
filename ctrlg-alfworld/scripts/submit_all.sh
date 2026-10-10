#!/usr/bin/env bash
set -euo pipefail

# Raw unmasked collections start fresh. RESUME=1 requires
# an existing collection with the same raw version and retry policy.
RESUME=${RESUME:-0}

submit() {
    local model="$1"
    local name="$2"
    local output="$3"

    sbatch \
        -t 72:00:00 \
        -p gpu-long \
        --gres=gpu:h100-96:1 \
        --job-name="$name" \
        --exclude="xgpi17,xgpi18,xgpi15" \
        --export="ALL,RESUME=$RESUME,MODEL=$model,MAX_STEPS=50,EPISODES=3553,OUTPUT=$output" \
        ctrlg-alfworld/slurm/collect_hmm_samples.sh
}

submit "Qwen/Qwen3.5-9B"   "9b"   "results/alfworld/9b_context_rollout_v1"
submit "Qwen/Qwen3.5-4B"   "4b"   "results/alfworld/4b_context_rollout_v1"
# submit "Qwen/Qwen3.5-2B"   "2b"   "results/alfworld/2b_context_rollout_v1"
# submit "Qwen/Qwen3.5-0.8B" "0.8b" "results/alfworld/0.8b_context_rollout_v1"
