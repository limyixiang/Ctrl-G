#!/usr/bin/env bash
set -euo pipefail

# State-format collections start fresh; set RESUME=1 only for an existing
# state-format run. Decision-format collections cannot be resumed.
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
        --export="ALL,RESUME=$RESUME,MODEL=$model,MAX_STEPS=50,SHOW_ADMISSIBLE_ACTIONS=0,EPISODES=3553,OUTPUT=$output" \
        ctrlg-alfworld/slurm/collect_hmm_samples.sh
}

submit "Qwen/Qwen3.5-9B"   "9b"   "results/alfworld/9b_state_rollout"
submit "Qwen/Qwen3.5-4B"   "4b"   "results/alfworld/4b_state_rollout"
# submit "Qwen/Qwen3.5-2B"   "2b"   "results/alfworld/2b_state_rollout"
# submit "Qwen/Qwen3.5-0.8B" "0.8b" "results/alfworld/0.8b_state_rollout"
