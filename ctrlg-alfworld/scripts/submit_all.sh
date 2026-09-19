#!/usr/bin/env bash
set -euo pipefail

submit() {
    local model="$1"
    local name="$2"
    local output="$3"

    sbatch \
        -t 24:00:00 \
        -p gpu \
        --gres=gpu:h100-47:1 \
        --job-name="$name" \
        --exclude="xgpi17" \
        --export="ALL,MODEL=$model,MAX_STEPS=50,OVERWRITE=1,SHOW_ADMISSIBLE_ACTIONS=0,EPISODES=100,OUTPUT=$output" \
        ctrlg-alfworld/slurm/collect_hmm_samples.sh
}

# submit "Qwen/Qwen3.5-9B"   "9b"   "results/alfworld/9b_policy_hmm"
# submit "Qwen/Qwen3.5-4B"   "4b"   "results/alfworld/4b_policy_hmm"
# submit "Qwen/Qwen3.5-2B"   "2b"   "results/alfworld/2b_policy_hmm"
submit "Qwen/Qwen3.5-0.8B" "0.8b" "results/alfworld/0.8b_policy_hmm"
