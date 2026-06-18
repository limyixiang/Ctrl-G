#!/bin/bash
set -x
# Configuration
# export LVD_PATH="dapo/DAPO-DAPO-Baseline-V7-S60/DAPO-DAPO-Baseline-V7-S60.lvd"
# export MODEL_PATH="/path/to/your/model_or_checkpoint"

export LVD_PATH="$1"
export MODEL_PATH="$2"

GPU_SPEC=${3:-"4,5"}  # Default to GPUs 4,5 if not specified
shift  # Remove first argument, keep the rest

# Check if GPU_SPEC is a number (backward compatibility) or a list
if [[ "$GPU_SPEC" =~ ^[0-9]+$ ]]; then
    # It's a number, use GPUs 0,1,2,...
    NUM_GPUS=$GPU_SPEC
    CUDA_VISIBLE_DEVICES=""
    for ((i=0; i<NUM_GPUS; i++)); do
        if [ $i -eq 0 ]; then
            CUDA_VISIBLE_DEVICES="$i"
        else
            CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$i"
        fi
    done
else
    # It's a list of GPUs
    CUDA_VISIBLE_DEVICES="$GPU_SPEC"
    NUM_GPUS=$(echo "$GPU_SPEC" | tr ',' '\n' | wc -l)
fi

echo "Running get_lvd_embedding.py on GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"

echo "Model Path: $MODEL_PATH"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    get_lvd_embedding.py \
    --file_path "$LVD_PATH" \
    --model_path "$MODEL_PATH"
    


