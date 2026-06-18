#!/bin/bash

# Parse command line arguments
RUN_DEBUG=false
RUN_DEV=true
RUN_LVD=true
RUN_TRAIN=true

# RUN_DEBUG=true
# RUN_DEV=false
# RUN_LVD=false
# RUN_TRAIN=false

# export MODEL_PATH="Qwen/Qwen3-8B-Base"
# export MODEL_NAME="Qwen3-8B-Base"
# export MODEL_PATH="/path/to/your/model_or_checkpoint"
# export MODEL_NAME="DAPO-DAPO-Baseline-V7-S60"

export MODEL_PATH="$1"
export MODEL_NAME="$2"
export INPUT_PATH="$3"
export OUTPUT_DIR="$4"
export IS_VLM="$5"

# Set the script name based on IS_VLM
if [ "$IS_VLM" = "True" ]; then
    SCRIPT_NAME="sample_data_instruct_vllm_img.py"
else
    SCRIPT_NAME="sample_data_instruct_vllm.py"
fi

# DEBUG
# 10*1 = 10
if [ "$RUN_DEBUG" = true ]; then
    echo "Running debug data generation..."
    python3 $SCRIPT_NAME \
        --model_name_or_path $MODEL_PATH \
        --input_file $INPUT_PATH \
        --total_chunks 1 \
        --n 10 \
        --max_data 100 \
        --max_new_tokens 500 \
        --output_file ${OUTPUT_DIR}/${MODEL_NAME}.debug \
        --vllm_url "http://localhost:8000/v1" \
        --vllm_api_key "EMPTY"
    echo "Development data generation completed."
fi

# DEV
# 2000*2 = 4000
if [ "$RUN_DEV" = true ]; then
    echo "Running development data generation..."
    python3 $SCRIPT_NAME \
        --model_name_or_path $MODEL_PATH \
        --input_file $INPUT_PATH \
        --total_chunks 1 \
        --n 2 \
        --max_new_tokens 500 \
        --output_file ${OUTPUT_DIR}/${MODEL_NAME}.dev \
        --vllm_url "http://localhost:8000/v1" \
        --vllm_api_key "EMPTY"
    echo "Development data generation completed."
fi

# LVD
# Important: Make sure you do not shuffle the LVD set!
if [ "$RUN_LVD" = true ]; then
    echo "Running LVD data generation..."
    python3 $SCRIPT_NAME \
        --model_name_or_path $MODEL_PATH \
        --input_file $INPUT_PATH \
        --total_chunks 1 \
        --n 10 \
        --lvd \
        --max_new_tokens 500 \
        --output_file ${OUTPUT_DIR}/${MODEL_NAME}.lvd \
        --vllm_url "http://localhost:8000/v1" \
        --vllm_api_key "EMPTY"
    echo "LVD data generation completed."
fi

# TRAIN - DAPO
if [ "$RUN_TRAIN" = true ]; then
    echo "Running training data generation..."
    python3 $SCRIPT_NAME \
        --model_name_or_path $MODEL_PATH \
        --input_file $INPUT_PATH \
        --total_chunks 100 \
        --n 10 \
        --shuffle \
        --max_new_tokens 500 \
        --output_file ${OUTPUT_DIR}/${MODEL_NAME}.train \
        --vllm_url "http://localhost:8000/v1" \
        --vllm_api_key "EMPTY"
    echo "Training data generation completed."
fi

echo "All requested operations completed."