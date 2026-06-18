# export MODEL="Qwen/Qwen3-8B-Base"
# export MODEL="/path/to/your/model_or_checkpoint"

# export MODEL="Qwen/Qwen2.5-Math-7B"
# export MODEL="/path/to/your/model_or_checkpoint"

# export GPUS="4,5,6,7"

export GPUS="$1"
export MODEL="$2"
export NUM_GPU=$(echo $GPUS | awk -F',' '{print NF}')
echo "Number of GPUs: $NUM_GPU"

# export VLLM_LOGGING_LEVEL=DEBUG

CUDA_VISIBLE_DEVICES=$GPUS vllm serve $MODEL \
    --data-parallel-size $NUM_GPU \
    --tensor-parallel-size 1