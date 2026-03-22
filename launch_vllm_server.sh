#!/bin/bash
# vLLM Server Launch Script for MCS-SQL
# 
# This script starts a vLLM API server with tensor parallelism for running
# large models (120B+) across multiple GPUs.
#
# Usage:
#   ./launch_vllm_server.sh [model_name] [tensor_parallel_size] [port]
#
# Examples:
#   ./launch_vllm_server.sh openai/gpt-oss-120b 4 8000
#   ./launch_vllm_server.sh openai/gpt-oss-20b 1 8000

# Default configuration
MODEL_NAME=${1:-"openai/gpt-oss-120b"}
TENSOR_PARALLEL_SIZE=${2:-4}
PORT=${3:-8000}

# vLLM performance settings
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=32768

echo "============================================================"
echo "vLLM Server Launch"
echo "============================================================"
echo "  Model:                $MODEL_NAME"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Port:                 $PORT"
echo "  GPU Memory Util:      $GPU_MEMORY_UTILIZATION"
echo "  Max Model Length:     $MAX_MODEL_LEN"
echo "  Max Batched Tokens:   $MAX_NUM_BATCHED_TOKENS"
echo "============================================================"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Count available GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo ""
echo "Available GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]; then
    echo "ERROR: Requested $TENSOR_PARALLEL_SIZE GPUs but only $GPU_COUNT available"
    exit 1
fi

# Set environment variables for better performance
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TEST_FORCE_OPENSOURCE=1

# Launch vLLM server
echo ""
echo "Starting vLLM API server..."
echo "Press Ctrl+C to stop"
echo ""

python -m vllm.entrypoints.api_server \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --port "$PORT" \
    --host "0.0.0.0" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --dtype "bfloat16" \
    --kv-cache-dtype "auto" \
    --enforce-eager "False" \
    --trust-remote-code

# Alternative: Run in background
# python -m vllm.entrypoints.api_server \
#     --model "$MODEL_NAME" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --port "$PORT" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --enable-chunked-prefill \
#     &
# 
# echo $! > vllm_server.pid
# echo "Server started with PID $(cat vllm_server.pid)"
# echo "To stop: kill $(cat vllm_server.pid)"
