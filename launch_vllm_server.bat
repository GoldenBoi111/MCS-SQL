@echo off
REM vLLM Server Launch Script for Windows (MCS-SQL)
REM 
REM This script starts a vLLM API server with tensor parallelism for running
REM large models (120B+) across multiple GPUs.
REM
REM Usage:
REM   launch_vllm_server.bat [model_name] [tensor_parallel_size] [port]
REM
REM Examples:
REM   launch_vllm_server.bat openai/gpt-oss-120b 4 8000

setlocal enabledelayedexpansion

REM Default configuration
set MODEL_NAME=%~1
if "%MODEL_NAME%"=="" set MODEL_NAME=openai/gpt-oss-120b

set TENSOR_PARALLEL_SIZE=%~2
if "%TENSOR_PARALLEL_SIZE%"=="" set TENSOR_PARALLEL_SIZE=4

set PORT=%~3
if "%PORT%"=="" set PORT=8000

REM vLLM performance settings
set GPU_MEMORY_UTILIZATION=0.95
set MAX_MODEL_LEN=8192
set MAX_NUM_BATCHED_TOKENS=32768

echo ============================================================
echo vLLM Server Launch
echo ============================================================
echo   Model:                %MODEL_NAME%
echo   Tensor Parallel Size: %TENSOR_PARALLEL_SIZE%
echo   Port:                 %PORT%
echo   GPU Memory Util:      %GPU_MEMORY_UTILIZATION%
echo   Max Model Length:     %MAX_MODEL_LEN%
echo   Max Batched Tokens:   %MAX_NUM_BATCHED_TOKENS%
echo ============================================================
echo.

REM Check GPU availability
echo Checking GPU availability...
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo.

REM Launch vLLM server
echo Starting vLLM API server...
echo Press Ctrl+C to stop
echo.

REM Set environment variables for better performance
set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python -m vllm.entrypoints.api_server ^
    --model %MODEL_NAME% ^
    --tensor-parallel-size %TENSOR_PARALLEL_SIZE% ^
    --port %PORT% ^
    --host 0.0.0.0 ^
    --gpu-memory-utilization %GPU_MEMORY_UTILIZATION% ^
    --max-model-len %MAX_MODEL_LEN% ^
    --max-num-batched-tokens %MAX_NUM_BATCHED_TOKENS% ^
    --enable-chunked-prefill ^
    --dtype bfloat16 ^
    --kv-cache-dtype auto ^
    --enforce-eager False ^
    --trust-remote-code
