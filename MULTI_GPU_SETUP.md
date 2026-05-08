# Multi-GPU Setup for GPT-OSS 120B on 4xA100

## Overview

This guide explains how to run the MCS-SQL benchmark on `openai/gpt-oss-120b` using 4 A100 GPUs (80 GB each) with vLLM tensor parallelism.

The important distinction is:

- `gpt-oss-120b` is one model
- the model is split across 4 GPUs
- this is not 4 independent copies of the model

## Hardware Requirements

- 4 NVIDIA A100 GPUs with 80 GB VRAM each
- CUDA 11.8+ and PyTorch 2.0+
- Roughly 280-300 GB total usable VRAM for the full 120B setup

## Model Configuration

Create or update `.env` in the project root:

```bash
# Model Configuration
LLM_MODEL_NAME=openai/gpt-oss-120b

# vLLM Configuration
VLLM_ENABLED=true
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_GPU_MEMORY_UTILIZATION=0.95
VLLM_MAX_MODEL_LEN=8192
VLLM_MAX_TOKENS=2048
VLLM_TEMPERATURE=0.3
```

## Recommended Run Flow

### 1. Smoke test

```bash
python engine/test_vllm_single.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/test_vllm_120b \
    --tensor-parallel-size 4
```

### 2. Full benchmark

```bash
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --tensor-parallel-size 4
```

## Architecture

### 4xA100 Setup with GPT-OSS 120B

```text
4x A100 GPUs (80GB each)
┌──────────────────────────────────────────────────────────────────────────┐
│                      1 GPT-OSS 120B model across 4 GPUs                 │
├──────────────────────────────────────────────────────────────────────────┤
│ GPU 0        GPU 1        GPU 2        GPU 3                            │
│ tensor-parallel shards cooperate on every question                      │
│ vLLM handles batching, KV cache, and generation                         │
└──────────────────────────────────────────────────────────────────────────┘
```

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| Model | openai/gpt-oss-120b |
| Parameters | 120 billion |
| GPUs | 4 |
| Tensor parallel size | 4 |
| VRAM per GPU | ~70-75 GB |
| Batch size | 8 |
| Benchmark | MINIDEV |

## Performance Expectations

The 120B model will be slower than smaller models, but it is the quality-first option for final MINIDEV runs.

| Configuration | Tensor Parallelism | Time per Question | 500 Questions |
|---------------|--------------------|------------------|---------------|
| GPT-OSS 120B | 4 GPUs | ~2-4 min | ~17-34 hours |

## Output Files

After completion:

```text
outputs/benchmark_vllm_120b/
├── gpu_0/
├── gpu_1/
├── gpu_2/
├── gpu_3/
├── benchmark_results_merged.json
└── detailed_report.json
```

## Troubleshooting

- Lower `VLLM_GPU_MEMORY_UTILIZATION` if you hit OOM
- Confirm all 4 GPUs appear in `nvidia-smi`
- Restart the run if a GPU process hangs

## Quick Reference

### Command for 4xA100 with GPT-OSS 120B

```bash
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --tensor-parallel-size 4
```
