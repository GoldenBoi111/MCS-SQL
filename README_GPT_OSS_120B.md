# MCS-SQL Benchmark with GPT-OSS 120B on 4 GPUs

## Quick Start

This setup runs `openai/gpt-oss-120b` distributed across 4 GPUs with vLLM tensor parallelism for the `MINIDEV` benchmark.

## Prerequisites

- 4 NVIDIA A100 GPUs with 80 GB VRAM each
- CUDA 11.8+ and PyTorch 2.0+
- Access to `openai/gpt-oss-120b`
- `vllm`, `openai`, and the project dependencies installed

## Setup

```bash
cd /path/to/MCS-SQL
pip install -r requirements.txt
cp .env.example .env
```

Set the model and vLLM values in `.env`:

```ini
LLM_MODEL_NAME=openai/gpt-oss-120b
VLLM_ENABLED=true
VLLM_TENSOR_PARALLEL_SIZE=4
```

## Run MINIDEV

```bash
# Smoke test: 1 question
python engine/test_vllm_single.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/test_vllm_120b \
    --tensor-parallel-size 4

# Full benchmark
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --tensor-parallel-size 4
```

## What This Means

- One 120B model instance is split across 4 GPUs
- Every question uses all 4 GPUs together
- Output is written under `outputs/benchmark_vllm_120b/`

## Output Layout

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

- If loading fails, confirm all 4 GPUs are visible in `nvidia-smi`
- If you hit memory pressure, lower `VLLM_GPU_MEMORY_UTILIZATION` to `0.90`
- For long runs, use `tmux` or `screen`

## Related Files

- `VLLM_120B_QUICKSTART.md`
- `VLLM_SETUP_GUIDE.md`
- `MULTI_GPU_SETUP.md`
