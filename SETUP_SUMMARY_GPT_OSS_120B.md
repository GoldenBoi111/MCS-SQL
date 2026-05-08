# GPT-OSS 120B MINIDEV Setup Summary

## Goal

Run `openai/gpt-oss-120b` on 4 GPUs for the `MINIDEV` benchmark using vLLM tensor parallelism.

## Minimal Configuration

```ini
LLM_MODEL_NAME=openai/gpt-oss-120b
VLLM_ENABLED=true
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_GPU_MEMORY_UTILIZATION=0.95
VLLM_MAX_MODEL_LEN=8192
VLLM_MAX_TOKENS=512
VLLM_TEMPERATURE=0.3
```

## Recommended Run

```bash
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --tensor-parallel-size 4
```

## Test First

```bash
python engine/test_vllm_single.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/test_vllm_120b \
    --tensor-parallel-size 4
```

## Key Behavior

- One 120B model instance, not 4 separate copies
- All 4 GPUs cooperate on every question
- Results are merged into `benchmark_results_merged.json`

## Files To Know

- `engine/vllm_model_manager.py`
- `engine/run_benchmark_vllm.py`
- `launch_vllm_server.sh`
- `launch_vllm_server.bat`
- `VLLM_SETUP_GUIDE.md`

## Notes

- The benchmark stays on `MINIDEV`
- If you need a smaller memory footprint, reduce `VLLM_GPU_MEMORY_UTILIZATION`
- For a separate API server, launch vLLM with `--tensor-parallel-size 4`
