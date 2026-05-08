# vLLM 120B Quick Start

## Overview

This implementation runs **1 copy of 120B distributed across 4 GPUs** using tensor parallelism. All 4 GPUs work together on each question.

---

## Installation

```bash
pip install vllm openai
```

---

## Test with 1 Question (RECOMMENDED FIRST)

Run this to get timing for extrapolation:

```bash
python engine/test_vllm_single.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/test_vllm_120b \
    --tensor-parallel-size 4
```

**Output will show:**
- Time for 1 question
- Extrapolated time for 500 questions

---

## Full Benchmark (After Testing)

```bash
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --tensor-parallel-size 4
```

## Auto Resume

If a run is interrupted, use the resume wrapper to continue from the first missing question:

```bash
python engine/run_benchmark_vllm_resume.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --tensor-parallel-size 4
```

The script inspects `outputs/benchmark_vllm_120b` and resumes automatically from the next missing index.

---

## Resume After Interruption

```bash
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --start 50 \
    --tensor-parallel-size 4
```

---

## Configuration (.env)

```ini
# Model (change to 120B)
LLM_MODEL_NAME=openai/gpt-oss-120b

# vLLM settings
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_GPU_MEMORY_UTILIZATION=0.95
VLLM_MAX_MODEL_LEN=8192
VLLM_MAX_TOKENS=2048
VLLM_TEMPERATURE=0.3
```

---

## Expected Memory Usage

| GPU | Memory |
|-----|--------|
| GPU 0 | ~70-75 GB |
| GPU 1 | ~70-75 GB |
| GPU 2 | ~70-75 GB |
| GPU 3 | ~70-75 GB |
| **Total** | **~280-300 GB** |

---

## Architecture

```
4× A100 GPUs (80GB each)
┌────────────────────────────────────────────────┐
│        SINGLE 120B MODEL (Tensor Parallel)     │
│  ┌──────────┬──────────┬──────────┬──────────┐ │
│  │ GPU 0    │ GPU 1    │ GPU 2    │ GPU 3    │ │
│  │ Layers   │ Layers   │ Layers   │ Layers   │ │
│  │ 0-29     │ 30-59    │ 60-89    │ 90-119   │ │
│  │ ~70 GB   │ ~70 GB   │ ~70 GB   │ ~70 GB   │ │
│  └──────────┴──────────┴──────────┴──────────┘ │
│                    ↓                            │
│         ALL Questions Sequentially              │
└────────────────────────────────────────────────┘
```

**Key point:** 1 model, not 4 copies. All GPUs collaborate on each question.

---

## Files Created

| File | Purpose |
|------|---------|
| `engine/vllm_model_manager.py` | vLLM integration layer |
| `engine/run_benchmark_vllm.py` | vLLM benchmark runner |
| `engine/test_vllm_single.py` | Single-question timing test |
| `launch_vllm_server.bat` | Windows server launch script |
| `launch_vllm_server.sh` | Linux server launch script |
| `VLLM_SETUP_GUIDE.md` | Full documentation |

---

## Troubleshooting

### Out of Memory

```bash
# Reduce memory utilization
VLLM_GPU_MEMORY_UTILIZATION=0.90

# Reduce max sequence length
VLLM_MAX_MODEL_LEN=4096
```

### Model Won't Load

First load downloads ~240GB. Be patient. Subsequent loads are faster.

### Slow Generation

This is expected for 120B. Use the timing test to extrapolate.

---

## Next Steps

1. **Install vLLM:** `pip install vllm openai`
2. **Update .env:** Set `LLM_MODEL_NAME=openai/gpt-oss-120b`
3. **Run timing test:** `python engine/test_vllm_single.py ...`
4. **Review extrapolated time**
5. **Run full benchmark** if acceptable
