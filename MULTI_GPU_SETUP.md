# Multi-GPU Setup for GPT-OSS 20B on 4×A100

## Overview

This guide explains how to run the MCS-SQL benchmark on **GPT-OSS 20B** using **4 A100 GPUs** (80 GB each) for maximum throughput.

## Hardware Requirements

- **4× NVIDIA A100 GPUs** (80 GB VRAM each)
- **CUDA 11.8+** and **PyTorch 2.0+**
- **~160-180 GB total VRAM** (40-45 GB per GPU for gpt-oss-20b)

## Model Configuration

### For GPT-OSS 20B (openai/gpt-oss-20b)

Create a `.env` file in the project root:

```bash
# Model Configuration
LLM_MODEL_NAME=openai/gpt-oss-20b
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.3

# Multi-GPU Configuration
NUM_GPUS=4
MODEL_COPIES_PER_GPU=1  # 20B model = ~40-45GB, 1 copy per 80GB A100
BATCH_SIZE=8
```

## Architecture

### 4×A100 Setup with GPT-OSS 20B

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         4× A100 GPUs (80GB each)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GPU 0          GPU 1          GPU 2          GPU 3                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                 │
│  │ gpt-oss │   │ gpt-oss │   │ gpt-oss │   │ gpt-oss │                 │
│  │  20B×1  │   │  20B×1  │   │  20B×1  │   │  20B×1  │                 │
│  │ ~42 GB  │   │ ~42 GB  │   │ ~42 GB  │   │ ~42 GB  │                 │
│  │ 125 Qs  │   │ 125 Qs  │   │ 125 Qs  │   │ 125 Qs  │                 │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘                 │
│       ↓             ↓             ↓             ↓                       │
│  (Processing)  (Processing)  (Processing)  (Processing)                 │
│       └────────────┬────────────┴─────────────┘                         │
│                    ↓                                                    │
│         Merged Results (500 questions total)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration Summary

| Parameter | Value |
|-----------|-------|
| **Model** | openai/gpt-oss-20b |
| **Parameters** | 20 billion |
| **VRAM per GPU** | ~40-45 GB |
| **Model Copies** | 4 total (1 per GPU) |
| **Batch Size** | 8 |
| **Questions per GPU** | 125 (for 500 total) |

## Usage

### Quick Start (4 GPUs)

```bash
# 1. Create .env file
cp .env.example .env

# 2. Verify model is set to gpt-oss-20b
grep LLM_MODEL_NAME .env
# Should show: LLM_MODEL_NAME=openai/gpt-oss-20b

# 3. Run benchmark on 4 GPUs
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_gpt_oss_20b \
    --multi-gpu \
    --num-gpus 4
```

### Test Run (First Question Only)

```bash
# Test with just 1 question to verify setup
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/test_gpt_oss_20b \
    --limit 1 \
    --multi-gpu \
    --num-gpus 4
```

### Full Benchmark (500 Questions)

```bash
# Run full benchmark on 500 questions
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_full_500 \
    --multi-gpu \
    --num-gpus 4
```

## Performance Expectations

### GPT-OSS 20B on 4×A100

| Configuration | Model Copies | Time per Question | 500 Questions |
|--------------|--------------|-------------------|---------------|
| 1 GPU (1 copy) | 1 total | ~5-6 min | ~42-50 hours |
| 4 GPUs (4 copies) | 4 total | ~1.5-2 min | **~12-17 hours** |

**Speedup: ~3-4x with 4 GPUs** (near-linear scaling)

### Comparison with Other Models

| Model | GPUs | Model Copies | Time/Question | 500 Questions |
|-------|------|--------------|---------------|---------------|
| **7B** | 4 | 8 total | ~45 sec | ~6 hours |
| **14B** | 4 | 4 total | ~1 min | ~8-10 hours |
| **20B (gpt-oss)** | 4 | 4 total | ~1.5-2 min | ~12-17 hours |
| **32B** | 4 | 4 total | ~2-3 min | ~17-25 hours |

**Note**: GPT-OSS 20B is **slower than 7B** but typically produces **higher quality SQL** (better accuracy).

### Why GPT-OSS 20B is Slower

1. **Fewer model copies**: 4 total vs 8 total for 7B (less parallelism)
2. **Slower generation**: ~30 tok/s vs ~60 tok/s (more parameters)
3. **Half the throughput**: 32 prompts/batch vs 64 prompts/batch for 7B

**Trade-off**: Use 7B for rapid prototyping, GPT-OSS 20B for final evaluation with better accuracy.

## How It Works

### Data Parallelism

1. **Question Distribution**: 500 questions split evenly across 4 GPUs (~125 each)
2. **Independent Processing**: Each GPU runs full benchmark pipeline independently
3. **Schema Linking**: Uses `MultiModelManager` with batch generation on each GPU
4. **SQL Generation**: Uses `MultiModelManager` with batch generation on each GPU
5. **SQL Selection**: Uses `MultiModelManager` with batch generation on each GPU
6. **Result Merging**: Results automatically merged after all GPUs complete

### Per-GPU Processing

Each GPU processes its chunk with:
- **Schema Linking**: 120 calls in parallel batches (batch size 8) using local model copy
- **SQL Generation**: 100 calls in parallel batches (batch size 8) using local model copy
- **SQL Selection**: 20 calls in parallel batches (batch size 8) using local model copy

### Model Loading Per GPU

```
GPU 0 Process (125 questions):
  └─ MultiModelManager (1 gpt-oss-20b copy on GPU 0)
      ├─ Schema Linking: 60 table + 60 column prompts → batch generation
      ├─ SQL Generation: 100 prompts → batch generation
      └─ SQL Selection: 20 prompts → batch generation

GPU 1 Process (125 questions):
  └─ MultiModelManager (1 gpt-oss-20b copy on GPU 1)
      └─ (same pipeline)

GPU 2 Process (125 questions):
  └─ MultiModelManager (1 gpt-oss-20b copy on GPU 2)
      └─ (same pipeline)

GPU 3 Process (125 questions):
  └─ MultiModelManager (1 gpt-oss-20b copy on GPU 3)
      └─ (same pipeline)
```

### Memory Management

- **Automatic GPU selection**: Each process sets `CUDA_VISIBLE_DEVICES`
- **Model placement**: Models loaded on specific GPU via `device_map=cuda:{gpu_id}`
- **Memory cleanup**: `torch.cuda.empty_cache()` between operations
- **Garbage collection**: `gc.collect()` to prevent memory leaks

## Output Structure

```
outputs/benchmark_gpt_oss_20b/
├── gpu_0/
│   └── benchmark_results.json    # Results from GPU 0 (questions 1-125)
├── gpu_1/
│   └── benchmark_results.json    # Results from GPU 1 (questions 126-250)
├── gpu_2/
│   └── benchmark_results.json    # Results from GPU 2 (questions 251-375)
├── gpu_3/
│   └── benchmark_results.json    # Results from GPU 3 (questions 376-500)
├── benchmark_results_merged.json # Combined results from all GPUs
└── detailed_report.json          # Comprehensive statistics
```

## Monitoring

### GPU Utilization

```bash
# Monitor all 4 GPUs in real-time
watch -n 1 nvidia-smi
```

### Expected GPU Usage (GPT-OSS 20B)

```
+-----------------------------------------------------------------------------+
| GPU  Name        | Memory-Usage | GPU-Util | Processes                     |
+------------------+--------------+----------+-------------------------------+
|   0  A100 80GB   | 42000 / 81920 MiB | 95% |   python (benchmark)          |
|   1  A100 80GB   | 42000 / 81920 MiB | 95% |   python (benchmark)          |
|   2  A100 80GB   | 42000 / 81920 MiB | 95% |   python (benchmark)          |
|   3  A100 80GB   | 42000 / 81920 MiB | 95% |   python (benchmark)          |
+-----------------------------------------------------------------------------+
```

**Expected:**
- ~42 GB VRAM usage per GPU (52% of 80 GB)
- ~95% GPU utilization during generation
- All 4 GPUs active simultaneously

## Troubleshooting

### OOM (Out of Memory)

If you see CUDA OOM errors:

1. **Reduce batch size** in `.env`:
   ```bash
   BATCH_SIZE=4  # Reduce from 8 to 4
   ```

2. **Reduce max tokens**:
   ```bash
   LLM_MAX_NEW_TOKENS=256  # Reduce from 512
   ```

3. **Verify only 1 copy per GPU**:
   ```bash
   MODEL_COPIES_PER_GPU=1  # Should be 1 for 20B models
   ```

### Slow Performance

If performance is slower than expected:

1. **Check GPU utilization** with `nvidia-smi` - should be >90%
2. **Ensure all 4 GPUs are being used** (look for 4 python processes)
3. **Check for CPU bottleneck** (data loading, SQL execution)
4. **Verify batch size is 8** (not 4 or lower)

### Process Hangs

If a process hangs:

1. **Kill all Python processes**: `pkill -f python`
2. **Clear GPU memory**: `nvidia-smi --gpu-reset`
3. **Restart with fewer GPUs**: `--num-gpus 2`
4. **Check system logs**: `dmesg | grep -i nvidia`

### Model Download Issues

If the model fails to download:

```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/hub

# Re-download model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('openai/gpt-oss-20b')"
```

## Changing Model

To use a different model, update `.env`:

```bash
# For GPT-OSS 20B (recommended for quality)
LLM_MODEL_NAME=openai/gpt-oss-20b
MODEL_COPIES_PER_GPU=1

# For 7B model (faster, lower quality)
LLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
MODEL_COPIES_PER_GPU=2

# For 14B model (balanced)
LLM_MODEL_NAME=Qwen/Qwen2.5-14B-Instruct
MODEL_COPIES_PER_GPU=1

# For 32B model (slower, best quality)
LLM_MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct
MODEL_COPIES_PER_GPU=1
```

The script auto-detects model size and adjusts copies accordingly.

## Best Practices

1. **Start small**: Test with `--limit 1` first
2. **Monitor GPUs**: Use `nvidia-smi` to watch utilization
3. **Use screen/tmux**: Long-running jobs should be in persistent sessions
4. **Save results frequently**: Results saved after each question
5. **Check logs**: Look for errors in GPU-specific output
6. **Set environment variable** for better memory management:
   ```bash
   export PYTORCH_ALLOC_CONF=expandable_segments:True
   ```

## Example Session

```bash
# Start a tmux session
tmux new -s gpt-oss-benchmark

# Set environment variables
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Navigate to project
cd /path/to/MCS-SQL

# Run benchmark on 4 GPUs
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_gpt_oss_20b \
    --multi-gpu \
    --num-gpus 4

# Detach from tmux (Ctrl+B, then D)
# Check progress later: tmux attach -t gpt-oss-benchmark
```

## Results Interpretation

After completion, check:

1. **Overall Accuracy**: Main metric for benchmark performance
2. **Difficulty Breakdown**: Accuracy by simple/moderate/challenging
3. **Per-Database Accuracy**: Which databases perform best/worst
4. **Execution Times**: Query performance statistics
5. **Error Analysis**: Check `detailed_report.json` for details

```bash
# View detailed report
cat outputs/benchmark_gpt_oss_20b/detailed_report.json | python -m json.tool | head -100

# Check overall accuracy
jq '.overall' outputs/benchmark_gpt_oss_20b/detailed_report.json

# Check per-database accuracy
jq '.by_database' outputs/benchmark_gpt_oss_20b/detailed_report.json

# View questions by accuracy
jq '[.[] | select(.is_correct == true)] | length' outputs/benchmark_gpt_oss_20b/benchmark_results_merged.json
```

## Detailed Report Sections

The benchmark generates a comprehensive report including:

- **📊 Overall Statistics**: Total questions, accuracy, correct/incorrect
- **⏱️ Execution Times**: Mean, median, min, max, std dev
- **🔧 Generation Stats**: Total generated, success rate, errors
- **📈 Confidence Stats**: Mean/median confidence, distribution
- **📚 Difficulty Breakdown**: Accuracy by difficulty level
- **🗄️ Per-Database Accuracy**: Ranking by database
- **🗄️ DB × Difficulty**: Granular breakdown
- **🎯 Selection Phase**: LLM selection vs majority vote

See `DETAILED_REPORT_GUIDE.md` for complete documentation.

---

## Quick Reference

### Command for 4×A100 with GPT-OSS 20B

```bash
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_gpt_oss_20b \
    --multi-gpu \
    --num-gpus 4
```

### Expected Results

- **Time**: ~12-17 hours for 500 questions
- **VRAM**: ~42 GB per GPU
- **Accuracy**: Higher than 7B models (exact % depends on dataset)
- **Speedup**: ~3-4x faster than single GPU

---

**Note**: The index loading (FAISS) is duplicated per GPU process. This is intentional for simplicity and isolation. Total RAM usage will be ~4× index size (~2-4 GB total).
