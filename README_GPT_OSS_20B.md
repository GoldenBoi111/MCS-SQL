# MCS-SQL Benchmark with GPT-OSS 20B on 4×A100

## Quick Start

### Prerequisites

- 4× NVIDIA A100 GPUs (80 GB VRAM each)
- CUDA 11.8+ and PyTorch 2.0+
- Hugging Face account with access to `openai/gpt-oss-20b`

### Setup (5 minutes)

```bash
# 1. Clone repository
cd /path/to/MCS-SQL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure for GPT-OSS 20B
cp .env.example .env

# 4. Verify configuration
grep LLM_MODEL_NAME .env
# Should show: LLM_MODEL_NAME=openai/gpt-oss-20b
```

### Run Benchmark

```bash
# Test run (1 question, verifies setup)
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --limit 1 \
    --multi-gpu \
    --num-gpus 4

# Full benchmark (500 questions, ~12-17 hours)
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_gpt_oss_20b \
    --multi-gpu \
    --num-gpus 4
```

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| **Model** | openai/gpt-oss-20b |
| **GPUs** | 4× A100 (80 GB each) |
| **Model Copies** | 4 total (1 per GPU) |
| **VRAM Usage** | ~42 GB per GPU |
| **Batch Size** | 8 |
| **Time (500 Qs)** | ~12-17 hours |
| **Time (per Q)** | ~1.5-2 minutes |

## Architecture

```
4× A100 GPUs (80GB each)
┌─────────────────────────────────────────────────┐
│  GPU 0    GPU 1    GPU 2    GPU 3              │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐            │
│  │ 20B │  │ 20B │  │ 20B │  │ 20B │            │
│  │ ×1  │  │ ×1  │  │ ×1  │  │ ×1  │            │
│  │ 125Q│  │ 125Q│  │ 125Q│  │ 125Q│            │
│  └─────┘  └─────┘  └─────┘  └─────┘            │
│     ↓        ↓        ↓        ↓                │
│     └────────┴────────┴────────┘                │
│                  ↓                               │
│        Merged Results (500Q)                     │
└─────────────────────────────────────────────────┘
```

## Performance

### GPT-OSS 20B on 4×A100

| Questions | Time | Speedup |
|-----------|------|---------|
| 1 | ~2 min | - |
| 10 | ~15-20 min | - |
| 100 | ~2.5-3 hours | - |
| 500 | ~12-17 hours | ~4x vs 1 GPU |

### Comparison with Other Models

| Model | GPUs | Time (500Q) | Accuracy |
|-------|------|-------------|----------|
| **GPT-OSS 20B** | 4 | ~12-17 hours | **Highest** |
| Qwen-7B | 4 | ~6 hours | Lower |
| Qwen-14B | 4 | ~8-10 hours | Medium |
| Qwen-32B | 4 | ~17-25 hours | High |

## Monitoring

### Watch GPU Usage

```bash
watch -n 1 nvidia-smi
```

**Expected:**
- ~42 GB VRAM per GPU (52% usage)
- ~95% GPU utilization
- All 4 GPUs active

### Check Progress

```bash
# Count completed questions
jq 'length' outputs/benchmark_gpt_oss_20b/benchmark_results_merged.json

# Check accuracy so far
jq '[.[] | select(.is_correct == true)] | length / length * 100' \
    outputs/benchmark_gpt_oss_20b/benchmark_results_merged.json
```

## Output Files

After completion:

```
outputs/benchmark_gpt_oss_20b/
├── gpu_0/benchmark_results.json    (Q1-125)
├── gpu_1/benchmark_results.json    (Q126-250)
├── gpu_2/benchmark_results.json    (Q251-375)
├── gpu_3/benchmark_results.json    (Q376-500)
├── benchmark_results_merged.json   (All 500)
└── detailed_report.json            (Statistics)
```

## View Results

```bash
# Overall accuracy
jq '.overall' outputs/benchmark_gpt_oss_20b/detailed_report.json

# By difficulty
jq '.by_difficulty' outputs/benchmark_gpt_oss_20b/detailed_report.json

# By database
jq '.by_database' outputs/benchmark_gpt_oss_20b/detailed_report.json

# Execution times
jq '.execution_times' outputs/benchmark_gpt_oss_20b/detailed_report.json
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in .env
echo "BATCH_SIZE=4" >> .env

# Or reduce max tokens
echo "LLM_MAX_NEW_TOKENS=256" >> .env
```

### Model Download Failed

```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub
huggingface-cli download openai/gpt-oss-20b
```

### Process Hung

```bash
# Kill all Python processes
pkill -f python

# Reset GPUs
nvidia-smi --gpu-reset

# Restart with fewer GPUs
python engine/run_benchmark.py --num-gpus 2 ...
```

## Long-Running Sessions

Use `tmux` or `screen` for multi-hour runs:

```bash
# Start tmux session
tmux new -s mcs-benchmark

# Set memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Run benchmark
cd /path/to/MCS-SQL
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_gpt_oss_20b \
    --multi-gpu \
    --num-gpus 4

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t mcs-benchmark
```

## Documentation

- **MULTI_GPU_SETUP.md** - Detailed multi-GPU setup guide
- **DETAILED_REPORT_GUIDE.md** - Understanding benchmark results
- **CONFIG.md** - Configuration options

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review MULTI_GPU_SETUP.md
3. Check GPU logs: `dmesg | grep -i nvidia`

---

**Ready to run?** Execute the Quick Start commands above! 🚀
