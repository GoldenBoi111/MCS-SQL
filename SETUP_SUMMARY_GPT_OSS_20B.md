# Setup Summary: GPT-OSS 20B on 4×A100 GPUs

## Everything Configured for GPT-OSS 20B

All files have been updated to run the MCS-SQL benchmark with **GPT-OSS 20B** on **4 A100 GPUs**.

---

## Quick Command Reference

### Test Run (Verify Setup)
```bash
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --limit 1 \
    --multi-gpu \
    --num-gpus 4
```

### Full Benchmark (500 Questions)
```bash
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_gpt_oss_20b \
    --multi-gpu \
    --num-gpus 4
```

---

## Configuration Files Updated

### 1. `.env.example`
```bash
LLM_MODEL_NAME=openai/gpt-oss-20b
MODEL_COPIES_PER_GPU=1
BATCH_SIZE=8
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.3
```

### 2. `engine/run_benchmark.py`
- Auto-detects gpt-oss-20b → uses 1 copy per GPU
- Passes GPU ID to MultiModelManager
- Generates detailed report with per-database breakdowns

### 3. `engine/schema_linking.py`
- Added `gpu_id` parameter to TransformersLLMClient
- Models load on specific GPU via `device_map=cuda:{gpu_id}`
- Schema linking uses parallel batch generation per GPU

### 4. `MULTI_GPU_SETUP.md`
- Complete guide for GPT-OSS 20B on 4×A100
- Architecture diagrams
- Performance expectations
- Troubleshooting

### 5. `DETAILED_REPORT_GUIDE.md`
- Updated with GPT-OSS 20B references
- Example commands for 4-GPU setup
- Expected runtime: ~12-17 hours

### 6. `README_GPT_OSS_20B.md` (NEW)
- Quick start guide
- Configuration summary
- Monitoring commands
- Troubleshooting

---

## Expected Performance

### Resource Usage

| Resource | Usage |
|----------|-------|
| **VRAM per GPU** | ~42 GB (52%) |
| **GPU Utilization** | ~95% |
| **Total VRAM** | ~168 GB (4×42 GB) |
| **System RAM** | ~4-8 GB (FAISS indexes) |

### Performance

| Metric | Value |
|--------|-------|
| **Time per Question** | ~1.5-2 minutes |
| **Time for 500 Questions** | ~12-17 hours |
| **Speedup vs 1 GPU** | ~4x |
| **SQL Generation Rate** | ~100 SQLs / 2 min = 50 SQL/min |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    4× A100 GPUs (80GB)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Process 0      Process 1      Process 2      Process 3     │
│  GPU 0          GPU 1          GPU 2          GPU 3          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐      │
│  │gpt-oss  │   │gpt-oss  │   │gpt-oss  │   │gpt-oss  │      │
│  │20B      │   │20B      │   │20B      │   │20B      │      │
│  │         │   │         │   │         │   │         │      │
│  │Schema   │   │Schema   │   │Schema   │   │Schema   │      │
│  │Linking  │   │Linking  │   │Linking  │   │Linking  │      │
│  │  ↓      │   │  ↓      │   │  ↓      │   │  ↓      │      │
│  │SQL Gen  │   │SQL Gen  │   │SQL Gen  │   │SQL Gen  │      │
│  │  ↓      │   │  ↓      │   │  ↓      │   │  ↓      │      │
│  │SQL Sel  │   │SQL Sel  │   │SQL Sel  │   │SQL Sel  │      │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘      │
│       ↓               ↓               ↓               ↓      │
│       └───────────────┴───────────────┴───────────────┘      │
│                              ↓                                │
│                    Merge Results                              │
│                    Generate Report                            │
└─────────────────────────────────────────────────────────────┘
```

**Per GPU:**
- 1 model copy (gpt-oss-20b)
- 125 questions
- ~42 GB VRAM
- Independent processing

---

## What Happens During Execution

### Phase 1: Setup (~2-3 minutes)
1. Load 4 GPUs
2. Distribute questions (125 per GPU)
3. Load gpt-oss-20b on each GPU (1 copy each)
4. Load FAISS indexes

### Phase 2: Processing (~12-17 hours)
**Per Question (per GPU):**

1. **Schema Linking** (~1-2 min)
   - 60 table linking prompts → batch generation (batch size 8)
   - 60 column linking prompts → batch generation (batch size 8)
   - Uses parallel generation across model copy

2. **SQL Generation** (~2-3 min)
   - 100 prompts (5 variations × 20 generations)
   - Batch generation (batch size 8)
   - Execute all 100 SQL candidates

3. **SQL Selection** (~0.5 min)
   - Present top candidates to LLM
   - 20 samples with majority voting
   - Select final SQL

4. **Save Results**
   - Append to benchmark_results.json
   - Includes execution times, confidence, correctness

### Phase 3: Merge & Report (~1-2 minutes)
1. Merge results from all 4 GPUs
2. Generate detailed_report.json
3. Print comprehensive statistics

---

## Detailed Report Contents

The final report includes:

### 📊 Overall Statistics
- Total questions, accuracy, correct/incorrect

### ⏱️ Execution Times
- Mean, median, min, max, std dev
- All 50,000 SQL executions (100 × 500)

### 🔧 Generation Statistics
- Total generated, success rate, errors
- Unique SQL variations

### 📈 Confidence Statistics
- Mean/median confidence
- High vs low confidence distribution

### 📚 Difficulty Breakdown
- Accuracy: simple / moderate / challenging

### 🗄️ Per-Database Accuracy
- Ranking by database (sorted by accuracy)
- Correct/total for each

### 🗄️ Per-Database × Difficulty
- Granular breakdown per database AND difficulty
- Average execution times

### 🎯 Selection Phase
- How often LLM selection used
- vs majority vote

---

## Monitoring Commands

### Real-Time GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Progress Tracking
```bash
# Questions completed
watch -n 60 'jq "length" outputs/benchmark_gpt_oss_20b/benchmark_results_merged.json'

# Current accuracy
watch -n 60 'jq "[.[] | select(.is_correct == true)] | length / length * 100" outputs/benchmark_gpt_oss_20b/benchmark_results_merged.json'
```

### Check Specific Database
```bash
jq '[.[] | select(.db_id == "car_retails")]' outputs/benchmark_gpt_oss_20b/benchmark_results_merged.json
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `.env.example` | Configuration template (copy to `.env`) |
| `engine/run_benchmark.py` | Main benchmark script (multi-GPU support) |
| `engine/schema_linking.py` | Schema linking with GPU-specific loading |
| `MULTI_GPU_SETUP.md` | Complete multi-GPU guide |
| `DETAILED_REPORT_GUIDE.md` | Understanding results |
| `README_GPT_OSS_20B.md` | Quick start for GPT-OSS 20B |
| `SETUP_SUMMARY_GPT_OSS_20B.md` | This file |

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce BATCH_SIZE to 4 in `.env` |
| Slow Performance | Check GPU utilization with `nvidia-smi` |
| Process Hung | `pkill -f python && nvidia-smi --gpu-reset` |
| Model Download Failed | `rm -rf ~/.cache/huggingface/hub` and retry |

---

## Next Steps

1. **Create `.env` file:**
   ```bash
   cp .env.example .env
   ```

2. **Verify configuration:**
   ```bash
   grep LLM_MODEL_NAME .env
   # Should show: openai/gpt-oss-20b
   ```

3. **Run test (1 question):**
   ```bash
   python engine/run_benchmark.py --limit 1 --multi-gpu --num-gpus 4
   ```

4. **Run full benchmark (500 questions):**
   ```bash
   python engine/run_benchmark.py --multi-gpu --num-gpus 4
   ```

5. **View results:**
   ```bash
   jq '.overall' outputs/benchmark_gpt_oss_20b/detailed_report.json
   ```

---

## Expected Output Example

```
====================================================================================================
                              DETAILED BENCHMARK REPORT
====================================================================================================

OVERALL STATISTICS
----------------------------------------------------------------------------------------------------
  Total Questions:     500
  Correct:             340 (68.00%)
  Incorrect:           160 (32.00%)

EXECUTION TIME STATISTICS (All SQL Queries)
----------------------------------------------------------------------------------------------------
  Total Queries Executed:  50,000
  Mean Execution Time:     11.23 ms
  Median Execution Time:   7.89 ms
  ...

🗄️  ACCURACY BY DATABASE
----------------------------------------------------------------------------------------------------
  Database                                  Correct    Total    Accuracy
  ----------------------------------------------------------------------
  car_retails                                    45       50      90.00%
  debit_card_specializing                        42       50      84.00%
  ...

💾 Detailed report saved to: outputs/benchmark_gpt_oss_20b/detailed_report.json
====================================================================================================
```

---

**Everything is ready! Run the Quick Start commands to begin.** 🚀
