# Benchmark Resume Guide

This guide explains how to stop, resume, and merge benchmark runs across multiple GPUs.

## Quick Reference

### Check Current Progress
```bash
python engine/check_progress.py outputs/benchmark/start_0_end_125
```

### Merge Results (after completion)
```bash
python engine/merge_benchmark_results.py outputs/benchmark/start_0_end_125
```

---

## Command Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--benchmark` | required | Path to mini_dev_sqlite.json |
| `--db_root` | required | Path to databases directory |
| `--output` | optional | Output directory (default: `outputs/benchmark_results`) |
| `--start` | optional | **Start index (inclusive)** - where to begin processing |
| `--end` | optional | **End index (exclusive)** - where to stop processing |
| `--multi-gpu` | flag | Enable multi-GPU mode |
| `--num-gpus` | int | Number of GPUs to use (default: 4) |

---

## Key Changes to run_benchmark.py

### 1. Index-Based Arguments (Not Count-Based)

**Before:** `--limit 100` = process 100 questions
**After:** `--end 100` = process questions 0-99 (100 questions)

### 2. Automatic Output Directory Naming

When using `--start` or `--end`, results are saved in a subdirectory:
```
outputs/benchmark/start_50_end_125/
├── gpu_0/
├── gpu_1/
├── benchmark_results_merged.json
└── detailed_report.json
```

### 3. Append Mode

Each GPU worker:
- Loads existing `benchmark_results.json` if it exists
- Appends new results to the existing list
- Saves after each question

This allows safe resumption of interrupted runs.

---

## Usage Examples

### Example 1: Fresh Run (Questions 0-124 on 2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1 python engine/run_benchmark.py \
    --benchmark /path/to/mini_dev_sqlite.json \
    --db_root /path/to/databases \
    --output outputs/benchmark \
    --end 125 \
    --multi-gpu --num-gpus 2
```

**Output:** `outputs/benchmark/start_0_end_125/`

---

### Example 2: Resume From Question 62

```bash
CUDA_VISIBLE_DEVICES=0,1 python engine/run_benchmark.py \
    --benchmark /path/to/mini_dev_sqlite.json \
    --db_root /path/to/databases \
    --output outputs/benchmark \
    --start 62 --end 125 \
    --multi-gpu --num-gpus 2
```

**Output:** `outputs/benchmark/start_62_end_125/`

---

### Example 3: Single GPU for Specific Range

```bash
CUDA_VISIBLE_DEVICES=0 python engine/run_benchmark.py \
    --benchmark /path/to/mini_dev_sqlite.json \
    --db_root /path/to/databases \
    --output outputs/benchmark_gpu0 \
    --start 0 --end 63 \
    --multi-gpu --num-gpus 1
```

---

### Example 4: Check Progress Before Stopping

```bash
# Check how many questions completed
python engine/check_progress.py outputs/benchmark/start_0_end_125

# Output example:
# ======================================================================
# BENCHMARK PROGRESS CHECK: outputs/benchmark/start_0_end_125
# ======================================================================
#
#   gpu_0: 31 questions (IDs: 0 - 30)
#   gpu_1: 31 questions (IDs: 31 - 61)
#
#   ==================================================
#   TOTAL: 62 questions completed
#   ==================================================
#
# ======================================================================
# TO RESUME - Use these start indices:
# ======================================================================
#
#   Highest question ID completed: 61
#   Next question to process: 62
#
#   To resume from next question:
#     --start 62
```

---

### Example 5: Merge Results After Completion

```bash
python engine/merge_benchmark_results.py outputs/benchmark/start_0_end_125

# Output:
# Found 2 GPU directories: ['gpu_0', 'gpu_1']
#   gpu_0: 62 results
#   gpu_1: 63 results
#
# Total merged results: 125
# Results sorted by question_id
#
# Merged results saved to: outputs/benchmark/start_0_end_125/benchmark_results_merged.json
#
# ======================================================================
# DETAILED BENCHMARK REPORT
# ======================================================================
# 📊 OVERALL STATISTICS
#   Total Questions:     125
#   Correct:             87 (69.60%)
#   ...
```

---

## Workflow: Stop, Git Pull, Resume

### Step 1: Check Progress
```bash
python engine/check_progress.py outputs/benchmark/start_0_end_125
```

Note the highest question ID completed (e.g., 61).

### Step 2: Stop Current Run
Press `Ctrl+C` in the terminal running the benchmark.

### Step 3: Git Pull
```bash
cd /path/to/MCS-SQL
git pull
```

### Step 4: Resume
```bash
CUDA_VISIBLE_DEVICES=0,1 python engine/run_benchmark.py \
    --benchmark /path/to/mini_dev_sqlite.json \
    --db_root /path/to/databases \
    --output outputs/benchmark \
    --start 62 --end 125 \
    --multi-gpu --num-gpus 2
```

### Step 5: Merge (After Completion)
```bash
python engine/merge_benchmark_results.py outputs/benchmark/start_62_end_125
```

---

## File Structure

```
outputs/benchmark/
├── start_0_end_125/           # First run (questions 0-124)
│   ├── gpu_0/
│   │   └── benchmark_results.json    # Questions assigned to GPU 0
│   ├── gpu_1/
│   │   └── benchmark_results.json    # Questions assigned to GPU 1
│   ├── benchmark_results_merged.json # ← Merged results
│   └── detailed_report.json          # ← Statistics
│
├── start_62_end_125/          # Resumed run (questions 62-124)
│   ├── gpu_0/
│   │   └── benchmark_results.json
│   ├── gpu_1/
│   │   └── benchmark_results.json
│   ├── benchmark_results_merged.json
│   └── detailed_report.json
│
└── manual_gpu0/               # Manual single-GPU run
    └── gpu_0/
        └── benchmark_results.json
```

---

## How Question Indices Work

The benchmark JSON is a list of questions:
```json
[
  {"question_id": 0, "question": "...", ...},
  {"question_id": 1, "question": "...", ...},
  ...
  {"question_id": 124, "question": "...", ...}
]
```

- `--start 50` = Skip questions 0-49, start from question at index 50
- `--end 125` = Stop before question at index 125 (process up to 124)

**Distribution across GPUs:**
```
Total: 125 questions (0-124)
2 GPUs:
  - GPU 0: questions 0-62 (63 questions)
  - GPU 1: questions 63-124 (62 questions)
```

---

## Troubleshooting

### Q: How do I know which questions were completed?
```bash
python engine/check_progress.py outputs/benchmark/start_0_end_125 0 20
```
Shows first 20 question IDs from each GPU.

### Q: Can I run GPUs separately?
Yes! Run one terminal per GPU:
```bash
# Terminal 1 - GPU 0, questions 0-62
CUDA_VISIBLE_DEVICES=0 python engine/run_benchmark.py \
    --benchmark ... --db_root ... \
    --output outputs/manual_0 --start 0 --end 63

# Terminal 2 - GPU 1, questions 63-124
CUDA_VISIBLE_DEVICES=1 python engine/run_benchmark.py \
    --benchmark ... --db_root ... \
    --output outputs/manual_1 --start 63 --end 125
```

Then merge manually:
```bash
# Copy results to single directory
mkdir outputs/merged
cp -r outputs/manual_0/gpu_0 outputs/merged/
cp -r outputs/manual_1/gpu_0 outputs/merged/gpu_1/
python engine/merge_benchmark_results.py outputs/merged
```

### Q: What if a GPU crashes mid-question?
The question won't be in the results file. Next run will re-process it (which is fine - results are independent per question).

### Q: Can I change GPU count between runs?
Yes! The script auto-distributes questions based on available GPUs:
```bash
# First run: 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python ... --num-gpus 4

# Resume: 2 GPUs (maybe 2 failed)
CUDA_VISIBLE_DEVICES=0,1 python ... --start 62 --end 125 --num-gpus 2
```

---

## Scripts Summary

| Script | Purpose |
|--------|---------|
| `run_benchmark.py` | Main benchmark runner |
| `merge_benchmark_results.py` | Merge GPU results into single file |
| `check_progress.py` | Check completion status |

All scripts support both multi-GPU and single-GPU modes.
