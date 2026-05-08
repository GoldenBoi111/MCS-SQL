# Detailed Benchmark Report Features

## Overview

The MCS-SQL benchmark now generates comprehensive reports with detailed statistics for analysis.

## Report Sections

### 1. OVERALL STATISTICS

```
OVERALL STATISTICS
----------------------------------------------------------------------------------------------------
  Total Questions:     500
  Correct:             325 (65.00%)
  Incorrect:           175 (35.00%)
```

**What it shows:**

- Total number of questions processed
- Overall execution accuracy (EX metric)
- Correct vs incorrect breakdown

---

### 2. ⏱️ EXECUTION TIME STATISTICS

```
⏱️  EXECUTION TIME STATISTICS (All SQL Queries)
----------------------------------------------------------------------------------------------------
  Total Queries Executed:  50,000
  Mean Execution Time:     12.45 ms
  Median Execution Time:   8.32 ms
  Std Dev:                 15.67 ms
  Min Execution Time:      0.52 ms
  Max Execution Time:      245.89 ms
```

**What it shows:**

- Total SQL queries executed (100 per question × 500 questions = 50,000)
- Average query execution speed
- Distribution of execution times
- Identifies slow-running queries

**Use case:** Identify performance bottlenecks in database schemas or complex queries

---

### 3. 🔧 GENERATION STATISTICS

```
🔧 GENERATION STATISTICS
----------------------------------------------------------------------------------------------------
  Total SQL Generated:     50,000
  Valid Executions:        47,500 (95.0% success rate)
  Execution Errors:        2,500 (5.0%)
  Unique SQL Variations:   1,234 (avg 2.5 per question)
```

**What it shows:**

- Total SQL candidates generated
- Success rate of SQL execution
- Error rate (syntax errors, timeouts)
- Diversity of generated SQL (unique variations)

**Use case:** Measure model's ability to generate syntactically correct SQL

---

### 4. 📈 CONFIDENCE STATISTICS

```
📈 CONFIDENCE STATISTICS (Majority Voting)
----------------------------------------------------------------------------------------------------
  Mean Confidence:         0.623
  Median Confidence:       0.670
  High Confidence (>0.5):  380 (76.0%)
  Low Confidence (<0.2):   45 (9.0%)
```

**What it shows:**

- Average confidence from majority voting
- Distribution of confidence scores
- How often the model is "confident" vs "uncertain"

**Use case:** Understand when the model is uncertain - low confidence may indicate ambiguous questions

---

### 5. 📚 ACCURACY BY DIFFICULTY

```
📚 ACCURACY BY DIFFICULTY
----------------------------------------------------------------------------------------------------
  Simple          280/350 ( 80.00%)
  Moderate         35/100 ( 35.00%)
  Challenging       8/40  ( 20.00%)
  Unknown           2/10  ( 20.00%)
```

**What it shows:**

- Accuracy breakdown by question difficulty
- Number of questions at each difficulty level
- Performance gap between easy and hard questions

**Use case:** Identify which difficulty levels need improvement

---

### 6. 🗄️ ACCURACY BY DATABASE

```
🗄️  ACCURACY BY DATABASE
----------------------------------------------------------------------------------------------------
  Database                                  Correct    Total    Accuracy
  ----------------------------------------------------------------------
  car_retails                                    45       50      90.00%
  debit_card_specializing                        42       50      84.00%
  financial                                     120      150      80.00%
  california_schools                             38       50      76.00%
  ...
```

**What it shows:**

- Per-database accuracy ranking
- Which databases the model performs best/worst on
- Distribution of questions across databases

**Use case:** Identify domain-specific weaknesses (e.g., model struggles with financial data)

---

### 7. 🗄️ PER-DATABASE BREAKDOWN BY DIFFICULTY

```
🗄️  PER-DATABASE BREAKDOWN BY DIFFICULTY
----------------------------------------------------------------------------------------------------

  📁 car_retails (50 questions)
  ----------------------------------------------------------------------
    Difficulty         Correct    Total    Accuracy   Avg Exec Time
    ----------------------------------------------------------------------
    Simple                  35       40      87.50%        8.45 ms
    Moderate                 8       10      80.00%       15.32 ms
    Challenging              2        5      40.00%       45.67 ms

  📁 debit_card_specializing (50 questions)
  ----------------------------------------------------------------------
    Difficulty         Correct    Total    Accuracy   Avg Exec Time
    ----------------------------------------------------------------------
    Simple                  30       35      85.71%        9.12 ms
    Moderate                 7       10      70.00%       18.45 ms
    Challenging              3        5      60.00%       52.34 ms
```

**What it shows:**

- Detailed breakdown per database AND difficulty
- Execution times per difficulty level
- Granular performance analysis

**Use case:** Deep dive into specific database + difficulty combinations

---

### 8. 🎯 SQL SELECTION PHASE STATISTICS

```
🎯 SQL SELECTION PHASE STATISTICS
----------------------------------------------------------------------------------------------------
  Selection Made:          420 (84.0%)
  Majority Vote Used:       80 (16.0%)
```

**What it shows:**

- How often the LLM selection phase was used
- How often majority vote winner was kept
- Impact of the selection phase

**Use case:** Understand the value added by the LLM selection phase

---

## Output Files

### 1. `benchmark_results_merged.json`

Complete results for every question including:

- Question text and ID
- Database ID
- Ground truth SQL
- Generated SQL (winner)
- Correctness flag
- Confidence score
- Selection phase details
- All metrics

### 2. `detailed_report.json`

Aggregated statistics in machine-readable format:

```json
{
  "overall": {
    "total": 500,
    "correct": 325,
    "accuracy": 65.0
  },
  "by_difficulty": {
    "simple": {"correct": 280, "total": 350, "accuracy": 80.0},
    "moderate": {"correct": 35, "total": 100, "accuracy": 35.0},
    ...
  },
  "by_database": {
    "car_retails": {"total": 50, "correct": 45, "accuracy": 90.0},
    ...
  },
  "execution_times": {
    "mean_ms": 12.45,
    "median_ms": 8.32,
    "min_ms": 0.52,
    "max_ms": 245.89,
    "std_ms": 15.67
  },
  "generation": {
    "total_generated": 50000,
    "valid_executions": 47500,
    "errors": 2500,
    "unique_sqls": 1234
  },
  "confidence": {
    "mean": 0.623,
    "median": 0.670,
    "high_confidence_ratio": 0.76
  }
}
```

---

## Example Analysis Workflows

### Find Weakest Databases

```python
import json

with open('outputs/detailed_report.json') as f:
    report = json.load(f)

# Sort databases by accuracy (ascending)
db_accuracy = [(db, data['accuracy']) for db, data in report['by_database'].items()]
db_accuracy.sort(key=lambda x: x[1])

print("Weakest databases:")
for db, acc in db_accuracy[:5]:
    print(f"  {db}: {acc:.2f}%")
```

### Analyze Error Patterns

```python
with open('outputs/benchmark_results_merged.json') as f:
    results = json.load(f)

# Find all incorrect results
errors = [r for r in results if not r['is_correct']]

# Group by database
from collections import defaultdict
db_errors = defaultdict(list)
for r in errors:
    db_errors[r['db_id']].append(r)

print("Error distribution:")
for db, err_list in db_errors.items():
    print(f"  {db}: {len(err_list)} errors")
```

### Correlation: Confidence vs Correctness

```python
with open('outputs/benchmark_results_merged.json') as f:
    results = json.load(f)

correct_high_conf = sum(1 for r in results if r['is_correct'] and r['winner_confidence'] > 0.5)
incorrect_high_conf = sum(1 for r in results if not r['is_correct'] and r['winner_confidence'] > 0.5)

print(f"High confidence (>0.5) and correct: {correct_high_conf}")
print(f"High confidence (>0.5) but wrong:   {incorrect_high_conf}")
```

### Execution Time Analysis

```python
import statistics

with open('outputs/detailed_report.json') as f:
    report = json.load(f)

exec_times = report['execution_times']
print(f"Mean: {exec_times['mean_ms']:.2f} ms")
print(f"Median: {exec_times['median_ms']:.2f} ms")
print(f"Std Dev: {exec_times['std_ms']:.2f} ms")

# High std dev indicates some queries are much slower
if exec_times['std_ms'] > exec_times['mean_ms']:
    print("⚠️  High variance in execution times - some queries are very slow")
```

---

## Usage

### Single GPU

```bash
python engine/run_benchmark.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_full

# Outputs:
#   - outputs/benchmark_full/benchmark_results.json
#   - outputs/benchmark_full/detailed_report.json
```

### Multi-GPU (4xA100 with GPT-OSS 120B)

```bash
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --tensor-parallel-size 4
```

Expected runtime depends on prompt length and system load, but the 120B vLLM path is the intended 4-GPU flow here.

```text
# Outputs:
#   - outputs/benchmark_vllm_120b/gpu_0/benchmark_results.json
#   - outputs/benchmark_vllm_120b/gpu_1/benchmark_results.json
#   - outputs/benchmark_vllm_120b/gpu_2/benchmark_results.json
#   - outputs/benchmark_vllm_120b/gpu_3/benchmark_results.json
#   - outputs/benchmark_vllm_120b/benchmark_results_merged.json
#   - outputs/benchmark_vllm_120b/detailed_report.json
```

---

## Performance Metrics Explained

### Execution Accuracy (EX)

- **Definition**: `1` if predicted SQL produces same result as ground truth, `0` otherwise
- **Comparison**: Set-based (order-independent)
- **Metric**: `EX = (correct predictions) / (total predictions)`

### Confidence Score

- **Formula**: `confidence(qi) = count(result_i) / N_valid`
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - `> 0.5`: Strong majority agreement
  - `0.2 - 0.5`: Moderate agreement
  - `< 0.2`: Low agreement (diverse outputs)

### Generation Success Rate

- **Formula**: `(valid executions) / (total generated)`
- **Typical**: 90-98%
- **Low rate indicates**: Syntax errors or invalid SQL

### Unique SQL Variations

- **Definition**: Number of distinct SQL queries generated per question
- **High number**: Model produces diverse solutions
- **Low number**: Model converges to similar SQL

---

## Tips for Analysis

1. **Start with Overall Statistics**: Get high-level accuracy
2. **Check Difficulty Breakdown**: Identify which levels need work
3. **Review Database Performance**: Find domain-specific issues
4. **Analyze Low Confidence**: Questions with confidence < 0.2 often reveal ambiguity
5. **Check Execution Times**: Slow queries may indicate complex joins or missing indexes
6. **Review Error Patterns**: Group errors by database/difficulty to find patterns
