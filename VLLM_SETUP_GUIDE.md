# MCS-SQL with vLLM: Complete Setup Guide

## Overview

This guide explains how to run MCS-SQL benchmarks with **vLLM** for high-throughput generation, specifically optimized for **120B+ parameter models** on multi-GPU setups.

## Why vLLM?

For your workload (120,000+ LLM calls, 500 questions × 100 SQL candidates each), vLLM provides:

1. **Tensor Parallelism**: Single 120B model distributed across 4 GPUs (vs. 4 separate 20B copies)
2. **PagedAttention**: 2-4× better KV cache memory efficiency
3. **Continuous Batching**: Automatic in-flight batching for 2-3× throughput improvement
4. **Structured Output**: Native JSON schema enforcement (replaces outlines)

### Performance Comparison

| Metric | Transformers (20B × 4) | vLLM (120B TP=4) |
|--------|----------------------|------------------|
| **Runtime (500Q)** | 30-40 hours | **50-80 hours** |
| **Model Quality** | 20B parameters | **120B parameters (6× larger)** |
| **OOM Risk** | Medium (you've experienced) | **Low (PagedAttention)** |
| **Memory Efficiency** | Pre-allocated KV cache | **Paged (2-4× better)** |
| **GPU Utilization** | ~60-70% | **~85-95%** |

---

## Installation

### Step 1: Install vLLM

vLLM requires specific CUDA versions. Choose the appropriate installation command:

**CUDA 12.1 (recommended for A100):**
```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
```

**Standard install (auto-detects CUDA):**
```bash
pip install vllm
```

### Step 2: Install OpenAI Client (for API mode)

```bash
pip install openai
```

### Step 3: Verify Installation

```bash
python -c "from vllm import LLM; print('vLLM installed successfully')"
```

---

## Quick Start: Running 120B on 4×A100

### Option A: Dedicated vLLM Benchmark Script (Recommended)

```bash
# Single command - starts vLLM and runs benchmark
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --tensor-parallel-size 4
```

This automatically:
1. Initializes vLLM with tensor parallelism across 4 GPUs
2. Loads the 120B model (configured in `.env`)
3. Runs the full benchmark pipeline
4. Saves results to `outputs/benchmark_vllm_120b/`

### Option B: Separate vLLM Server + Benchmark Client

**Terminal 1: Start vLLM API Server**

```bash
# Linux/Mac
./launch_vllm_server.sh openai/gpt-oss-120b 4 8000

# Windows
launch_vllm_server.bat openai/gpt-oss-120b 4 8000

# Or manually
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192
```

**Terminal 2: Run Benchmark**

```bash
python engine/run_benchmark_vllm.py \
    --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
    --db_root minidev/MINIDEV/dev_databases/ \
    --output outputs/benchmark_vllm_120b \
    --vllm-url http://localhost:8000
```

**Benefits of Option B:**
- Server can run on different machine
- Multiple benchmark clients can share same model
- Easier to monitor server separately
- Can keep server running between benchmark runs

---

## Configuration

### Environment Variables (.env)

```ini
# =============================================================================
# vLLM Configuration (for 120B+ models)
# =============================================================================

# Enable vLLM backend (set to true for 120B models)
VLLM_ENABLED=true

# vLLM tensor parallel size (number of GPUs for single model)
VLLM_TENSOR_PARALLEL_SIZE=4

# vLLM GPU memory utilization (0.90-0.95 recommended)
VLLM_GPU_MEMORY_UTILIZATION=0.95

# vLLM maximum model length (prompt + output tokens)
VLLM_MAX_MODEL_LEN=8192

# vLLM API server URL (if running server separately)
VLLM_API_URL=http://localhost:8000

# vLLM maximum tokens to generate
VLLM_MAX_TOKENS=2048

# vLLM temperature for sampling
VLLM_TEMPERATURE=0.3

# Your model (change to 120B)
LLM_MODEL_NAME=openai/gpt-oss-120b
```

### vLLM Server Flags Explained

| Flag | Recommended Value | Description |
|------|------------------|-------------|
| `--tensor-parallel-size` | 4 | GPUs for tensor parallelism |
| `--gpu-memory-utilization` | 0.95 | Fraction of VRAM to use |
| `--max-model-len` | 8192 | Max sequence length (prompt + output) |
| `--max-num-batched-tokens` | 32768 | Max tokens per batch |
| `--enable-chunked-prefill` | (flag) | Handle long prompts without OOM |
| `--dtype` | bfloat16 | Model dtype (A100 native) |
| `--kv-cache-dtype` | auto | KV cache dtype (auto-selects) |
| `--enforce-eager` | False | Use CUDA graphs for speed |

---

## Architecture Comparison

### Transformers Approach (20B × 4)

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

**Total Parameters:** 80B (4 × 20B)  
**Total VRAM:** ~168GB (4 × 42GB)

### vLLM Approach (120B TP=4)

```
4× A100 GPUs (80GB each)
┌─────────────────────────────────────────────────┐
│           Single 120B Model (Tensor Parallel)   │
│  ┌─────────────────────────────────────────┐    │
│  │  GPU 0  │  GPU 1  │  GPU 2  │  GPU 3   │    │
│  │  Layer  │  Layer  │  Layer  │  Layer   │    │
│  │  0-29   │  30-59  │  60-89  │  90-119  │    │
│  │  ~70GB  │  ~70GB  │  ~70GB  │  ~70GB   │    │
│  └─────────────────────────────────────────┘    │
│                    ↓                              │
│         All 500 Questions                        │
│         (Sequential Processing)                  │
└─────────────────────────────────────────────────┘
```

**Total Parameters:** 120B (1 model)  
**Total VRAM:** ~280-300GB (distributed)

---

## Memory Breakdown for 120B on 4×A100

### Model Weights
- 120B parameters × 2 bytes (bfloat16) = **240GB**
- Per GPU: 240GB / 4 = **60GB**

### KV Cache (vLLM PagedAttention)
- Depends on sequence length and batch size
- vLLM allocates dynamically (2-4× more efficient than transformers)
- Estimated: **10-15GB per GPU**

### Activation Memory
- During forward pass: **~5GB per GPU**

### Total Per GPU
- Weights: 60GB
- KV Cache: 10-15GB
- Activations: 5GB
- **Total: ~75-80GB** (fits in 80GB A100 with 95% memory utilization)

---

## Expected Performance

### Runtime Estimates (500 Questions)

| Model | Backend | GPUs | Time | Notes |
|-------|---------|------|------|-------|
| 20B | Transformers | 4 | 30-40 hours | Your current baseline |
| 120B | vLLM | 4 | 50-80 hours | **6× larger model** |
| 120B | Transformers | 8+ | 100+ hours | Would need 8+ GPUs |

### Why 120B is Slower but Better

- **Slower**: 6× more parameters = more compute per token
- **Better**: Higher accuracy, better reasoning, fewer hallucinations
- **Trade-off**: Accept 1.5-2× longer runtime for significantly better quality

---

## Monitoring

### GPU Utilization

```bash
# Watch all GPUs
watch -n 1 nvidia-smi

# Expected output during generation:
# +-----------------------------------------------------------------------------+
# | GPU  Name        | Memory-Usage | GPU-Util | Processes                     |
# +------------------+--------------+----------+-------------------------------+
# |   0  A100 80GB   | 76000 / 81920 MiB | 95% |   python (vllm)               |
# |   1  A100 80GB   | 76000 / 81920 MiB | 95% |   python (vllm)               |
# |   2  A100 80GB   | 76000 / 81920 MiB | 95% |   python (vllm)               |
# |   3  A100 80GB   | 76000 / 81920 MiB | 95% |   python (vllm)               |
# +-----------------------------------------------------------------------------+
```

### vLLM Server Logs

The server outputs throughput metrics:
```
INFO:     Generated 1234 tokens in 45.67s (27.02 tokens/s)
INFO:     Batch size: 32, Avg prompt length: 512 tokens
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Reduce GPU memory utilization:**
   ```bash
   --gpu-memory-utilization 0.90  # Reduce from 0.95
   ```

2. **Reduce max model length:**
   ```bash
   --max-model-len 4096  # Reduce from 8192
   ```

3. **Enable chunked prefill (if not already):**
   ```bash
   --enable-chunked-prefill
   ```

### Slow Loading

**Symptoms:** Model takes >10 minutes to load

**Solution:** This is normal for 120B. First load downloads weights (~240GB). Subsequent loads are faster from cache.

### Tensor Parallelism Errors

**Symptoms:**
```
RuntimeError: Expected all tensors to be on the same device
```

**Solution:** Ensure `--tensor-parallel-size` matches available GPUs:
```bash
nvidia-smi  # Count GPUs
python engine/run_benchmark_vllm.py --tensor-parallel-size 4  # Match count
```

### vLLM Not Generating

**Symptoms:** Server running but no output

**Check:**
```bash
# Test server directly
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "SELECT", "max_tokens": 10}'
```

---

## Advanced Configuration

### Custom Model Loading

If using a local model path:
```bash
python engine/run_benchmark_vllm.py \
    --benchmark ... \
    --db_root ... \
    --output outputs/benchmark_local \
    --tensor-parallel-size 4

# In .env:
LLM_MODEL_NAME=/path/to/local/model
```

### Multi-Node vLLM (8+ GPUs)

For models larger than 4 GPUs can handle:
```bash
# Node 1
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 0.0.0.0 \
    --distributed-executor-backend mp

# Node 2 (same command, different port)
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --port 8001 \
    --host 0.0.0.0 \
    --distributed-executor-backend mp
```

### Custom Sampling Parameters

Edit `engine/run_benchmark_vllm.py`:
```python
from vllm import SamplingParams

params = SamplingParams(
    max_tokens=512,
    temperature=0.3,
    top_p=0.95,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)
```

---

## Migration from Transformers

### Code Changes Required

**Minimal changes** - the vLLM integration is designed to be drop-in compatible:

1. **Use `run_benchmark_vllm.py` instead of `run_benchmark.py`**
2. **Update `.env` with vLLM settings**
3. **Change model to 120B**

### What Stays the Same

- FAISS index loading
- Schema linking logic
- SQL execution and evaluation
- Majority voting
- Result merging
- Output format

### What Changes

- Model loading (vLLM vs. transformers)
- Batch generation (continuous vs. static)
- Memory management (PagedAttention vs. pre-allocated)

---

## Best Practices

1. **Start with a test run:**
   ```bash
   python engine/run_benchmark_vllm.py \
       --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
       --db_root minidev/MINIDEV/dev_databases/ \
       --output outputs/test_vllm \
       --limit 5 \
       --tensor-parallel-size 4
   ```

2. **Use tmux/screen for long runs:**
   ```bash
   tmux new -s vllm-benchmark
   python engine/run_benchmark_vllm.py ...
   # Detach: Ctrl+B, D
   # Reattach: tmux attach -t vllm-benchmark
   ```

3. **Monitor GPU temperature:**
   ```bash
   nvidia-smi dmon -i 0,1,2,3
   ```

4. **Save checkpoints periodically:**
   The script auto-saves after each question. Resume with:
   ```bash
   python engine/run_benchmark_vllm.py \
       --start 100 \
       ...
   ```

5. **Keep server running between runs:**
   Use Option B (separate server) to avoid reload time.

---

## Performance Tuning

### Optimal Settings for 120B on 4×A100

```bash
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --enforce-eager False
```

### If OOM Persists

```bash
--gpu-memory-utilization 0.90 \
--max-model-len 4096 \
--max-num-batched-tokens 16384
```

### For Maximum Throughput

```bash
--gpu-memory-utilization 0.98 \
--max-num-batched-tokens 65536 \
--num-scheduler-steps 10
```

---

## Summary

**vLLM is essential for running 120B on 4×A100 80GB because:**

1. ✅ **Memory Efficiency**: PagedAttention fits 120B in 280-300GB (vs. 480GB+ for transformers)
2. ✅ **Tensor Parallelism**: Single model across 4 GPUs (vs. 4 separate 20B copies)
3. ✅ **Throughput**: Continuous batching = 2-3× faster generation
4. ✅ **Stability**: Better memory management = fewer OOM errors

**Expected Results:**
- **Runtime**: 50-80 hours for 500 questions
- **Quality**: Significant accuracy improvement from 20B → 120B
- **Stability**: No more OOM crashes

**Next Steps:**
1. Install vLLM: `pip install vllm`
2. Update `.env`: Set `LLM_MODEL_NAME=openai/gpt-oss-120b`
3. Run test: `python engine/run_benchmark_vllm.py --limit 5 ...`
4. Run full benchmark: Remove `--limit` flag

---

For questions or issues, check:
- vLLM documentation: https://docs.vllm.ai/
- MCS-SQL engine logs: `outputs/benchmark_vllm_*/`
