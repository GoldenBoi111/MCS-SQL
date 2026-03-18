# CUDA Error Logs

This directory contains detailed logs of CUDA out-of-memory (OOM) errors and other GPU-related issues that occurred during benchmark execution.

## Error Log Files

Each error log is a JSON file with the following naming convention:
```
{timestamp}_{error_type}_{question_id}.json
```

Example: `20260318_142530_CudaOutOfMemoryError_42.json`

## Log Structure

Each log file contains:

```json
{
  "timestamp": "20260318_142530",
  "error_type": "CudaOutOfMemoryError",
  "error_message": "CUDA out of memory. Tried to allocate 9.35 GiB...",
  "traceback": "...",
  "context": {
    "phase": "SQL_GENERATION",
    "question_id": 42,
    "batch_size": 6,
    "num_prompts": 100
  },
  "gpu_info": {
    "cuda_available": true,
    "device_count": 4,
    "gpu_0": {
      "name": "NVIDIA A100-SXM4-80GB",
      "allocated_gb": 65.70,
      "reserved_gb": 10.14,
      "free_gb": 2.90
    }
  },
  "pytorch_version": "2.1.0"
}
```

## Error Phases

Errors can occur in these phases:

| Phase | Description |
|-------|-------------|
| `SCHEMA_LINKING` | Extracting relevant tables/columns from database schema |
| `SQL_GENERATION` | Generating 100 SQL candidates (5 prompts × 20 samples) |
| `SQL_GENERATION_RECOVERY` | Retry attempt with smaller batch size |
| `SQL_SELECTION` | Selecting best SQL from high-confidence candidates |
| `SQL_SELECTION_RECOVERY` | Retry attempt for selection phase |
| `POST_QUESTION_CLEANUP` | Memory cleanup after processing a question |

## Recovery Actions

When an OOM error occurs, the system attempts:

1. **Log the error** with full GPU memory state
2. **Clear GPU memory** (gc.collect + torch.cuda.empty_cache)
3. **Retry with smaller batch size** (batch_size=6 → batch_size=2)
4. **CPU offloading** (temporarily move model weights to CPU)
5. **Continue processing** (use empty responses if recovery fails)

## Monitoring

To monitor errors in real-time:

```bash
# Watch for new error logs
watch -n 5 'ls -lt errors/ | head -10'

# View latest error
cat $(ls -t errors/*.json | head -1) | jq .

# Count errors by type
ls errors/*.json | xargs -I {} jq -r '.error_type' {} | sort | uniq -c
```

## Troubleshooting

### Frequent OOM Errors

If you see many OOM errors:

1. **Reduce batch size** in `run_benchmark.py`:
   ```python
   # Line ~366: Change batch_size from 6 to 4 or 2
   all_responses = multi_model.generate_parallel(..., batch_size=4)
   ```

2. **Use fewer GPUs** (reduces total memory pressure):
   ```bash
   CUDA_VISIBLE_DEVICES=0,2 python run_benchmark.py ... --num-gpus 2
   ```

3. **Reduce model size** or use quantization

### High Memory Usage After Cleanup

If `allocated_gb > 60GB` after cleanup:

- Check for memory leaks in custom code
- Ensure all tensors are properly deleted
- Consider restarting the benchmark

## Contact

For issues, contact the development team with:
- Error log file path
- GPU configuration
- Benchmark parameters used
