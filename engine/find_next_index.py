"""
Find the next array index to resume benchmark from.

This script compares completed question_ids against the original benchmark JSON
to find the next array index (position) that needs processing.

Usage:
    python engine/find_next_index.py \
        --benchmark /path/to/mini_dev_sqlite.json \
        --output-dir outputs/benchmark_results
"""

import json
import argparse
import sys
from pathlib import Path


def find_next_index(benchmark_path: str, output_dir: str):
    """
    Find the next array index to resume from.
    
    Args:
        benchmark_path: Path to original benchmark JSON
        output_dir: Directory containing gpu_X subdirectories (or parent directory)
    """
    # Load original benchmark
    with open(benchmark_path, 'r') as f:
        benchmark = json.load(f)
    
    print(f"Loaded benchmark: {len(benchmark)} questions")
    
    # Load all completed question_ids from results (search recursively)
    output_path = Path(output_dir)
    completed_qids = set()
    total_results = 0
    
    # Find all gpu_X directories recursively
    gpu_dirs = sorted([d for d in output_path.rglob("gpu_*") if d.is_dir()])
    
    for gpu_dir in gpu_dirs:
        results_file = gpu_dir / "benchmark_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                gpu_results = json.load(f)
            for r in gpu_results:
                qid = r.get("question_id")
                if qid is not None:
                    completed_qids.add(qid)
            total_results += len(gpu_results)
            print(f"  {gpu_dir.parent.name}/{gpu_dir.name}: {len(gpu_results)} results")
    
    # Also check merged files
    for merged_file in output_path.rglob("benchmark_results_merged.json"):
        with open(merged_file, "r") as f:
            merged_results = json.load(f)
        for r in merged_results:
            qid = r.get("question_id")
            if qid is not None:
                completed_qids.add(qid)
        total_results += len(merged_results)
        print(f"  Merged ({merged_file.parent.name}): {len(merged_results)} results")
    
    print(f"\nTotal completed question_ids: {len(completed_qids)}")
    print(f"Total results (may include duplicates): {total_results}")
    
    # Find first unprocessed question in benchmark
    next_index = None
    for i, q in enumerate(benchmark):
        qid = q.get("question_id")
        if qid not in completed_qids:
            next_index = i
            break
    
    if next_index is None:
        print(f"\n✅ All {len(benchmark)} questions completed!")
        return
    
    next_qid = benchmark[next_index].get("question_id")
    remaining = len(benchmark) - next_index
    
    print(f"\n{'='*70}")
    print(f"NEXT QUESTION TO PROCESS:")
    print(f"{'='*70}")
    print(f"  Array index (position in JSON): {next_index}")
    print(f"  question_id field value: {next_qid}")
    print(f"  Remaining questions: {remaining}")
    print(f"\n  Use this command to resume:")
    print(f"    --start {next_index}")
    print(f"\n  Example:")
    print(f"    python run_benchmark.py --start {next_index} --multi-gpu --num-gpus 2 ...")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find next array index to resume benchmark")
    parser.add_argument("--benchmark", required=True, help="Path to original benchmark JSON")
    parser.add_argument("--output-dir", required=True, help="Directory with gpu_X results")
    
    args = parser.parse_args()
    
    find_next_index(args.benchmark, args.output_dir)
