"""
Check benchmark progress - shows how many questions each GPU has completed.

Usage:
    python engine/check_progress.py outputs/benchmark/start_0_end_125
"""

import json
import os
import sys
from pathlib import Path


def check_progress(output_dir: str):
    """
    Check progress of benchmark runs.
    
    Args:
        output_dir: Directory containing gpu_X subdirectories
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    # Find all gpu_X directories
    gpu_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("gpu_")])
    
    if not gpu_dirs:
        print(f"Error: No gpu_X directories found in {output_dir}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK PROGRESS CHECK: {output_dir}")
    print(f"{'='*70}\n")
    
    total_results = 0
    max_question_ids = {}
    
    for gpu_dir in gpu_dirs:
        results_file = gpu_dir / "benchmark_results.json"
        
        if not results_file.exists():
            print(f"  {gpu_dir.name}: No results file found")
            continue
        
        with open(results_file, "r") as f:
            gpu_results = json.load(f)
        
        # Find max question_id in this GPU's results
        if gpu_results:
            max_qid = max(r.get("question_id", 0) for r in gpu_results)
            min_qid = min(r.get("question_id", 0) for r in gpu_results)
            max_question_ids[gpu_dir.name] = max_qid
            print(f"  {gpu_dir.name}: {len(gpu_results)} questions (IDs: {min_qid} - {max_qid})")
            total_results += len(gpu_results)
        else:
            print(f"  {gpu_dir.name}: 0 questions (empty file)")
    
    # Check for merged file
    merged_file = output_path / "benchmark_results_merged.json"
    if merged_file.exists():
        with open(merged_file, "r") as f:
            merged_results = json.load(f)
        print(f"\n  Merged file: {len(merged_results)} questions")
    
    print(f"\n  {'='*50}")
    print(f"  TOTAL: {total_results} questions completed")
    print(f"  {'='*50}")
    
    # Calculate next start indices for each GPU
    print(f"\n{'='*70}")
    print(f"TO RESUME - Use these start indices:")
    print(f"{'='*70}")
    
    # Assuming questions are distributed evenly, calculate next index per GPU
    if max_question_ids:
        # Find the global max question ID completed
        global_max = max(max_question_ids.values())
        print(f"\n  Highest question ID completed: {global_max}")
        print(f"  Next question to process: {global_max + 1}")
        print(f"\n  To resume from next question:")
        print(f"    --start {global_max + 1}")
    
    print(f"\n{'='*70}\n")


def show_question_ids(output_dir: str, gpu_id: int = None, limit: int = 10):
    """
    Show question IDs from results files.
    
    Args:
        output_dir: Directory containing gpu_X subdirectories
        gpu_id: Specific GPU to check (None for all)
        limit: Max question IDs to show per GPU
    """
    output_path = Path(output_dir)
    
    if gpu_id is not None:
        gpu_dirs = [output_path / f"gpu_{gpu_id}"]
    else:
        gpu_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("gpu_")])
    
    print(f"\n{'='*70}")
    print(f"QUESTION IDs in results (showing first {limit} per GPU):")
    print(f"{'='*70}\n")
    
    for gpu_dir in gpu_dirs:
        results_file = gpu_dir / "benchmark_results.json"
        
        if not results_file.exists():
            print(f"  {gpu_dir.name}: No results file")
            continue
        
        with open(results_file, "r") as f:
            gpu_results = json.load(f)
        
        question_ids = [r.get("question_id", "N/A") for r in gpu_results[:limit]]
        print(f"  {gpu_dir.name}: {question_ids}")
        if len(gpu_results) > limit:
            print(f"    ... and {len(gpu_results) - limit} more")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_progress.py <output_directory> [gpu_id] [limit]")
        print("\nExamples:")
        print("  python check_progress.py outputs/benchmark/start_0_end_125")
        print("  python check_progress.py outputs/benchmark/start_0_end_125 0")
        print("  python check_progress.py outputs/benchmark/start_0_end_125 0 20")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    gpu_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    check_progress(output_dir)
    
    if gpu_id is not None or len(sys.argv) > 2:
        show_question_ids(output_dir, gpu_id, limit)
