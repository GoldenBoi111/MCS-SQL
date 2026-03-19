"""
Check benchmark progress - shows how many questions each GPU has completed.

Usage:
    python engine/check_progress.py outputs/benchmark/start_0_end_125
"""

import json
import os
import sys
from pathlib import Path


def check_progress(output_dir: str, benchmark_path: str = None):
    """
    Check progress of benchmark runs.
    
    Args:
        output_dir: Directory containing gpu_X subdirectories
        benchmark_path: Optional path to original benchmark JSON for range analysis
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
    all_question_ids = []
    gpu_results_info = {}
    
    for gpu_dir in gpu_dirs:
        results_file = gpu_dir / "benchmark_results.json"
        
        if not results_file.exists():
            print(f"  {gpu_dir.name}: No results file found")
            continue
        
        with open(results_file, "r") as f:
            gpu_results = json.load(f)
        
        # Collect question_ids
        question_ids = [r.get("question_id") for r in gpu_results if r.get("question_id") is not None]
        all_question_ids.extend(question_ids)
        
        if gpu_results:
            max_qid = max(r.get("question_id", 0) for r in gpu_results)
            min_qid = min(r.get("question_id", 0) for r in gpu_results)
            gpu_results_info[gpu_dir.name] = {
                'count': len(gpu_results),
                'min_qid': min_qid,
                'max_qid': max_qid,
                'question_ids': set(question_ids)
            }
            print(f"  {gpu_dir.name}: {len(gpu_results)} questions (question_id: {min_qid} - {max_qid})")
            total_results += len(gpu_results)
        else:
            print(f"  {gpu_dir.name}: 0 questions (empty file)")
            gpu_results_info[gpu_dir.name] = {
                'count': 0,
                'min_qid': None,
                'max_qid': None,
                'question_ids': set()
            }
    
    # Check for merged file
    merged_file = output_path / "benchmark_results_merged.json"
    if merged_file.exists():
        with open(merged_file, "r") as f:
            merged_results = json.load(f)
        print(f"\n  Merged file: {len(merged_results)} questions")
    
    print(f"\n  {'='*50}")
    print(f"  TOTAL: {total_results} questions completed")
    print(f"  {'='*50}")
    
    # If benchmark path provided, analyze original distribution
    if benchmark_path and os.path.exists(benchmark_path):
        print(f"\n{'='*70}")
        print(f"ORIGINAL GPU DISTRIBUTION (from benchmark JSON):")
        print(f"{'='*70}")
        
        with open(benchmark_path, 'r') as f:
            benchmark = json.load(f)
        
        total_questions = len(benchmark)
        num_gpus = len(gpu_dirs)
        chunk_size = (total_questions + num_gpus - 1) // num_gpus
        
        print(f"\n  Total questions in benchmark: {total_questions}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Questions per GPU: ~{chunk_size}")
        print(f"\n  Original distribution:")
        
        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_questions)
            
            if start_idx < total_questions:
                # Get question_ids for this GPU's range
                gpu_qids = set(benchmark[j].get("question_id") for j in range(start_idx, end_idx))
                
                # Check how many this GPU completed
                if f"gpu_{i}" in gpu_results_info:
                    completed = len(gpu_results_info[f"gpu_{i}"]['question_ids'])
                    remaining = len(gpu_qids - gpu_results_info[f"gpu_{i}"]['question_ids'])
                    print(f"    GPU {i}: array indices {start_idx} - {end_idx-1} ({completed} done, {remaining} remaining)")
                else:
                    print(f"    GPU {i}: array indices {start_idx} - {end_idx-1} (no results)")
            else:
                print(f"    GPU {i}: (no questions assigned)")
        
        print(f"\n{'='*70}")
    
    # Calculate next array index (not question_id!)
    print(f"\n{'='*70}")
    print(f"TO RESUME - Use array index (position in JSON list):")
    print(f"{'='*70}")
    
    if all_question_ids:
        print(f"\n  ⚠️  NOTE: question_id values are NOT sequential array indices!")
        print(f"  Your benchmark uses question_id values like: {sorted(all_question_ids)[:5]}...")
        print(f"\n  To find the next array index:")
        print(f"    1. Open your benchmark JSON file")
        print(f"    2. Find which array position contains the next unprocessed question")
        print(f"    3. Use that position as --start")
        print(f"\n  Example: If question at position 106 is next, use --start 106")
    
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
        print("Usage: python check_progress.py <output_directory> [benchmark_path] [gpu_id] [limit]")
        print("\nExamples:")
        print("  python check_progress.py outputs/benchmark_results")
        print("  python check_progress.py outputs/benchmark_results /path/to/mini_dev_sqlite.json")
        print("  python check_progress.py outputs/benchmark_results /path/to/mini_dev_sqlite.json 0")
        print("  python check_progress.py outputs/benchmark_results /path/to/mini_dev_sqlite.json 0 20")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    benchmark_path = sys.argv[2] if len(sys.argv) > 2 else None
    gpu_id = int(sys.argv[3]) if len(sys.argv) > 3 else None
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    check_progress(output_dir, benchmark_path)
    
    if gpu_id is not None or len(sys.argv) > 3:
        show_question_ids(output_dir, gpu_id, limit)
