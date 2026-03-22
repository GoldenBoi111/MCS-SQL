"""
Quick Test Script for vLLM 120B - Single Question Timing

This script runs exactly 1 question to measure timing for extrapolation.

Usage:
    python engine/test_vllm_single.py \
        --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
        --db_root minidev/MINIDEV/dev_databases/ \
        --output outputs/test_vllm_single
"""

import argparse
import json
import os
import time
from pathlib import Path

from config import Config
from run_benchmark_vllm import run_vllm_benchmark


def main():
    parser = argparse.ArgumentParser(description="Test vLLM 120B with single question")
    parser.add_argument("--benchmark", required=True, help="Path to mini_dev_sqlite.json")
    parser.add_argument("--db_root", required=True, help="Path to databases directory")
    parser.add_argument("--output", default="outputs/test_vllm_single")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of GPUs for tensor parallelism")
    
    args = parser.parse_args()
    
    print("="*70)
    print("vLLM 120B - Single Question Timing Test")
    print("="*70)
    print()
    print("This will run exactly 1 question to measure timing.")
    print("Use the results to extrapolate full benchmark runtime.")
    print()
    
    # Load benchmark to show which question we're testing
    with open(args.benchmark, 'r') as f:
        questions = json.load(f)
    
    test_question = questions[0]
    print(f"Testing with question 1:")
    print(f"  ID: {test_question.get('question_id', 0)}")
    print(f"  DB: {test_question['db_id']}")
    print(f"  Q:  {test_question['question'][:100]}...")
    print()
    
    # Record start time
    start_time = time.time()
    
    # Run benchmark with limit=1
    try:
        run_vllm_benchmark(
            benchmark_path=args.benchmark,
            db_root=args.db_root,
            output_dir=args.output,
            tensor_parallel_size=args.tensor_parallel_size,
            limit=1,
        )
        
        elapsed = time.time() - start_time
        
        print()
        print("="*70)
        print("TIMING RESULTS")
        print("="*70)
        print(f"  Single question (1/500):  {elapsed:.2f} seconds")
        print(f"                            {elapsed/60:.2f} minutes")
        print()
        print("EXTRAPOLATED RUNTIME FOR 500 QUESTIONS:")
        print(f"  500 questions:  {elapsed * 500:.2f} seconds")
        print(f"                  {elapsed * 500 / 60:.2f} minutes")
        print(f"                  {elapsed * 500 / 3600:.2f} hours")
        print(f"                  {elapsed * 500 / 86400:.2f} days")
        print()
        print("NOTE: This is a rough estimate. Actual runtime may vary due to:")
        print("  - Question difficulty variation")
        print("  - SQL execution time differences")
        print("  - System load and thermal throttling")
        print("="*70)
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print()
        print(f"Test interrupted after {elapsed:.2f} seconds")
        print("Partial results may be available in:", args.output)


if __name__ == "__main__":
    main()
