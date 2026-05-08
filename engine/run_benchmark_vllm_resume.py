"""
Auto-resume wrapper for the vLLM benchmark runner.

This script inspects the provided output directory, finds completed
question IDs from prior benchmark files, computes the next missing
array index in the benchmark JSON, and resumes from there.

Usage:
    python engine/run_benchmark_vllm_resume.py \
        --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
        --db_root minidev/MINIDEV/dev_databases/ \
        --output outputs/benchmark_vllm_120b
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Set, Tuple

from run_benchmark_vllm import load_benchmark, run_vllm_benchmark


def collect_completed_question_ids(resume_dir: str) -> Set[int]:
    """
    Collect completed question IDs from prior benchmark outputs.

    Searches recursively for benchmark_results.json and
    benchmark_results_merged.json files.
    """
    completed_qids: Set[int] = set()

    if not resume_dir:
        return completed_qids

    resume_path = Path(resume_dir)
    if not resume_path.exists():
        return completed_qids

    result_files = []
    result_files.extend(
        sorted(
            path
            for path in resume_path.rglob("benchmark_results.json")
            if "merged" not in path.name
        )
    )
    result_files.extend(sorted(resume_path.rglob("benchmark_results_merged.json")))

    seen_files = set()
    for results_file in result_files:
        if results_file in seen_files:
            continue
        seen_files.add(results_file)

        try:
            with open(results_file, "r", encoding="utf-8") as handle:
                results = json.load(handle)
        except Exception:
            continue

        if not isinstance(results, list):
            continue

        for row in results:
            if not isinstance(row, dict):
                continue
            qid = row.get("question_id")
            if qid is None:
                continue
            try:
                completed_qids.add(int(qid))
            except (TypeError, ValueError):
                continue

    return completed_qids


def find_next_start_index(
    benchmark_path: str,
    completed_qids: Set[int],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the next array index to process.

    Returns:
        (start_index, next_question_id)
    """
    benchmark = load_benchmark(benchmark_path)

    for index, question in enumerate(benchmark):
        qid = question.get("question_id", index)
        try:
            qid_int = int(qid)
        except (TypeError, ValueError):
            qid_int = index

        if qid_int not in completed_qids:
            return index, qid_int

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Auto-resume vLLM benchmark")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSON")
    parser.add_argument("--db_root", required=True, help="Path to database root directory")
    parser.add_argument(
        "--output",
        default="outputs/benchmark_vllm_120b",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Directory to inspect for previous progress (defaults to --output)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument("--vllm-url", type=str, default=None, help="Optional vLLM API URL")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Manual start index override (skips auto-detection)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed resume point without running the benchmark",
    )

    args = parser.parse_args()

    resume_dir = args.resume_dir or args.output

    if args.start is not None:
        start_index = args.start
        next_question_id = None
        print(f"Using manual start index: {start_index}")
    else:
        completed_qids = collect_completed_question_ids(resume_dir)
        print(f"Loaded {len(completed_qids)} completed question_id values from {resume_dir}")
        start_index, next_question_id = find_next_start_index(args.benchmark, completed_qids)

        if start_index is None:
            print("All questions in the benchmark are already completed.")
            return

        print(f"Auto-resume will start at array index {start_index}")
        print(f"Next question_id: {next_question_id}")

    print(f"Output directory: {args.output}")

    if args.dry_run:
        print("Dry run requested. No benchmark was started.")
        return

    run_vllm_benchmark(
        benchmark_path=args.benchmark,
        db_root=args.db_root,
        output_dir=args.output,
        tensor_parallel_size=args.tensor_parallel_size,
        vllm_url=args.vllm_url,
        limit=args.limit,
        start_index=start_index,
        end_index=args.end,
    )


if __name__ == "__main__":
    main()
