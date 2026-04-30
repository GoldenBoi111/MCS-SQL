"""
Resumable Spider benchmark runner for MCS-SQL.

This file keeps the existing Spider benchmark flow intact, but adds:
- explicit GPU ID selection
- explicit GPU start offsets
- resume from previous outputs
- dry-run planning

It reuses the current Spider runner for the actual model/eval work.
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from config import Config
from run_benchmark_spider import gpu_worker, load_benchmark
from run_benchmark_spider import run_spider_benchmark


RESULT_FILENAMES = {
    "benchmark_results.json",
    "benchmark_results_merged.json",
    "all_outputs.json",
}


@dataclass(frozen=True)
class GPUPlan:
    gpu_id: int
    start_index: int
    end_index: int
    completed_count: int
    remaining_count: int
    question_ids: List[int]
    resume_from: Optional[int]
    stop_source: str


def load_questions_with_ids(benchmark_path: str) -> List[Dict]:
    questions = load_benchmark(benchmark_path)
    prepared_questions = []
    for index, question in enumerate(questions):
        question_copy = dict(question)
        question_copy.setdefault("question_id", index)
        prepared_questions.append(question_copy)
    return prepared_questions


def parse_int_list(values: Optional[Sequence[int]]) -> Optional[List[int]]:
    if values is None:
        return None
    return [int(value) for value in values]


def collect_completed_question_ids(progress_dir: Optional[str]) -> Set[int]:
    if not progress_dir:
        return set()

    progress_path = Path(progress_dir)
    if not progress_path.exists():
        return set()

    completed_qids: Set[int] = set()
    candidate_files = []

    for filename in RESULT_FILENAMES:
        candidate_files.extend(progress_path.rglob(filename))

    for results_file in candidate_files:
        try:
            with open(results_file, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict) and row.get("question_id") is not None:
                        completed_qids.add(int(row["question_id"]))
        except Exception:
            continue

    return completed_qids


def discover_gpu_resume_starts(resume_dir: Optional[str]) -> List[Tuple[int, int]]:
    """
    Look inside the resume directory and infer each GPU's start question index
    from the first completed question in that GPU's output file.
    """
    if not resume_dir:
        return []

    resume_path = Path(resume_dir)
    if not resume_path.exists():
        return []

    discovered: List[Tuple[int, int]] = []
    for gpu_dir in sorted([path for path in resume_path.iterdir() if path.is_dir() and path.name.startswith("gpu_")]):
        try:
            gpu_id = int(gpu_dir.name.split("_", 1)[1])
        except Exception:
            continue

        results_file = None
        for filename in ["all_outputs.json", "benchmark_results.json", "benchmark_results_merged.json"]:
            candidate = gpu_dir / filename
            if candidate.exists():
                results_file = candidate
                break

        if results_file is None:
            continue

        try:
            with open(results_file, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if not isinstance(data, list):
                continue
            question_ids = [
                int(row["question_id"])
                for row in data
                if isinstance(row, dict) and row.get("question_id") is not None
            ]
            if question_ids:
                discovered.append((gpu_id, min(question_ids)))
        except Exception:
            continue

    discovered.sort(key=lambda item: item[1])
    return discovered


def build_gpu_ranges(
    total_questions: int,
    gpu_ids: List[int],
    gpu_starts: Optional[List[int]] = None,
    resume_dir: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")

    if gpu_starts is not None and len(gpu_starts) != len(gpu_ids):
        raise ValueError(
            f"--gpu-starts must have the same number of entries as --gpu-ids "
            f"({len(gpu_starts)} != {len(gpu_ids)})"
        )

    if gpu_starts is None:
        chunk_size = (total_questions + len(gpu_ids) - 1) // len(gpu_ids)
        gpu_starts = [i * chunk_size for i in range(len(gpu_ids))]

    discovered_resume_starts = discover_gpu_resume_starts(resume_dir)
    discovered_start_by_gpu = {gpu_id: start for gpu_id, start in discovered_resume_starts}
    discovered_start_values = sorted(start for _, start in discovered_resume_starts)

    ranges = []
    for index, gpu_id in enumerate(gpu_ids):
        if resume_dir and gpu_id in discovered_start_by_gpu:
            start_index = discovered_start_by_gpu[gpu_id]
        else:
            start_index = max(0, gpu_starts[index])

        if discovered_start_values:
            end_candidates = [value for value in discovered_start_values if value > start_index]
            end_index = min(end_candidates) if end_candidates else total_questions
        else:
            end_index = gpu_starts[index + 1] if index + 1 < len(gpu_starts) else total_questions

        end_index = min(end_index, total_questions)
        start_index = min(start_index, total_questions)
        if end_index < start_index:
            end_index = start_index
        ranges.append((gpu_id, start_index, end_index))
    return ranges


def build_gpu_plans(
    questions: List[Dict],
    gpu_ranges: List[Tuple[int, int, int]],
    completed_qids: Set[int],
    resume_dir: Optional[str] = None,
) -> List[GPUPlan]:
    plans: List[GPUPlan] = []
    discovered_resume_starts = discover_gpu_resume_starts(resume_dir)
    discovered_start_by_gpu = {gpu_id: start for gpu_id, start in discovered_resume_starts}
    discovered_start_values = sorted(start for _, start in discovered_resume_starts)

    for gpu_id, start_index, end_index in gpu_ranges:
        chunk = questions[start_index:end_index]
        remaining = [question for question in chunk if int(question["question_id"]) not in completed_qids]
        question_ids = [int(question["question_id"]) for question in chunk]
        resume_from = int(remaining[0]["question_id"]) if remaining else None
        stop_source = "dataset_end"
        if discovered_start_values:
            end_candidates = [value for value in discovered_start_values if value > start_index]
            if end_candidates:
                stop_value = min(end_candidates)
                for other_gpu_id, other_start in discovered_resume_starts:
                    if other_start == stop_value:
                        stop_source = f"resume_dir gpu_{other_gpu_id} start={other_start}"
                        break
            else:
                stop_source = "dataset_end"
        elif end_index < len(questions):
            stop_source = f"explicit_start_next_gpu={end_index}"
        plans.append(
            GPUPlan(
                gpu_id=gpu_id,
                start_index=start_index,
                end_index=end_index,
                completed_count=len(chunk) - len(remaining),
                remaining_count=len(remaining),
                question_ids=question_ids,
                resume_from=resume_from,
                stop_source=stop_source,
            )
        )

    return plans


def print_plans(plans: List[GPUPlan], dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "RESUME PLAN"
    print(f"\n{'=' * 80}")
    print(f"{mode}")
    print(f"{'=' * 80}")
    for plan in plans:
        if plan.question_ids:
            start_qid = plan.question_ids[0]
            end_qid = plan.question_ids[-1]
        else:
            start_qid = None
            end_qid = None

        print(
            f"GPU {plan.gpu_id}: indices {plan.start_index}..{plan.end_index - 1 if plan.end_index > plan.start_index else plan.start_index - 1} "
            f"(question_id {start_qid}..{end_qid})"
        )
        print(
            f"  completed={plan.completed_count}, remaining={plan.remaining_count}, "
            f"resume_from={plan.resume_from}"
        )
        print(f"  stops_at={plan.end_index} via {plan.stop_source}")
    print(f"{'=' * 80}\n")


def prepare_gpu_chunks(
    questions: List[Dict],
    plans: List[GPUPlan],
    completed_qids: Set[int],
) -> List[Tuple[int, List[Dict], int]]:
    gpu_chunks: List[Tuple[int, List[Dict], int]] = []

    for plan in plans:
        chunk = questions[plan.start_index : plan.end_index]
        remaining_chunk = [
            question for question in chunk if int(question["question_id"]) not in completed_qids
        ]
        if remaining_chunk:
            start_index = int(remaining_chunk[0]["question_id"])
            gpu_chunks.append((plan.gpu_id, remaining_chunk, start_index))

    return gpu_chunks


def merge_gpu_results(output_dir: str) -> str:
    output_path = Path(output_dir)
    result_files = sorted(output_path.rglob("benchmark_results.json"))
    merged_results = []
    seen_qids = set()

    for results_file in result_files:
        try:
            with open(results_file, "r", encoding="utf-8") as file_handle:
                file_results = json.load(file_handle)
            if not isinstance(file_results, list):
                continue
            for row in file_results:
                qid = row.get("question_id")
                if qid is None or qid in seen_qids:
                    continue
                seen_qids.add(qid)
                merged_results.append(row)
        except Exception:
            continue

    merged_results.sort(key=lambda item: item.get("question_id", 0))
    merged_file = output_path / "benchmark_results_merged.json"
    with open(merged_file, "w", encoding="utf-8") as file_handle:
        json.dump(merged_results, file_handle, indent=2)
    return str(merged_file)


def launch_workers(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    plans: List[GPUPlan],
    gpu_chunks: List[Tuple[int, List[Dict], int]],
    faiss_index: Optional[str],
    faiss_index_masked: Optional[str],
    generation_batch_size: int,
    retrieval_k: int,
) -> None:
    gpu_to_chunk = {gpu_id: (chunk, start_index) for gpu_id, chunk, start_index in gpu_chunks}

    for plan in plans:
        gpu_dir = Path(output_dir) / f"gpu_{plan.gpu_id}"
        gpu_dir.mkdir(parents=True, exist_ok=True)

    print(f"Launching {len(gpu_chunks)} GPU worker(s)...")
    processes = []

    for plan in plans:
        if plan.gpu_id not in gpu_to_chunk:
            continue

        chunk, start_index = gpu_to_chunk[plan.gpu_id]
        gpu_output_dir = str(Path(output_dir) / f"gpu_{plan.gpu_id}")
        process = mp.Process(
            target=gpu_worker,
            args=(
                plan.gpu_id,
                benchmark_path,
                db_root,
                gpu_output_dir,
                chunk,
                start_index,
                faiss_index,
                faiss_index_masked,
                generation_batch_size,
                retrieval_k,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


def run_resumable_spider_benchmark(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    gpu_ids: List[int],
    gpu_starts: Optional[List[int]] = None,
    resume_dir: Optional[str] = None,
    resume: bool = False,
    dry_run: bool = False,
    faiss_index: Optional[str] = None,
    faiss_index_masked: Optional[str] = None,
    generation_batch_size: int = 8,
    retrieval_k: int = 20,
) -> None:
    questions = load_questions_with_ids(benchmark_path)
    total_questions = len(questions)
    model_name = Config().LLM_MODEL_NAME.lower()
    is_120b = "120b" in model_name

    completed_qids = collect_completed_question_ids(resume_dir if resume else None)
    if resume and completed_qids:
        print(f"Loaded {len(completed_qids)} completed question_ids from {resume_dir}")
    elif resume:
        print("Resume requested, but no prior outputs were found.")

    gpu_ranges = build_gpu_ranges(total_questions, gpu_ids, gpu_starts, resume_dir if resume else None)
    plans = build_gpu_plans(questions, gpu_ranges, completed_qids, resume_dir=resume_dir)

    print_plans(plans, dry_run=dry_run)

    if dry_run:
        return

    gpu_chunks = prepare_gpu_chunks(questions, plans, completed_qids)
    if not gpu_chunks:
        print("No unfinished questions remain. Nothing to run.")
        return

    if is_120b and len(gpu_ids) > 1:
        print(
            "[WARN] 120B model detected. Falling back to a single process so "
            "model-parallel loading still works across all visible GPUs."
        )
        remaining_questions = []
        for _, chunk, _ in gpu_chunks:
            remaining_questions.extend(chunk)
        remaining_questions.sort(key=lambda question: int(question["question_id"]))
        os.makedirs(output_dir, exist_ok=True)
        run_spider_benchmark(
            benchmark_path=benchmark_path,
            db_root=db_root,
            output_dir=output_dir,
            limit=None,
            gpu_id=None,
            questions_chunk=remaining_questions,
            start_index=int(remaining_questions[0]["question_id"]) if remaining_questions else 0,
            faiss_index=faiss_index,
            faiss_index_masked=faiss_index_masked,
            generation_batch_size=generation_batch_size,
            retrieval_k=retrieval_k,
        )
        merged_file = merge_gpu_results(output_dir)
        print(f"Merged results saved to: {merged_file}")
        return

    os.makedirs(output_dir, exist_ok=True)
    launch_workers(
        benchmark_path=benchmark_path,
        db_root=db_root,
        output_dir=output_dir,
        plans=plans,
        gpu_chunks=gpu_chunks,
        faiss_index=faiss_index,
        faiss_index_masked=faiss_index_masked,
        generation_batch_size=generation_batch_size,
        retrieval_k=retrieval_k,
    )

    merged_file = merge_gpu_results(output_dir)
    print(f"Merged results saved to: {merged_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Spider benchmark runner for MCS-SQL")
    parser.add_argument("--benchmark", required=True, help="Path to spider dev JSON")
    parser.add_argument("--db_root", required=True, help="Path to Spider databases root")
    parser.add_argument("--output", required=True, help="Output directory for this run")
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Directory to inspect for prior progress (defaults to --output when --resume is set)",
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        required=True,
        help="Explicit GPU IDs to use, e.g. --gpu-ids 0 1 2 3",
    )
    parser.add_argument(
        "--gpu-starts",
        nargs="+",
        type=int,
        default=None,
        help="Explicit start indices per GPU, same order as --gpu-ids",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from prior outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    parser.add_argument(
        "--auto-stop-from-resume",
        action="store_true",
        help="Infer GPU stop boundaries from the resume directory",
    )
    parser.add_argument("--faiss-index", default=None, help="Path to Spider standard FAISS index")
    parser.add_argument(
        "--faiss-index-masked",
        default=None,
        help="Path to Spider masked FAISS index",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=8,
        help="Max prompts per model batch during generation",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=20,
        help="Number of retrieved examples from each FAISS index",
    )

    args = parser.parse_args()

    resume_dir = args.resume_dir or (args.output if (args.resume or args.auto_stop_from_resume) else None)
    run_resumable_spider_benchmark(
        benchmark_path=args.benchmark,
        db_root=args.db_root,
        output_dir=args.output,
        gpu_ids=args.gpu_ids,
        gpu_starts=args.gpu_starts,
        resume_dir=resume_dir,
        resume=args.resume or args.auto_stop_from_resume,
        dry_run=args.dry_run,
        faiss_index=args.faiss_index,
        faiss_index_masked=args.faiss_index_masked,
        generation_batch_size=args.generation_batch_size,
        retrieval_k=args.retrieval_k,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
