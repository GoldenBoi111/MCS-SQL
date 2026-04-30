"""
Spider benchmark runner for MCS-SQL.

This version is tuned for Spider dev data and Spider-built FAISS indices.
Key differences from the MiniDev runner:
- No evidence field is required.
- Retrieval is optional but supported through Spider FAISS indices.
- Generation is chunked to avoid the 100-prompt single-batch OOM pattern.
- Empty/invalid generations are reported as generation failures.
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from config import Config
from error_logger import ErrorLogger, MemoryManager, check_gpu_memory, clear_gpu_memory
from literal_masker import LiteralMasker
from schema_linking import SchemaLinker, TransformersLLMClient
from training_dataset_indexer import TrainingDatasetIndexer
from training_dataset_indexer_masked import MaskedTrainingDatasetIndexer

from run_benchmark import build_examples_text, execute_sql_with_timeout, get_sample_table_contents, setup_multi_gpu


logger = logging.getLogger(__name__)


SPIDER_SQL_GENERATION_PROMPT = """### Given a database schema, question, and retrieved examples, generate the correct SQLite SQL query.

### Relevant examples from index:
{examples}

### SQLite SQL tables, with their properties:
{schema_text}

### Sample rows of each table in csv format:
{sample_contents}

### Question:
{question}

You need to only create the SQL and provide brief reasoning.
Your answer should strictly follow this JSON format:
{
"reasoning": "", // Brief reasoning steps for generating SQL.
"sql": "", // The final generated SQL.
}
### Your Answer:
"""


SPIDER_SQL_SELECTION_PROMPT = """### When a DB schema and a question are given, and up to five SQLite queries are given, choose the most accurate SQL.

### SQLite SQL tables, with their properties:
{schema_text}

### Question:
{question}

### Candidate SQLs:
{candidate_sqls}

### Instruction:
- Pick the SQL that best answers the question.
- If there is no clear winner, choose the first SQL.
- Provide a short explanation following the checklist order.
- Your answer should strictly follow this JSON format.
{{
"reasoning": "", // The reasoning steps for choosing the best SQL.
"sql": "", // The final chosen SQL.
}}
### Your Answer:
"""


def load_benchmark(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3].strip()

    start_idx = response.find("{")
    if start_idx == -1:
        return None

    brace_count = 0
    end_idx = -1
    in_string = False
    escape_next = False

    for index, char in enumerate(response[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = index + 1
                    break

    if end_idx <= start_idx:
        return None

    json_str = response[start_idx:end_idx]
    if json_str.rstrip().endswith("```"):
        json_str = json_str.rstrip()[:-3]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def extract_sql_from_response(response: str) -> Optional[str]:
    parsed = parse_json_response(response)
    if parsed:
        sql_query = parsed.get("sql", "")
        if sql_query:
            return sql_query.strip()

    match = re.search(r"SELECT.*?(?:;|$)", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()

    return None


def generate_batch_in_chunks(
    llm_client: TransformersLLMClient,
    prompts: List[str],
    batch_size: int,
    error_logger: ErrorLogger,
    phase: str,
    question_id: int,
) -> List[str]:
    """
    Generate prompts in smaller chunks to avoid a single oversized batch.
    If a chunk still OOMs, recursively split it until it fits.
    """
    if not prompts:
        return []

    if len(prompts) <= batch_size:
        try:
            return llm_client.generate_batch(prompts, stop_sequences=None)
        except RuntimeError as exc:
            if "CUDA out of memory" not in str(exc):
                raise

            error_logger.log_cuda_error(
                exc,
                {
                    "phase": phase,
                    "question_id": question_id,
                    "batch_size": len(prompts),
                },
            )
            clear_gpu_memory(verbose=True)

            if len(prompts) == 1:
                return [""]

            mid = len(prompts) // 2
            left = generate_batch_in_chunks(
                llm_client,
                prompts[:mid],
                max(1, batch_size // 2),
                error_logger,
                phase,
                question_id,
            )
            right = generate_batch_in_chunks(
                llm_client,
                prompts[mid:],
                max(1, batch_size // 2),
                error_logger,
                phase,
                question_id,
            )
            return left + right

    responses: List[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        responses.extend(
            generate_batch_in_chunks(
                llm_client, chunk, batch_size, error_logger, phase, question_id
            )
        )
    return responses


def build_spider_generation_prompt(
    examples_text: str,
    schema_text: str,
    sample_contents: str,
    question: str,
) -> str:
    return (
        SPIDER_SQL_GENERATION_PROMPT.replace("{examples}", examples_text)
        .replace("{schema_text}", schema_text)
        .replace("{sample_contents}", sample_contents)
        .replace("{question}", question)
    )


def build_spider_selection_prompt(
    schema_text: str,
    question: str,
    candidate_sqls_text: str,
) -> str:
    return (
        SPIDER_SQL_SELECTION_PROMPT.replace("{schema_text}", schema_text)
        .replace("{question}", question)
        .replace("{candidate_sqls}", candidate_sqls_text)
    )


def run_spider_benchmark(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    limit: int = None,
    gpu_id: int = None,
    questions_chunk: List[Dict] = None,
    start_index: int = 0,
    faiss_index: Optional[str] = None,
    faiss_index_masked: Optional[str] = None,
    generation_batch_size: int = 8,
    retrieval_k: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "benchmark_results.json")
    results_detail: List[Dict[str, Any]] = []
    completed_qids = set()
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as file_handle:
            results_detail = json.load(file_handle)
        completed_qids = set(
            row.get("question_id") for row in results_detail if row.get("question_id") is not None
        )
        print(f"Loaded {len(results_detail)} existing results from {results_file}")
        print(f"Completed question_ids: {sorted(completed_qids)}")
        print("Already-completed questions will be skipped")

    error_logger = ErrorLogger(output_dir)
    memory_manager = MemoryManager(threshold_gb=5.0, errors_dir=output_dir)

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        print(f"Running on GPU {gpu_id}")

    config = Config()
    model_name = config.LLM_MODEL_NAME.lower()
    is_120b = "120b" in model_name

    print(f"\n{'=' * 70}")
    print("Loading model with Spider settings")
    print(f"{'=' * 70}")
    print(f"  Model: {config.LLM_MODEL_NAME}")
    print(f"  Retrieval k: {retrieval_k}")
    print(f"  Generation batch size: {generation_batch_size}")
    print(f"{'=' * 70}\n")

    if is_120b:
        llm_client = TransformersLLMClient(
            model_name=config.LLM_MODEL_NAME,
            device=config.LLM_DEVICE,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            use_model_parallel=True,
            gpu_memory_gb=75,
        )
    else:
        llm_client = TransformersLLMClient(
            model_name=config.LLM_MODEL_NAME,
            device=config.LLM_DEVICE,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            gpu_id=gpu_id,
        )

    linker = SchemaLinker(
        pt=config.TABLE_LINKING_ITERATIONS,
        pc=config.COLUMN_LINKING_ITERATIONS,
        n=config.MAJORITY_VOTE_N,
        llm_client=llm_client,
    )

    standard_indexer = None
    if faiss_index:
        print(f"Loading Spider standard index: {faiss_index}")
        standard_indexer = TrainingDatasetIndexer(
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            index_type=config.FAISS_INDEX_TYPE,
        )
        standard_indexer.load(faiss_index)

    masked_indexer = None
    if faiss_index_masked:
        print(f"Loading Spider masked index: {faiss_index_masked}")
        masked_indexer = MaskedTrainingDatasetIndexer(
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            index_type=config.FAISS_INDEX_TYPE,
        )
        masked_indexer.load(faiss_index_masked)

    literal_masker = LiteralMasker(llm_client=llm_client)

    if questions_chunk is not None:
        questions = questions_chunk
        print(f"Processing chunk of {len(questions)} questions on GPU {gpu_id}")
    else:
        questions = load_benchmark(benchmark_path)
        if limit:
            questions = questions[:limit]

    print(f"Loaded {len(questions)} questions")

    difficulty_results = {"simple": [], "moderate": [], "challenging": [], "unknown": []}

    for q_idx, q in enumerate(questions):
        global_q_idx = start_index + q_idx

        db_id = q["db_id"]
        question = q["question"]
        ground_truth = q["SQL"]
        evidence = q.get("evidence", "")
        difficulty = q.get("difficulty", "unknown")

        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        print(f"\n[{global_q_idx + 1}] Q: {question[:80]}...")

        if not os.path.exists(db_path):
            print(f"  Warning: DB not found at {db_path}")
            continue

        if global_q_idx in completed_qids:
            print("  Skipping already-completed question")
            continue

        print("  Running Schema Linking...")
        t0 = time.time()
        full_schema = linker.load_schema(db_path)

        is_low, free_gb, _ = check_gpu_memory(threshold_gb=8.0)
        if is_low:
            print(f"  [WARNING] GPU memory low before schema linking: {free_gb:.2f}GB free")
            clear_gpu_memory(verbose=True)

        try:
            linking_res = linker.link_schema(full_schema, question, evidence)
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                print("\n[CUDA OOM] Schema linking failed...")
                error_logger.log_cuda_error(
                    exc,
                    {
                        "phase": "SCHEMA_LINKING",
                        "question_id": q_idx,
                        "db_id": db_id,
                    },
                )
                linking_res = type("obj", (object,), {"tables": [], "columns": []})
            else:
                raise

        all_db_tables = set(full_schema.keys())
        original_tables = list(linking_res.tables)
        linking_res.tables = [table for table in linking_res.tables if table in all_db_tables]

        removed_tables = set(original_tables) - all_db_tables
        if removed_tables:
            print(f"  [WARNING] Removed {len(removed_tables)} hallucinated table(s): {removed_tables}")

        valid_columns = []
        for col_entry in linking_res.columns:
            if "." in col_entry:
                table_part, col_part = col_entry.split(".", 1)
                if table_part in full_schema and col_part in full_schema[table_part]:
                    valid_columns.append(col_entry)
            else:
                for table in linking_res.tables:
                    if col_entry in full_schema.get(table, []):
                        valid_columns.append(f"{table}.{col_entry}")
                        break

        linking_res.columns = valid_columns

        selected_tables = linking_res.tables if linking_res.tables else list(full_schema.keys())
        schema_text = linker.format_schema_for_prompt(full_schema, selected_tables)
        print(
            f"  Schema Linking took {time.time() - t0:.2f}s "
            f"(Found {len(linking_res.tables)} tables, {len(linking_res.columns)} columns)"
        )

        del full_schema
        gc.collect()

        print("  Retrieving examples from FAISS...")
        std_examples = []
        msk_examples = []

        if standard_indexer is not None:
            try:
                standard_results = standard_indexer.search(question, top_k=retrieval_k)
                std_examples = [
                    {"question": sq, "sql": sql, "metadata": meta}
                    for (sq, sql, meta, score) in standard_results
                ]
            except Exception as exc:
                print(f"  Warning: standard retrieval failed: {exc}")

        if masked_indexer is not None:
            try:
                masked_q = literal_masker.mask_question(question)
                masked_results = masked_indexer.search(masked_q, top_k=retrieval_k)
                msk_examples = [
                    {
                        "question": mq,
                        "sql": msql,
                        "orig_question": oq,
                        "orig_sql": osql,
                        "metadata": meta,
                    }
                    for (mq, oq, msql, osql, meta, score) in masked_results
                ]
            except Exception as exc:
                print(f"  Warning: masked retrieval failed: {exc}")

        print(f"  Standard examples: {len(std_examples)}, Masked examples: {len(msk_examples)}")

        prompt_variations = [
            ("masked_only", msk_examples[:10]),
            ("standard_only", std_examples[:10]),
        ]
        for index in range(3):
            mixed = std_examples[:5] + msk_examples[:5]
            rem_pool = std_examples[5:] + msk_examples[5:]
            if rem_pool:
                mixed += random.sample(rem_pool, min(10 - len(mixed), len(rem_pool)))
            random.shuffle(mixed)
            prompt_variations.append((f"mixed_{index}", mixed))

        print(f"  Built {len(prompt_variations)} prompt variations")
        for prompt_name, example_list in prompt_variations:
            print(f"    {prompt_name}: {len(example_list)} examples")

        sample_contents = get_sample_table_contents(
            db_path, list(linking_res.tables), sample_size=3
        )
        print(f"  Sample table contents:\n{sample_contents[:500]}...")

        all_prompts: List[str] = []
        prompt_metadata: List[Tuple[str, int]] = []

        for prompt_name, example_list in prompt_variations:
            examples_text = build_examples_text(example_list)
            base_prompt = build_spider_generation_prompt(
                examples_text=examples_text,
                schema_text=schema_text,
                sample_contents=sample_contents,
                question=question,
            )
            for gen_idx in range(config.MAJORITY_VOTE_N):
                all_prompts.append(base_prompt)
                prompt_metadata.append((prompt_name, gen_idx))

        print(f"  Built {len(all_prompts)} prompts for parallel generation...")
        print(f"  Running chunked generation (chunk size={generation_batch_size})...")

        is_low, free_gb, _ = check_gpu_memory(threshold_gb=10.0)
        if is_low:
            print(f"  [WARNING] GPU memory low before generation: {free_gb:.2f}GB free")
            clear_gpu_memory(verbose=True)

        all_responses = generate_batch_in_chunks(
            llm_client=llm_client,
            prompts=all_prompts,
            batch_size=generation_batch_size,
            error_logger=error_logger,
            phase="SQL_GENERATION",
            question_id=q_idx,
        )

        del all_prompts
        gc.collect()
        torch.cuda.empty_cache()

        print("  Parsing responses...")
        generated_candidates = []
        for index, response in enumerate(all_responses):
            prompt_name, gen_idx = prompt_metadata[index]
            print(f"    Gen {index + 1}/{len(all_responses)} - Response length: {len(response)}")

            sql_query = extract_sql_from_response(response)
            if sql_query:
                print(f"      Parsed SQL: {sql_query[:100]}...")
                generated_candidates.append(
                    {"sql": sql_query, "prompt_type": prompt_name, "gen_idx": gen_idx}
                )
            else:
                print(f"      No SQL extracted! Sample: {response[:150]}...")

        print(f"  Generated {len(generated_candidates)} valid SQL candidates")

        del all_responses
        del prompt_metadata
        gc.collect()
        torch.cuda.empty_cache()

        if not generated_candidates:
            print("  No SQL candidates were generated. Saving generation failure result...")
            failure_result = {
                "question_id": q.get("question_id", global_q_idx),
                "question": question,
                "db_id": db_id,
                "difficulty": difficulty,
                "ground_truth": ground_truth,
                "winner_sql": None,
                "is_correct": False,
                "winner_confidence": 0.0,
                "execution_times": [],
                "high_confidence_alternatives": [],
                "selection": {
                    "selected_sql": None,
                    "reasoning": None,
                    "candidates_count": 0,
                },
                "metrics": {
                    "generated": 0,
                    "execution_errors": 0,
                    "valid_generations": 0,
                    "unique_valid_sqls": 0,
                },
                "failure_info": {
                    "phase": "GENERATION",
                    "reason": "No SQL candidates were extracted from model responses",
                    "generated_candidates_sample": [],
                    "error_count": 0,
                    "generation_batch_size": generation_batch_size,
                },
            }

            results_detail.append(failure_result)
            with open(results_file, "w", encoding="utf-8") as file_handle:
                json.dump(results_detail, file_handle, indent=2)
            print(f"  Saved failure result to {results_file}")
            difficulty_results[difficulty].append(False)
            continue

        all_executions = []
        execution_errors = 0
        sql_to_result_cache = {}

        print("  Executing candidates...")
        for candidate in generated_candidates:
            sql = candidate["sql"]
            if sql in sql_to_result_cache:
                success, result_value, exec_time = sql_to_result_cache[sql]
            else:
                success, result_value, exec_time = execute_sql_with_timeout(db_path, sql)
                sql_to_result_cache[sql] = (success, result_value, exec_time)

            if success:
                all_executions.append(
                    {
                        "sql": sql,
                        "result_str": result_value,
                        "exec_time": exec_time,
                        "prompt_type": candidate["prompt_type"],
                        "gen_idx": candidate["gen_idx"],
                    }
                )
            else:
                execution_errors += 1

        n_valid = len(all_executions)
        print(f"  Valid executing queries: {n_valid} (Errors: {execution_errors})")

        if n_valid == 0:
            print("  No queries executed successfully. Saving execution failure result...")
            failure_result = {
                "question_id": q.get("question_id", global_q_idx),
                "question": question,
                "db_id": db_id,
                "difficulty": difficulty,
                "ground_truth": ground_truth,
                "winner_sql": None,
                "is_correct": False,
                "winner_confidence": 0.0,
                "execution_times": [],
                "high_confidence_alternatives": [],
                "selection": {
                    "selected_sql": None,
                    "reasoning": None,
                    "candidates_count": 0,
                },
                "metrics": {
                    "generated": len(generated_candidates),
                    "execution_errors": execution_errors,
                    "valid_generations": 0,
                    "unique_valid_sqls": 0,
                },
                "failure_info": {
                    "phase": "EXECUTION",
                    "reason": "No queries executed successfully - all generated SQL had syntax errors or timeouts",
                    "generated_candidates_sample": [
                        {"sql": cand["sql"], "prompt_type": cand["prompt_type"]}
                        for cand in generated_candidates[:10]
                    ],
                    "error_count": execution_errors,
                },
            }

            results_detail.append(failure_result)
            with open(results_file, "w", encoding="utf-8") as file_handle:
                json.dump(results_detail, file_handle, indent=2)
            print(f"  Saved failure result to {results_file}")
            difficulty_results[difficulty].append(False)

            del generated_candidates
            del all_executions
            del sql_to_result_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        result_groups: Dict[frozenset, List[Dict[str, Any]]] = {}
        for execution_item in all_executions:
            result_set = execution_item["result_str"]
            result_groups.setdefault(result_set, []).append(
                {"sql": execution_item["sql"], "exec_time": execution_item["exec_time"]}
            )

        group_normalizers = {
            result_set: min(item["exec_time"] for item in items)
            for result_set, items in result_groups.items()
        }

        result_counts = Counter(item["result_str"] for item in all_executions)
        most_common_result, top_count = result_counts.most_common(1)[0]
        winning_sqls = list(
            set(
                item["sql"]
                for item in all_executions
                if item["result_str"] == most_common_result
            )
        )
        representative_sql = min(winning_sqls, key=len)

        gt_success, gt_res_set, gt_time = execute_sql_with_timeout(db_path, ground_truth)
        winner_confidence = top_count / n_valid

        high_conf_sqls = []
        processed_results = set()
        for result_set, count in result_counts.items():
            confidence = count / n_valid
            if confidence > 0.2 and result_set not in processed_results:
                processed_results.add(result_set)
                group_sqls = list(
                    set(
                        item["sql"]
                        for item in all_executions
                        if item["result_str"] == result_set
                    )
                )
                representative = min(group_sqls, key=len)
                high_conf_sqls.append(
                    {
                        "sql": representative,
                        "confidence": confidence,
                        "count": count,
                        "best_exec_time": group_normalizers[result_set],
                        "all_sqls_in_group": group_sqls,
                    }
                )

        high_conf_sqls.sort(key=lambda item: item["confidence"], reverse=True)

        print("\n  Top 5 High-Confidence Queries:")
        for rank, query_result in enumerate(high_conf_sqls[:5], 1):
            print(
                f"    {rank}. [Conf: {query_result['confidence']:.2f}, "
                f"Time: {query_result['best_exec_time']:.3f}s] "
                f"{query_result['sql'][:150]}..."
            )

        high_conf_candidates = [candidate for candidate in high_conf_sqls if candidate["confidence"] > 0.2]
        selected_sql = None
        selection_reasoning = None
        is_correct = False

        if len(high_conf_candidates) == 1:
            print("\n  Only one high-confidence candidate - skipping selection stage")
            selected_sql = high_conf_candidates[0]["sql"]
            selection_reasoning = "Single candidate - no selection needed"
            representative_sql = selected_sql
        elif len(high_conf_candidates) > 1:
            print("\n  Running SQL Selection Phase...")
            selection_candidates = high_conf_candidates[:5]
            candidate_sqls_text = "\n".join(
                f"{index + 1}. {candidate['sql']}" for index, candidate in enumerate(selection_candidates)
            )

            selection_prompt = build_spider_selection_prompt(
                schema_text=schema_text,
                question=question,
                candidate_sqls_text=candidate_sqls_text,
            )

            print(f"    Candidates (confidence > 0.2): {len(selection_candidates)}")
            selection_prompts = [selection_prompt] * config.MAJORITY_VOTE_N
            print(f"    Generating {len(selection_prompts)} selection responses...")

            selection_responses = generate_batch_in_chunks(
                llm_client=llm_client,
                prompts=selection_prompts,
                batch_size=max(1, generation_batch_size),
                error_logger=error_logger,
                phase="SQL_SELECTION",
                question_id=q_idx,
            )

            del selection_prompts
            gc.collect()
            torch.cuda.empty_cache()

            selection_votes = []
            print(f"    Parsing {len(selection_responses)} selection responses...")
            for selection_index, selection_response in enumerate(selection_responses):
                sql = extract_sql_from_response(selection_response)
                if sql:
                    selection_votes.append({"sql": sql, "reasoning": ""})
                    print(f"      Sample {selection_index + 1}: {sql[:80]}...")

            if selection_votes:
                sql_counts = Counter(vote["sql"] for vote in selection_votes)
                most_common_sql, vote_count = sql_counts.most_common(1)[0]
                selected_sql = most_common_sql
                selection_reasoning = f"Majority vote ({vote_count}/{len(selection_votes)} votes)"
                representative_sql = selected_sql
                print(
                    f"    Majority vote: {selected_sql[:150]}... "
                    f"({vote_count}/{len(selection_votes)} votes)"
                )
            else:
                print("    No valid selection responses, keeping majority vote result")
        else:
            print("    No high-confidence candidates (confidence > 0.2) for selection")

        if selected_sql:
            selected_success, selected_res_set, _ = execute_sql_with_timeout(db_path, selected_sql)
            is_correct = gt_success and selected_success and selected_res_set == gt_res_set
            print(f"    Selected SQL correctness: {'CORRECT' if is_correct else 'INCORRECT'}")
        else:
            majority_success, majority_res_set, _ = execute_sql_with_timeout(db_path, representative_sql)
            is_correct = gt_success and majority_success and majority_res_set == gt_res_set
            print(f"\n  Majority Vote Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            print(f"    Confidence: {winner_confidence:.2f} ({top_count}/{n_valid})")

        difficulty_results[difficulty].append(is_correct)
        execution_times = [item["exec_time"] for item in all_executions if "exec_time" in item]

        results_detail.append(
            {
                "question_id": q.get("question_id", global_q_idx),
                "question": question,
                "db_id": db_id,
                "difficulty": difficulty,
                "ground_truth": ground_truth,
                "winner_sql": representative_sql,
                "is_correct": is_correct,
                "winner_confidence": winner_confidence,
                "execution_times": execution_times,
                "high_confidence_alternatives": high_conf_sqls,
                "selection": {
                    "selected_sql": selected_sql if len(high_conf_candidates) > 0 else None,
                    "reasoning": selection_reasoning if len(high_conf_candidates) > 0 else None,
                    "candidates_count": len(high_conf_candidates),
                },
                "metrics": {
                    "generated": len(generated_candidates),
                    "execution_errors": execution_errors,
                    "valid_generations": n_valid,
                    "unique_valid_sqls": len(set(item["sql"] for item in all_executions)),
                },
            }
        )

        with open(results_file, "w", encoding="utf-8") as file_handle:
            json.dump(results_detail, file_handle, indent=2)

        if "generated_candidates" in locals():
            del generated_candidates
        if "all_executions" in locals():
            del all_executions
        if "sql_to_result_cache" in locals():
            del sql_to_result_cache
        if "selection_responses" in locals():
            del selection_responses
        if "selection_votes" in locals():
            del selection_votes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nBenchmark complete.")


def run_spider_benchmark_single_gpu(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    start: int = None,
    end: int = None,
    gpu_id: int = None,
    faiss_index: Optional[str] = None,
    faiss_index_masked: Optional[str] = None,
    generation_batch_size: int = 8,
    retrieval_k: int = 20,
):
    questions = load_benchmark(benchmark_path)
    total_questions = len(questions)

    if start and start > 0:
        print(f"Starting from question index {start}...")
        questions = questions[start:]
    if end and end > 0:
        print(f"Limiting to question index {end} (exclusive)...")
        questions = questions[: end - (start if start else 0)]

    print(f"Processing {len(questions)} questions (from index {start or 0} to {end or total_questions})")

    if start or end:
        subdir_parts = []
        if start and start > 0:
            subdir_parts.append(f"start_{start}")
        if end and end > 0:
            subdir_parts.append(f"end_{end}")
        output_dir = os.path.join(output_dir, "_".join(subdir_parts))

    gpu_output_dir = os.path.join(output_dir, f"gpu_{gpu_id if gpu_id is not None else 0}")
    os.makedirs(gpu_output_dir, exist_ok=True)

    run_spider_benchmark(
        benchmark_path=benchmark_path,
        db_root=db_root,
        output_dir=gpu_output_dir,
        limit=None,
        gpu_id=gpu_id,
        questions_chunk=questions,
        start_index=start if start else 0,
        faiss_index=faiss_index,
        faiss_index_masked=faiss_index_masked,
        generation_batch_size=generation_batch_size,
        retrieval_k=retrieval_k,
    )


def gpu_worker(
    gpu_id,
    benchmark_path,
    db_root,
    output_dir,
    questions_chunk,
    start_index=0,
    faiss_index=None,
    faiss_index_masked=None,
    generation_batch_size=8,
    retrieval_k=20,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.empty_cache()
    gc.collect()

    run_spider_benchmark(
        benchmark_path=benchmark_path,
        db_root=db_root,
        output_dir=output_dir,
        limit=None,
        gpu_id=0,
        questions_chunk=questions_chunk,
        start_index=start_index,
        faiss_index=faiss_index,
        faiss_index_masked=faiss_index_masked,
        generation_batch_size=generation_batch_size,
        retrieval_k=retrieval_k,
    )


def run_multi_gpu_benchmark(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    start: int = None,
    end: int = None,
    num_gpus: int = 4,
    faiss_index: Optional[str] = None,
    faiss_index_masked: Optional[str] = None,
    generation_batch_size: int = 4,
    retrieval_k: int = 20,
):
    import multiprocessing as mp
    config = Config()
    model_name = config.LLM_MODEL_NAME.lower()
    is_120b = "120b" in model_name

    if is_120b:
        print(
            "\n[WARN] 120B model detected. "
            "This runner uses one process per GPU for data parallelism, "
            "but 120B needs all GPUs visible to a single process for model parallelism."
        )
        print("[WARN] Falling back to single-process model-parallel execution.")
        run_spider_benchmark_single_gpu(
            benchmark_path=benchmark_path,
            db_root=db_root,
            output_dir=output_dir,
            start=start,
            end=end,
            gpu_id=None,
            faiss_index=faiss_index,
            faiss_index_masked=faiss_index_masked,
            generation_batch_size=generation_batch_size,
            retrieval_k=retrieval_k,
        )
        return

    mp.set_start_method("spawn", force=True)

    gpu_ids = setup_multi_gpu(num_gpus)
    num_gpus = len(gpu_ids)

    questions = load_benchmark(benchmark_path)
    if start and start > 0:
        print(f"Resuming from question index {start}...")
        questions = questions[start:]
    if end and end > 0:
        print(f"Limiting to question index {end} (exclusive)...")
        questions = questions[:end]

    if start or end:
        subdir_parts = []
        if start and start > 0:
            subdir_parts.append(f"start_{start}")
        if end and end > 0:
            subdir_parts.append(f"end_{end}")
        output_dir = os.path.join(output_dir, "_".join(subdir_parts))

    print(f"\nTotal questions: {len(questions)}")
    print(f"Distributing across {num_gpus} GPUs...")
    print(f"Output directory: {output_dir}")

    chunk_size = (len(questions) + num_gpus - 1) // num_gpus
    question_chunks = []
    chunk_start_indices = []
    for index in range(num_gpus):
        start_idx = index * chunk_size
        end_idx = min(start_idx + chunk_size, len(questions))
        if start_idx < len(questions):
            question_chunks.append(questions[start_idx:end_idx])
            chunk_start_indices.append(start + start_idx if start else start_idx)
        else:
            question_chunks.append([])
            chunk_start_indices.append(0)

    for index, chunk in enumerate(question_chunks):
        print(f"  GPU {index}: {len(chunk)} questions (start index: {chunk_start_indices[index]})")

    gpu_output_dirs = []
    for index in range(num_gpus):
        gpu_output_dir = os.path.join(output_dir, f"gpu_{index}")
        os.makedirs(gpu_output_dir, exist_ok=True)
        gpu_output_dirs.append(gpu_output_dir)

    print(f"\nStarting {num_gpus} parallel benchmark processes...")

    processes = []
    for index in range(num_gpus):
        if question_chunks[index]:
            process = mp.Process(
                target=gpu_worker,
                args=(
                    gpu_ids[index],
                    benchmark_path,
                    db_root,
                    gpu_output_dirs[index],
                    question_chunks[index],
                    chunk_start_indices[index],
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

    print("\nAll GPU processes completed!")

    print("\nMerging results from all GPUs...")
    all_results = []
    for index, gpu_output_dir in enumerate(gpu_output_dirs):
        results_path = os.path.join(gpu_output_dir, "benchmark_results.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as file_handle:
                gpu_results = json.load(file_handle)
                all_results.extend(gpu_results)
                print(f"  GPU {index}: {len(gpu_results)} results")

    merged_output_file = os.path.join(output_dir, "benchmark_results_merged.json")
    with open(merged_output_file, "w", encoding="utf-8") as file_handle:
        json.dump(all_results, file_handle, indent=2)
    print(f"  Merged results saved to: {merged_output_file}")


def main():
    parser = argparse.ArgumentParser(description="Spider benchmark runner for MCS-SQL")
    parser.add_argument("--benchmark", required=True, help="Path to spider dev JSON")
    parser.add_argument("--db_root", required=True, help="Path to Spider databases root")
    parser.add_argument("--output", default="outputs/spider_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=4)
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

    if args.multi_gpu:
        run_multi_gpu_benchmark(
            args.benchmark,
            args.db_root,
            args.output,
            args.start,
            args.end,
            args.num_gpus,
            args.faiss_index,
            args.faiss_index_masked,
            args.generation_batch_size,
            args.retrieval_k,
        )
    else:
        run_spider_benchmark_single_gpu(
            args.benchmark,
            args.db_root,
            args.output,
            args.start,
            args.end,
            args.gpu_id,
            args.faiss_index,
            args.faiss_index_masked,
            args.generation_batch_size,
            args.retrieval_k,
        )


if __name__ == "__main__":
    main()
