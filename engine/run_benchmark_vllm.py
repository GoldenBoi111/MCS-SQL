"""
vLLM Benchmark Runner for MCS-SQL - 120B Model

This script runs the MCS-SQL benchmark using vLLM with tensor parallelism.
Designed for 120B models on 4×A100 80GB GPUs.

IMPORTANT: This uses 1 model distributed across 4 GPUs (tensor parallel),
NOT 4 separate model copies. All 4 GPUs work together on each question.

Usage:
    # Test with 1 question first (RECOMMENDED)
    python engine/run_benchmark_vllm.py \
        --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
        --db_root minidev/MINIDEV/dev_databases/ \
        --output outputs/benchmark_vllm_120b \
        --limit 1

    # Full benchmark (500 questions)
    python engine/run_benchmark_vllm.py \
        --benchmark minidev/MINIDEV/mini_dev_sqlite.json \
        --db_root minidev/MINIDEV/dev_databases/ \
        --output outputs/benchmark_vllm_120b \
        --tensor-parallel-size 4
"""

import argparse
import gc
import json
import logging
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import random
import re
import sqlite3
import statistics
import time
import threading
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch

from json_schemas import SQL_GENERATION_SCHEMA, SQL_SELECTION_SCHEMA
from config import Config
from literal_masker import LiteralMasker
from schema_linking import SchemaLinker
from training_dataset_indexer import TrainingDatasetIndexer
from training_dataset_indexer_masked import MaskedTrainingDatasetIndexer
from error_logger import ErrorLogger, MemoryManager, check_gpu_memory, clear_gpu_memory

# Multi-GPU setup
def setup_multi_gpu(num_gpus: int = 4):
    """Setup multi-GPU environment and return list of GPU IDs."""
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    if available_gpus < num_gpus:
        print(f"Warning: Requested {num_gpus} GPUs, but only {available_gpus} available")
        num_gpus = available_gpus

    gpu_ids = list(range(num_gpus))
    print(f"Using GPUs: {gpu_ids}")

    for i in gpu_ids:
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    return gpu_ids


logger = logging.getLogger(__name__)

# vLLM support
try:
    from vllm_model_manager import vLLMModelManager, vLLMAPIClient, create_vllm_manager
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not installed. Install with: pip install vllm")


def load_benchmark(json_path: str) -> List[Dict[str, Any]]:
    """Load benchmark questions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def execute_sql_with_timeout(
    db_path: str, 
    sql: str, 
    timeout: int = 5
) -> Tuple[bool, frozenset, float]:
    """Execute SQL and return (success, result_frozenset, execution_time)."""
    start_time = time.time()
    result = {"success": False, "results": frozenset(), "error": None}
    conn = None
    
    def execute_query():
        nonlocal conn
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            result["success"] = True
            result["results"] = frozenset(results)
            conn.close()
        except Exception as e:
            result["error"] = str(e)
            if conn:
                conn.close()
    
    thread = threading.Thread(target=execute_query)
    thread.start()
    thread.join(timeout=timeout)
    
    exec_time = time.time() - start_time
    
    if thread.is_alive():
        if conn:
            try:
                conn.interrupt()
            except:
                pass
        thread.join(timeout=1)
        return False, frozenset(), exec_time
    
    if result["success"]:
        return True, result["results"], exec_time
    else:
        return False, frozenset(), exec_time


def build_examples_text(examples: List[Dict[str, Any]]) -> str:
    """Format the retrieved examples for the prompt."""
    parts = ["<examples>"]
    for ex in examples:
        q_text = ex.get('orig_question', ex.get('question', ''))
        sql_text = ex.get('orig_sql', ex.get('sql', ''))
        q_text = q_text.replace('{', '{{').replace('}', '}}')
        sql_text = sql_text.replace('{', '{{').replace('}', '}}')
        parts.append(f"# Question: {q_text}")
        parts.append(f"# Gold SQL: {sql_text}")
        if 'metadata' in ex and ex['metadata'].get('evidence'):
            evidence = ex['metadata']['evidence'].replace('{', '{{').replace('}', '}}')
            parts.append(f"# Evidence: {evidence}")
        parts.append("")
    parts.append("</examples>")
    return "\n".join(parts)


def get_sample_table_contents(
    db_path: str, 
    tables: List[str], 
    sample_size: int = 3
) -> str:
    """Get sample contents from each table in CSV format."""
    parts = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for table in tables:
            try:
                cursor.execute(f'SELECT * FROM "{table}" LIMIT {sample_size}')
                rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                parts.append(f"Table: {table}")
                parts.append(" | ".join(column_names))
                parts.append("-" * 50)
                for row in rows:
                    parts.append(" | ".join(str(val) if val is not None else "NULL" for val in row))
                parts.append("")
            except Exception as e:
                parts.append(f"Table: {table} (Error sampling: {e})")
                parts.append("")
        conn.close()
    except Exception as e:
        parts.append(f"Error connecting to database: {e}")
    return "\n".join(parts)


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response with robust error handling."""
    try:
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
        
        for i, char in enumerate(response[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
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
                        end_idx = i + 1
                        break
        
        if end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            if json_str.rstrip().endswith("```"):
                json_str = json_str.rstrip()[:-3]
            return json.loads(json_str)
    except Exception as e:
        print(f"    JSON parse error: {e}")
    
    return None


def extract_sql_from_response(response: str) -> Optional[str]:
    """Extract SQL query from LLM response using multiple methods."""
    # Method 1: Parse JSON
    parsed = parse_json_response(response)
    if parsed and "sql" in parsed:
        return parsed["sql"]
    
    # Method 2: Regex fallback
    match = re.search(r'SELECT.*?(?:;|$)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    
    return None


def run_vllm_benchmark(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    vllm_model: Optional[vLLMModelManager] = None,
    vllm_url: Optional[str] = None,
    tensor_parallel_size: int = 4,
    limit: int = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
):
    """
    Run benchmark using vLLM for high-throughput generation.
    
    Args:
        benchmark_path: Path to benchmark JSON file
        db_root: Path to database root directory
        output_dir: Output directory for results
        vllm_model: Pre-initialized vLLM model manager (optional)
        vllm_url: URL to running vLLM API server (optional)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        limit: Optional limit on number of questions
        start_index: Starting index for resuming
        end_index: Optional end index (exclusive)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load existing results if they exist (for appending)
    results_file = os.path.join(output_dir, "benchmark_results.json")
    results_detail = []
    completed_qids = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_detail = json.load(f)
        completed_qids = set(r.get("question_id") for r in results_detail if r.get("question_id"))
        print(f"Loaded {len(results_detail)} existing results from {results_file}")
        print(f"Completed question_ids: {sorted(completed_qids)}")
        print("Already-completed questions will be skipped")

    # Initialize error logger and memory manager
    error_logger = ErrorLogger(output_dir)
    memory_manager = MemoryManager(threshold_gb=5.0, errors_dir=output_dir)

    # Initialize vLLM model
    if vllm_model is None:
        if vllm_url:
            print(f"Connecting to vLLM API server at {vllm_url}...")
            config = Config()
            vllm_client = vLLMAPIClient(
                base_url=vllm_url,
                model_name=config.LLM_MODEL_NAME,
            )
        else:
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not installed. Install with: pip install vllm")
            
            config = Config()
            print(f"\nInitializing vLLM model with tensor parallel size {tensor_parallel_size}...")
            vllm_client = create_vllm_manager(
                model_name=config.LLM_MODEL_NAME,
                num_gpus=tensor_parallel_size,
                max_tokens=config.LLM_MAX_NEW_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
    else:
        vllm_client = vllm_model
    
    config = Config()
    print("\nResolved configuration:")
    print(f"  FAISS_INDEX: {config.FAISS_INDEX}")
    print(f"  FAISS_INDEX_MASKED: {config.FAISS_INDEX_MASKED}")
    print(f"  PROMPTS_DIR: {config.PROMPTS_DIR}")
    
    # Setup schema linker with vLLM
    # Note: Schema linker needs JSON support, so we use the vLLM client directly
    linker = SchemaLinker(
        pt=config.TABLE_LINKING_ITERATIONS,
        pc=config.COLUMN_LINKING_ITERATIONS,
        n=config.MAJORITY_VOTE_N,
        llm_client=vllm_client,
    )
    
    # Load Indexes
    print("Loading Standard Index...")
    standard_indexer = TrainingDatasetIndexer(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        index_type=config.FAISS_INDEX_TYPE
    )
    standard_indexer.load(config.FAISS_INDEX)
    
    print("Loading Masked Index...")
    masked_indexer = MaskedTrainingDatasetIndexer(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        index_type=config.FAISS_INDEX_TYPE
    )
    masked_indexer.load(config.FAISS_INDEX_MASKED)
    
    # Literal masker
    literal_masker = LiteralMasker(llm_client=vllm_client)
    
    # Load prompt templates
    with open(config.PROMPTS_DIR / "SQL_generation.txt", "r") as f:
        prompt_template = f.read()
    
    with open(config.PROMPTS_DIR / "SQL_selection.txt", "r") as f:
        selection_template = f.read()
    
    # Load questions
    questions = load_benchmark(benchmark_path)
    
    # Apply start/end indices
    if start_index > 0:
        print(f"Resuming from question index {start_index}...")
        questions = questions[start_index:]
    if end_index and end_index > 0:
        print(f"Limiting to question index {end_index} (exclusive)...")
        questions = questions[:end_index - start_index]
    
    if limit:
        questions = questions[:limit]
    
    print(f"Loaded {len(questions)} questions")
    
    difficulty_results = {
        "simple": [],
        "moderate": [],
        "challenging": [],
        "unknown": []
    }
    
    for q_idx, q in enumerate(questions):
        global_q_idx = start_index + q_idx
        
        db_id = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "")
        ground_truth = q["SQL"]
        difficulty = q.get("difficulty", "unknown")
        
        # Skip already-completed questions
        qid = q.get("question_id", global_q_idx)
        if qid in completed_qids:
            print(f"\n[{global_q_idx+1}/{len(questions)}] Skipping completed question (ID: {qid})")
            continue
        
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        print(f"\n[{global_q_idx+1}/{len(questions)}] Q: {question[:80]}...")
        
        if not os.path.exists(db_path):
            print(f"  Warning: DB not found at {db_path}")
            continue
        
        # 1. Schema Linking
        print("  Running Schema Linking...")
        t0 = time.time()
        full_schema = linker.load_schema(db_path)

        # Check memory before schema linking
        is_low, free_gb, allocated_gb = check_gpu_memory(threshold_gb=8.0)
        if is_low:
            print(f"  [WARNING] GPU memory low before schema linking: {free_gb:.2f}GB free")
            clear_gpu_memory(verbose=True)
        
        try:
            linking_res = linker.link_schema(full_schema, question, evidence)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\n[CUDA OOM] Schema linking failed...")
                error_logger.log_cuda_error(e, {
                    "phase": "SCHEMA_LINKING",
                    "question_id": qid,
                    "db_id": db_id,
                })
                linking_res = type('obj', (object,), {'tables': [], 'columns': []})
            else:
                raise
        
        # Filter tables: keep only tables that exist in the database schema
        all_db_tables = set(full_schema.keys())
        original_tables = linking_res.tables
        linking_res.tables = [t for t in linking_res.tables if t in all_db_tables]
        
        # Log if any tables were removed
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
        
        linked_schema_dict = {t: full_schema[t] for t in linking_res.tables if t in full_schema}
        schema_text = linker.format_schema_for_prompt(linked_schema_dict)
        print(f"  Schema Linking took {time.time() - t0:.2f}s ({len(linking_res.tables)} tables, {len(linking_res.columns)} columns)")
        
        del full_schema
        del linked_schema_dict
        
        # 2. Retrieve Examples
        print("  Retrieving examples from FAISS...")
        k = 20
        standard_results = standard_indexer.search(question, top_k=k)
        masked_q = literal_masker.mask_question(question)
        masked_results = masked_indexer.search(masked_q, top_k=k)
        
        std_examples = [
            {"question": sq, "sql": sql, "metadata": meta}
            for (sq, sql, meta, score) in standard_results
        ]
        msk_examples = [
            {"question": mq, "sql": msql, "orig_question": oq, "orig_sql": osql, "metadata": meta}
            for (mq, oq, msql, osql, meta, score) in masked_results
        ]
        
        del standard_results
        del masked_results
        del masked_q
        
        # 3. Build 5 Prompt Variations
        prompt_variations = [
            ("masked_only", msk_examples[:10]),
            ("standard_only", std_examples[:10]),
        ]
        for i in range(3):
            mixed = std_examples[:5] + msk_examples[:5]
            rem_pool = std_examples[5:] + msk_examples[5:]
            if rem_pool:
                mixed += random.sample(rem_pool, min(10 - len(mixed), len(rem_pool)))
            random.shuffle(mixed)
            prompt_variations.append((f"mixed_{i}", mixed))
        
        del std_examples
        del msk_examples
        
        # 4. Generate 100 SQL candidates using vLLM batch generation
        print("  Generating SQL candidates (5 × 20 = 100)...")
        generated_candidates = []
        
        sample_contents = get_sample_table_contents(db_path, list(linking_res.tables), sample_size=3)
        
        # Build all 100 prompts
        all_prompts = []
        prompt_metadata = []
        
        for p_name, ex_list in prompt_variations:
            ex_text = build_examples_text(ex_list)
            base_prompt = (
                prompt_template
                .replace("{examples}", ex_text)
                .replace("{schema_text}", schema_text)
                .replace("{sample_contents}", sample_contents)
                .replace("{question}", question)
                .replace("{evidence}", evidence)
            )
            
            for gen_idx in range(config.MAJORITY_VOTE_N):
                all_prompts.append(base_prompt)
                prompt_metadata.append((p_name, gen_idx))
        
        del prompt_variations

        # Generate all responses using vLLM's continuous batching with guided JSON
        print(f"  Running vLLM batch generation with JSON schema ({len(all_prompts)} prompts)...")

        if hasattr(vllm_client, 'generate_json_batch'):
            # vLLMModelManager - use guided JSON with SQL_GENERATION_SCHEMA
            all_json_results = vllm_client.generate_json_batch(
                all_prompts,
                json_schema=SQL_GENERATION_SCHEMA,
                batch_size=32,
                show_progress=True,
            )
        else:
            # vLLMAPIClient - generate one by one, parse manually
            all_json_results = []
            for prompt in all_prompts:
                response = vllm_client.generate(prompt)
                parsed = parse_json_response(response)
                if parsed:
                    all_json_results.append(parsed)
                else:
                    all_json_results.append({"error": "JSON parse failed", "raw_response": response[:200]})

        del all_prompts

        # Extract SQL from guided JSON results
        print("  Extracting SQL from JSON results...")
        for i, json_result in enumerate(all_json_results):
            p_name, gen_idx = prompt_metadata[i]
            sql_query = json_result.get("sql", "")

            if sql_query:
                generated_candidates.append({
                    "sql": sql_query,
                    "prompt_type": p_name,
                    "gen_idx": gen_idx
                })

        del all_json_results
        del prompt_metadata
        
        print(f"  Generated {len(generated_candidates)} valid SQL candidates")
        
        # 5. Execute and Majority Voting
        all_executions = []
        execution_errors = 0
        sql_to_result_cache = {}
        
        print("  Executing candidates...")
        for cand in generated_candidates:
            sql = cand["sql"]
            if sql in sql_to_result_cache:
                success, res_val, exec_time = sql_to_result_cache[sql]
            else:
                success, res_val, exec_time = execute_sql_with_timeout(db_path, sql)
                sql_to_result_cache[sql] = (success, res_val, exec_time)
            
            if success:
                all_executions.append({
                    "sql": sql,
                    "result_str": res_val,
                    "exec_time": exec_time,
                })
            else:
                execution_errors += 1
        
        N_valid = len(all_executions)
        print(f"  Valid executions: {N_valid} (Errors: {execution_errors})")
        
        if N_valid == 0:
            print("  No queries executed successfully. Saving failure result...")

            # Save comprehensive failure result with all available info
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
                    "candidates_count": 0
                },
                "metrics": {
                    "generated": len(generated_candidates),
                    "execution_errors": execution_errors,
                    "valid_generations": 0,
                    "unique_valid_sqls": 0
                },
                "failure_info": {
                    "phase": "EXECUTION",
                    "reason": "No queries executed successfully - all generated SQL had syntax errors or timeouts",
                    "generated_candidates_sample": [
                        {"sql": cand["sql"], "prompt_type": cand["prompt_type"]}
                        for cand in generated_candidates[:10]
                    ] if generated_candidates else [],
                    "error_count": execution_errors
                }
            }

            results_detail.append(failure_result)

            # Save intermediate results immediately on failure
            with open(results_file, "w") as f:
                json.dump(results_detail, f, indent=2)
            print(f"  Saved failure result to {results_file}")

            # Track as incorrect for difficulty stats
            difficulty_results[difficulty].append(False)

            # Continue to next question with memory cleanup
            del generated_candidates
            del all_executions
            del sql_to_result_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        # Group executions by result_set and track best (minimum) execution time per group
        result_groups = {}  # result_frozenset -> list of {sql, exec_time}
        for exec_item in all_executions:
            res_set = exec_item["result_str"]
            if res_set not in result_groups:
                result_groups[res_set] = []
            result_groups[res_set].append({
                "sql": exec_item["sql"],
                "exec_time": exec_item["exec_time"]
            })

        # Find best (minimum) execution time per group - this is the normalizer
        group_normalizers = {}  # result_frozenset -> best_exec_time
        for res_set, items in result_groups.items():
            group_normalizers[res_set] = min(item["exec_time"] for item in items)

        # Calculate confidence for each execution using the formula:
        # confidence(qi) = 1/N * sum from j=1 to N of (exec(qi) = exec(qj))
        # where N is the number of valid executions (excluding timeouts and syntax errors)
        # This simplifies to: confidence = count(result_i) / N
        execution_confidences = []  # list of {sql, result_str, confidence, exec_time}
        for exec_item in all_executions:
            res_set = exec_item["result_str"]
            count_same_result = len(result_groups[res_set])
            confidence = count_same_result / N_valid
            execution_confidences.append({
                "sql": exec_item["sql"],
                "result_str": res_set,
                "confidence": confidence,
                "exec_time": exec_item["exec_time"],
                "normalized_by": group_normalizers[res_set]
            })

        # Find the result group with highest confidence (most common result)
        result_counts = Counter(item["result_str"] for item in all_executions)
        most_common_result, top_count = result_counts.most_common(1)[0]

        # Find all unique SQLs that produced the winning result
        winning_sqls = list(set(
            item["sql"] for item in all_executions
            if item["result_str"] == most_common_result
        ))

        # Pick the shortest winning SQL as the representative
        representative_sql = min(winning_sqls, key=len)

        # Execution evaluation against ground truth (for reference, not final verdict)
        gt_success, gt_res_set, gt_time = execute_sql_with_timeout(db_path, ground_truth)
        winner_confidence = top_count / N_valid

        # Collect queries with confidence > 0.2, grouped by result with best speed as normalizer
        high_conf_sqls = []
        processed_results = set()
        for res_set, count in result_counts.items():
            conf = count / N_valid
            if conf > 0.2 and res_set not in processed_results:
                processed_results.add(res_set)
                # Find all unique SQLs for this result group
                group_sqls = list(set(
                    item["sql"] for item in all_executions
                    if item["result_str"] == res_set
                ))
                # Pick representative (shortest SQL)
                rep = min(group_sqls, key=len)
                high_conf_sqls.append({
                    "sql": rep,
                    "confidence": conf,
                    "count": count,
                    "best_exec_time": group_normalizers[res_set],
                    "all_sqls_in_group": group_sqls
                })

        # Sort by confidence descending
        high_conf_sqls.sort(key=lambda x: x["confidence"], reverse=True)

        print("\n  Top 5 High-Confidence Queries:")
        for i, q_res in enumerate(high_conf_sqls[:5], 1):
            print(f"    {i}. [Conf: {q_res['confidence']:.2f}, Time: {q_res['best_exec_time']:.3f}s] {q_res['sql'][:150]}...")

        # SQL Selection Phase: Use LLM to select the best SQL from all high-confidence candidates
        # Following the paper: present candidates as multiple-choice, sample n responses, majority vote
        # Use ALL candidates that pass the confidence threshold (> 0.2), not just top 3
        high_conf_candidates = [c for c in high_conf_sqls if c['confidence'] > 0.2]

        selected_sql = None
        selection_reasoning = None
        is_correct = False  # Will be set after selection

        # Skip selection stage if only one valid candidate - use it directly
        if len(high_conf_candidates) == 1:
            print("\n  Only one high-confidence candidate - skipping selection stage")
            selected_sql = high_conf_candidates[0]['sql']
            selection_reasoning = "Single candidate - no selection needed"
            representative_sql = selected_sql
        elif len(high_conf_candidates) > 1:
            print("\n  Running SQL Selection Phase...")
            # Format candidate SQLs as numbered list (multiple-choice format)
            # Max 5 can pass 0.2 threshold (mathematically), typically 1-3
            selection_candidates = high_conf_candidates[:5]
            candidate_sqls_text = "\n".join(
                f"{i+1}. {c['sql']}" for i, c in enumerate(selection_candidates)
            )

            selection_prompt = (
                selection_template
                .replace("{schema_text}", schema_text)
                .replace("{question}", question)
                .replace("{evidence}", evidence)
                .replace("{candidate_sqls}", candidate_sqls_text)
            )

            print(f"    Candidates (confidence > 0.2): {len(selection_candidates)}")

            # Sample config.MAJORITY_VOTE_N responses from LLM for majority voting
            n_selection_samples = config.MAJORITY_VOTE_N
            selection_votes = []

            print(f"    Generating {n_selection_samples} selection responses...")

            # Create 20 copies of the selection prompt
            selection_prompts = [selection_prompt] * n_selection_samples

            # Generate all 20 responses using vLLM with guided JSON
            selection_responses_json = []

            try:
                if hasattr(vllm_client, 'generate_json_batch'):
                    # vLLMModelManager - use guided JSON with SQL_SELECTION_SCHEMA
                    selection_responses_json = vllm_client.generate_json_batch(
                        selection_prompts,
                        json_schema=SQL_SELECTION_SCHEMA,
                        batch_size=16,
                        show_progress=False,
                    )
                else:
                    # vLLMAPIClient - generate one by one, parse manually
                    for prompt in selection_prompts:
                        response = vllm_client.generate(prompt)
                        parsed = parse_json_response(response)
                        if parsed:
                            selection_responses_json.append(parsed)
                        else:
                            selection_responses_json.append({"error": "JSON parse failed"})
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\n[CUDA OOM] Selection failed, attempting recovery...")

                    error_logger.log_cuda_error(e, {
                        "phase": "SQL_SELECTION",
                        "question_id": qid,
                    })

                    # Retry with smaller batch
                    try:
                        clear_gpu_memory(verbose=True)
                        if hasattr(vllm_client, 'generate_json_batch'):
                            selection_responses_json = vllm_client.generate_json_batch(
                                selection_prompts,
                                json_schema=SQL_SELECTION_SCHEMA,
                                batch_size=8,
                                show_progress=False,
                            )
                        else:
                            for prompt in selection_prompts:
                                response = vllm_client.generate(prompt)
                                parsed = parse_json_response(response)
                                if parsed:
                                    selection_responses_json.append(parsed)
                                else:
                                    selection_responses_json.append({"error": "JSON parse failed"})
                        print("  Selection recovery successful!")
                    except Exception as recovery_error:
                        error_logger.log_cuda_error(recovery_error, {
                            "phase": "SQL_SELECTION_RECOVERY",
                            "question_id": qid,
                        })
                        selection_responses_json = [{"error": "OOM recovery failed"}] * len(selection_prompts)
                else:
                    raise

            # Cleanup selection prompts immediately after responses are gotten
            del selection_prompts
            gc.collect()
            torch.cuda.empty_cache()

            # Extract SQL from guided JSON results
            print(f"    Extracting SQL from {len(selection_responses_json)} selection responses...")
            for sel_idx, json_result in enumerate(selection_responses_json):
                sql = json_result.get("sql", "")
                if sql:
                    selection_votes.append(sql)
                    print(f"      Sample {sel_idx+1}: {sql[:80]}...")

            # Majority voting on selection
            if selection_votes:
                sql_counts = Counter(selection_votes)
                most_common_sql, vote_count = sql_counts.most_common(1)[0]

                selected_sql = most_common_sql
                selection_reasoning = f"Majority vote ({vote_count}/{len(selection_votes)} votes)"

                print(f"    Majority vote: {selected_sql[:150]}... ({vote_count}/{len(selection_votes)} votes)")
                representative_sql = selected_sql

                # Cleanup selection data after majority voting
                del selection_votes
                del selection_responses_json
            else:
                print("    No valid selection responses, keeping majority vote result")
        else:
            print("    No high-confidence candidates (confidence > 0.2) for selection")

        # Re-evaluate correctness with the selected SQL using official BIRD EX metric
        if selected_sql:
            selected_success, selected_res_set, _ = execute_sql_with_timeout(db_path, selected_sql)
            # Official BIRD EX: compare sets (order-independent)
            is_correct = (gt_success and selected_success and selected_res_set == gt_res_set)
            print(f"    Selected SQL correctness: {'CORRECT' if is_correct else 'INCORRECT'}")
        else:
            # No selection was made, report on majority vote result
            majority_success, majority_res_set, _ = execute_sql_with_timeout(db_path, representative_sql)
            is_correct = (gt_success and majority_success and majority_res_set == gt_res_set)
            print(f"\n  Majority Vote Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            print(f"    Confidence: {winner_confidence:.2f} ({top_count}/{N_valid})")

        # Track results by difficulty
        difficulty_results[difficulty].append(is_correct)

        # Collect execution times for this question
        execution_times = [item["exec_time"] for item in all_executions if "exec_time" in item]

        results_detail.append({
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
                "candidates_count": len(high_conf_candidates)
            },
            "metrics": {
                "generated": len(generated_candidates),
                "execution_errors": execution_errors,
                "valid_generations": N_valid,
                "unique_valid_sqls": len(set(item["sql"] for item in all_executions))
            }
        })

        # Save intermediate results
        with open(results_file, "w") as f:
            json.dump(results_detail, f, indent=2)

        # ========== MEMORY CLEANUP TO PREVENT VRAM LEAKS ==========
        # Clear GPU memory after each question to prevent OOM

        # Delete large intermediate variables that are no longer needed
        if 'generated_candidates' in locals(): del generated_candidates
        if 'all_executions' in locals(): del all_executions
        if 'sql_to_result_cache' in locals(): del sql_to_result_cache
        if 'std_examples' in locals(): del std_examples
        if 'msk_examples' in locals(): del msk_examples
        if 'prompt_variations' in locals(): del prompt_variations
        if 'linking_res' in locals(): del linking_res
        if 'full_schema' in locals(): del full_schema
        if 'sample_contents' in locals(): del sample_contents
        if 'all_prompts' in locals(): del all_prompts
        if 'all_responses' in locals(): del all_responses
        if 'selection_prompts' in locals(): del selection_prompts
        if 'selection_responses_json' in locals(): del selection_responses_json
        # Now safe to delete these (selection phase is complete)
        if 'schema_text' in locals(): del schema_text
        if 'question' in locals(): del question
        if 'evidence' in locals(): del evidence
        if 'representative_sql' in locals(): del representative_sql
        if 'selection_reasoning' in locals(): del selection_reasoning
        if 'high_conf_sqls' in locals(): del high_conf_sqls
        if 'high_conf_candidates' in locals(): del high_conf_candidates
        if 'result_groups' in locals(): del result_groups
        if 'group_normalizers' in locals(): del group_normalizers
        if 'execution_confidences' in locals(): del execution_confidences
        if 'winning_sqls' in locals(): del winning_sqls
        if 'result_counts' in locals(): del result_counts
        if 'most_common_result' in locals(): del most_common_result
        if 'top_count' in locals(): del top_count
        if 'most_common_sql' in locals(): del most_common_sql
        if 'vote_count' in locals(): del vote_count
        if 'selected_sql' in locals(): del selected_sql
        if 'selected_success' in locals(): del selected_success
        if 'selected_res_set' in locals(): del selected_res_set
        if 'majority_success' in locals(): del majority_success
        if 'majority_res_set' in locals(): del majority_res_set
        if 'gt_success' in locals(): del gt_success
        if 'gt_res_set' in locals(): del gt_res_set
        if 'gt_time' in locals(): del gt_time
        if 'winner_confidence' in locals(): del winner_confidence
        if 'processed_results' in locals(): del processed_results
        if 'candidate_sqls_text' in locals(): del candidate_sqls_text
        if 'selection_prompt' in locals(): del selection_prompt
        if 'selection_votes' in locals(): del selection_votes
        if 'sql_counts' in locals(): del sql_counts
        if 'selected_reasoning' in locals(): del selected_reasoning
        if 'is_correct' in locals(): del is_correct
        if 'execution_times' in locals(): del execution_times
        if 'ex_text' in locals(): del ex_text
        if 'base_prompt' in locals(): del base_prompt

        # Force Python garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  [Memory Cleanup] Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

            # Check if memory is still high and log warning
            if allocated > 60.0:  # More than 60GB allocated
                print(f"  [WARNING] High memory usage detected!")
                error_logger.log_cuda_error(
                    Exception("High memory usage after cleanup"),
                    {
                        "phase": "POST_QUESTION_CLEANUP",
                        "question_id": qid,
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                    }
                )
        # ===========================================================

    # Final cleanup of persistent resources
    print("\nPerforming final cleanup of persistent resources...")

    # Delete primary controllers
    if 'linker' in locals(): del linker
    if 'standard_indexer' in locals(): del standard_indexer
    if 'masked_indexer' in locals(): del masked_indexer
    if 'literal_masker' in locals(): del literal_masker
    if 'vllm_client' in locals(): del vllm_client
    if 'error_logger' in locals(): del error_logger
    if 'memory_manager' in locals(): del memory_manager

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Final memory status: Allocated={torch.cuda.memory_allocated()/1e9:.2f}GB, Reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

    # Generate detailed report
    generate_detailed_report(results_detail, output_dir)


def generate_detailed_report(results: List[Dict], output_dir: str):
    """
    Generate comprehensive benchmark report with detailed statistics.

    Args:
        results: List of result dictionaries from all questions
        output_dir: Directory to save report
    """
    import statistics

    print("\n" + "="*100)
    print(" " * 30 + "DETAILED BENCHMARK REPORT")
    print("="*100)

    # Overall Statistics
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    overall_acc = (correct / total * 100) if total > 0 else 0

    print(f"\n📊 OVERALL STATISTICS")
    print("-"*100)
    print(f"  Total Questions:     {total}")
    print(f"  Correct:             {correct} ({overall_acc:.2f}%)")
    print(f"  Incorrect:           {total - correct} ({100 - overall_acc:.2f}%)")

    # Execution Time Statistics
    all_exec_times = []
    for r in results:
        if "execution_times" in r:
            all_exec_times.extend(r["execution_times"])

    if all_exec_times:
        print(f"\n⏱️ EXECUTION TIME STATISTICS (All SQL Queries)")
        print("-"*100)
        print(f"  Total Queries Executed:  {len(all_exec_times)}")
        print(f"  Mean Execution Time:     {statistics.mean(all_exec_times)*1000:.2f} ms")
        print(f"  Median Execution Time:   {statistics.median(all_exec_times)*1000:.2f} ms")
        print(f"  Std Dev:                 {statistics.stdev(all_exec_times)*1000:.2f} ms" if len(all_exec_times) > 1 else "  Std Dev: N/A")
        print(f"  Min Execution Time:      {min(all_exec_times)*1000:.2f} ms")
        print(f"  Max Execution Time:      {max(all_exec_times)*1000:.2f} ms")

    # Generation Statistics
    total_generated = sum(r.get("metrics", {}).get("generated", 0) for r in results)
    total_valid = sum(r.get("metrics", {}).get("valid_generations", 0) for r in results)
    total_errors = sum(r.get("metrics", {}).get("execution_errors", 0) for r in results)
    unique_sqls = sum(r.get("metrics", {}).get("unique_valid_sqls", 0) for r in results)

    print(f"\n🔧 GENERATION STATISTICS")
    print("-"*100)
    print(f"  Total SQL Generated:     {total_generated}")
    if total_generated > 0:
        print(f"  Valid Executions:        {total_valid} ({total_valid/total_generated*100:.1f}% success rate)")
        print(f"  Execution Errors:        {total_errors} ({total_errors/total_generated*100:.1f}%)")
        print(f"  Unique SQL Variations:   {unique_sqls} (avg {unique_sqls/total:.1f} per question)")
    else:
        print(f"  No SQL generated (check for errors)")

    # Confidence Statistics
    confidences = [r.get("winner_confidence", 0) for r in results]
    if confidences:
        print(f"\n📈 CONFIDENCE STATISTICS (Majority Voting)")
        print("-"*100)
        print(f"  Mean Confidence:         {statistics.mean(confidences):.3f}")
        print(f"  Median Confidence:       {statistics.median(confidences):.3f}")
        print(f"  High Confidence (>0.5):  {sum(1 for c in confidences if c > 0.5)} ({sum(1 for c in confidences if c > 0.5)/len(confidences)*100:.1f}%)")
        print(f"  Low Confidence (<0.2):   {sum(1 for c in confidences if c < 0.2)} ({sum(1 for c in confidences if c < 0.2)/len(confidences)*100:.1f}%)")

    # Difficulty Breakdown
    print(f"\n📚 ACCURACY BY DIFFICULTY")
    print("-"*100)
    difficulty_stats = {}
    for diff in ["simple", "moderate", "challenging", "unknown"]:
        diff_results = [r for r in results if r.get("difficulty") == diff]
        if diff_results:
            diff_correct = sum(1 for r in diff_results if r.get("is_correct", False))
            diff_total = len(diff_results)
            diff_acc = (diff_correct / diff_total * 100) if diff_total > 0 else 0
            difficulty_stats[diff] = {
                "correct": diff_correct,
                "total": diff_total,
                "accuracy": diff_acc
            }
            print(f"  {diff.capitalize():<15} {diff_correct:>3}/{diff_total:<3} ({diff_acc:>6.2f}%)")

    # Per-Database Breakdown
    print(f"\n🗄️  ACCURACY BY DATABASE")
    print("-"*100)
    db_results = {}
    for r in results:
        db_id = r.get("db_id", "unknown")
        if db_id not in db_results:
            db_results[db_id] = []
        db_results[db_id].append(r)

    # Sort databases by accuracy (descending)
    db_accuracy = []
    for db_id, db_qs in db_results.items():
        db_correct = sum(1 for r in db_qs if r.get("is_correct", False))
        db_total = len(db_qs)
        db_acc = (db_correct / db_total * 100) if db_total > 0 else 0
        db_accuracy.append((db_id, db_acc, db_correct, db_total))

    db_accuracy.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'Database':<40} {'Correct':>10} {'Total':>8} {'Accuracy':>10}")
    print("  " + "-"*70)
    for db_id, acc, correct_count, total_count in db_accuracy:
        db_name = db_id.split('/')[-1] if '/' in db_id else db_id
        print(f"  {db_name:<40} {correct_count:>10} {total_count:>8} {acc:>9.2f}%")

    # Selection Phase Statistics
    selection_made = sum(1 for r in results if r.get("selection", {}).get("selected_sql") is not None)
    print(f"\n🎯 SQL SELECTION PHASE STATISTICS")
    print("-"*100)
    if total > 0:
        print(f"  Selection Made:          {selection_made} ({selection_made/total*100:.1f}%)")
        print(f"  Majority Vote Used:      {total - selection_made} ({(total - selection_made)/total*100:.1f}%)")
    else:
        print(f"  No results to report")

    # Save detailed report to file
    report_data = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": overall_acc
        },
        "by_difficulty": difficulty_stats,
        "by_database": {},
        "execution_times": {
            "mean_ms": statistics.mean(all_exec_times)*1000 if all_exec_times else 0,
            "median_ms": statistics.median(all_exec_times)*1000 if all_exec_times else 0,
            "min_ms": min(all_exec_times)*1000 if all_exec_times else 0,
            "max_ms": max(all_exec_times)*1000 if all_exec_times else 0,
            "std_ms": statistics.stdev(all_exec_times)*1000 if len(all_exec_times) > 1 else 0
        },
        "generation": {
            "total_generated": total_generated,
            "valid_executions": total_valid,
            "errors": total_errors,
            "unique_sqls": unique_sqls
        },
        "confidence": {
            "mean": statistics.mean(confidences) if confidences else 0,
            "median": statistics.median(confidences) if confidences else 0,
            "high_confidence_ratio": sum(1 for c in confidences if c > 0.5)/len(confidences) if confidences else 0
        }
    }

    # Add per-database stats
    for db_id, db_qs in db_results.items():
        db_correct = sum(1 for r in db_qs if r.get("is_correct", False))
        db_total = len(db_qs)
        db_acc = (db_correct / db_total * 100) if db_total > 0 else 0
        report_data["by_database"][db_id] = {
            "total": db_total,
            "correct": db_correct,
            "accuracy": db_acc
        }

    report_file = os.path.join(output_dir, "detailed_report.json")
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n💾 Detailed report saved to: {report_file}")
    print("="*100)


# =============================================================================
# Multi-GPU Worker Functions (must be at module level for pickling)
# =============================================================================

def run_vllm_single_gpu(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    start: int = None,
    end: int = None,
    gpu_id: int = None,
    tensor_parallel_size: int = 4,
    vllm_url: str = None,
):
    """
    Run vLLM benchmark on a single GPU with start/end index support.

    Args:
        benchmark_path: Path to benchmark JSON file
        db_root: Path to database root directory
        output_dir: Output directory for results
        start: Start index (inclusive)
        end: End index (exclusive)
        gpu_id: Specific GPU ID to use
        tensor_parallel_size: Number of GPUs for tensor parallelism
        vllm_url: Optional vLLM API server URL
    """
    # Load all questions
    questions = load_benchmark(benchmark_path)
    total_questions = len(questions)

    # Apply start/end indices
    if start and start > 0:
        print(f"Starting from question index {start}...")
        questions = questions[start:]
    if end and end > 0:
        print(f"Limiting to question index {end} (exclusive)...")
        questions = questions[:end - (start if start else 0)]

    print(f"Processing {len(questions)} questions (from index {start or 0} to {end or total_questions})")

    # Build output subdirectory from start/end
    if start or end:
        subdir_parts = []
        if start and start > 0:
            subdir_parts.append(f"start_{start}")
        if end and end > 0:
            subdir_parts.append(f"end_{end}")
        output_dir = os.path.join(output_dir, "_".join(subdir_parts))

    # Set GPU ID for output directory
    gpu_output_dir = os.path.join(output_dir, f"gpu_{gpu_id if gpu_id is not None else 0}")
    os.makedirs(gpu_output_dir, exist_ok=True)

    # Run benchmark
    run_vllm_benchmark(
        benchmark_path=benchmark_path,
        db_root=db_root,
        output_dir=gpu_output_dir,
        tensor_parallel_size=tensor_parallel_size,
        vllm_url=vllm_url,
        start_index=start if start else 0,
        end_index=end,
    )


def gpu_worker(gpu_id, benchmark_path, db_root, output_dir, questions_chunk, start_index=0,
               tensor_parallel_size=4, vllm_url=None):
    """
    Worker function to run benchmark on a specific GPU.
    Must be at module level (not nested) for multiprocessing pickling.
    """
    import os

    # Set CUDA visible device BEFORE any torch operations
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # Run benchmark with gpu_id=0 since CUDA_VISIBLE_DEVICES makes it see only one GPU
    run_vllm_benchmark(
        benchmark_path=benchmark_path,
        db_root=db_root,
        output_dir=output_dir,
        tensor_parallel_size=tensor_parallel_size,
        vllm_url=vllm_url,
        limit=None,  # Already chunked
        start_index=start_index,
    )


def run_multi_gpu_benchmark(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    start: int = None,
    end: int = None,
    num_gpus: int = 4,
    tensor_parallel_size: int = 4,
    vllm_url: str = None,
):
    """
    Run benchmark across multiple GPUs using data parallelism.
    Each GPU processes a different subset of questions independently.

    Args:
        benchmark_path: Path to benchmark JSON file
        db_root: Path to database root directory
        output_dir: Output directory for results
        start: Optional start index (inclusive, for resuming)
        end: Optional end index (exclusive, for limiting)
        num_gpus: Number of GPUs to use
        tensor_parallel_size: Number of GPUs per vLLM model
        vllm_url: Optional vLLM API server URL
    """
    import multiprocessing as mp

    # CRITICAL: Use 'spawn' method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    # Setup GPUs
    gpu_ids = setup_multi_gpu(num_gpus)
    num_gpus = len(gpu_ids)

    # Load all questions
    questions = load_benchmark(benchmark_path)
    original_start = start if start and start > 0 else 0
    original_end = end if end and end > 0 else None

    # Apply start/end indices for resuming or partial runs
    if original_start > 0:
        print(f"Resuming from question index {original_start}...")
        questions = questions[original_start:]
    if original_end is not None:
        print(f"Limiting to question index {original_end} (exclusive)...")
        questions = questions[: max(0, original_end - original_start)]

    # Build output subdirectory from start/end to keep runs separate
    if start or end:
        subdir_parts = []
        if original_start > 0:
            subdir_parts.append(f"start_{original_start}")
        if original_end is not None:
            subdir_parts.append(f"end_{original_end}")
        output_dir = os.path.join(output_dir, "_".join(subdir_parts))

    print(f"\nTotal questions: {len(questions)}")
    print(f"Distributing across {num_gpus} GPUs...")
    print(f"Output directory: {output_dir}")

    # Split questions evenly across GPUs
    chunk_size = (len(questions) + num_gpus - 1) // num_gpus
    question_chunks = []
    chunk_start_indices = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(questions))
        if start_idx < len(questions):
            question_chunks.append(questions[start_idx:end_idx])
            chunk_start_indices.append(start + start_idx if start else start_idx)
        else:
            question_chunks.append([])
            chunk_start_indices.append(0)

    for i, chunk in enumerate(question_chunks):
        print(f"  GPU {i}: {len(chunk)} questions (start index: {chunk_start_indices[i]})")

    # Create output directories for each GPU
    gpu_output_dirs = []
    for i in range(num_gpus):
        gpu_output_dir = os.path.join(output_dir, f"gpu_{i}")
        os.makedirs(gpu_output_dir, exist_ok=True)
        gpu_output_dirs.append(gpu_output_dir)

    # Run benchmarks in parallel (one process per GPU)
    print(f"\nStarting {num_gpus} parallel benchmark processes...")

    # Start processes
    processes = []
    for i in range(num_gpus):
        if question_chunks[i]:  # Only start if there are questions
            p = mp.Process(
                target=gpu_worker,
                args=(gpu_ids[i], benchmark_path, db_root, gpu_output_dirs[i],
                      question_chunks[i], chunk_start_indices[i],
                      tensor_parallel_size, vllm_url)
            )
            p.start()
            processes.append(p)

    # Wait for all to complete
    for p in processes:
        p.join()

    print("\nAll GPU processes completed!")

    # Merge results from all GPUs
    print("\nMerging results from all GPUs...")
    all_results = []

    for i, gpu_output_dir in enumerate(gpu_output_dirs):
        results_file = os.path.join(gpu_output_dir, "benchmark_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                gpu_results = json.load(f)
                all_results.extend(gpu_results)
                print(f"  GPU {i}: {len(gpu_results)} results")

    # Save merged results
    merged_output_file = os.path.join(output_dir, "benchmark_results_merged.json")
    with open(merged_output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate detailed report
    generate_detailed_report(all_results, output_dir)

    print(f"\nMerged results saved to: {merged_output_file}")
    print(f"Speedup: ~{num_gpus}x faster than single GPU")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCS-SQL Benchmark with vLLM")
    parser.add_argument("--benchmark", required=True, help="Path to mini_dev_sqlite.json")
    parser.add_argument("--db_root", required=True, help="Path to databases directory")
    parser.add_argument("--output", default="outputs/benchmark_vllm_120b")
    parser.add_argument("--tensor-parallel-size", type=int, default=4,
                        help="Number of GPUs for tensor parallelism (default: 4)")
    parser.add_argument("--vllm-url", type=str, default=None,
                        help="URL to running vLLM API server (optional)")
    parser.add_argument("--start", type=int, default=0, help="Start index for resuming")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs (data parallelism)")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--gpu-id", type=int, default=None, help="Specific GPU ID for single-GPU mode")

    args = parser.parse_args()

    if args.vllm_url:
        print(f"\nUsing vLLM API mode: {args.vllm_url}")
        print("Local vLLM import is not required for generation.")
    else:
        print("\nUsing local vLLM mode (tensor parallel on this machine).")

    if args.multi_gpu:
        run_multi_gpu_benchmark(
            args.benchmark,
            args.db_root,
            args.output,
            args.start,
            args.end,
            args.num_gpus,
            args.tensor_parallel_size,
            args.vllm_url,
        )
    else:
        # Single GPU mode with start/end support
        run_vllm_single_gpu(
            args.benchmark,
            args.db_root,
            args.output,
            args.start,
            args.end,
            args.gpu_id,
            args.tensor_parallel_size,
            args.vllm_url,
        )
