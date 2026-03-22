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
    
    # Load existing results if they exist
    results_file = os.path.join(output_dir, "benchmark_results.json")
    results_detail = []
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results_detail = json.load(f)
        print(f"Loaded {len(results_detail)} existing results from {results_file}")
    
    # Initialize error logger
    error_logger = ErrorLogger(output_dir)
    
    # Initialize vLLM model
    if vllm_model is None:
        if vllm_url:
            print(f"Connecting to vLLM API server at {vllm_url}...")
            vllm_client = vLLMAPIClient(base_url=vllm_url)
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
        
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        print(f"\n[{global_q_idx+1}/{len(questions)}] Q: {question[:80]}...")
        
        if not os.path.exists(db_path):
            print(f"  Warning: DB not found at {db_path}")
            continue
        
        # 1. Schema Linking
        print("  Running Schema Linking...")
        t0 = time.time()
        full_schema = linker.load_schema(db_path)
        
        try:
            linking_res = linker.link_schema(full_schema, question, evidence)
        except Exception as e:
            print(f"  Schema linking error: {e}")
            linking_res = type('obj', (object,), {'tables': [], 'columns': []})
        
        # Filter tables and columns
        all_db_tables = set(full_schema.keys())
        linking_res.tables = [t for t in linking_res.tables if t in all_db_tables]
        
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
        
        # Generate all responses using vLLM's continuous batching
        print(f"  Running vLLM batch generation ({len(all_prompts)} prompts)...")
        
        if hasattr(vllm_client, 'generate_batch'):
            # vLLMModelManager
            all_responses = vllm_client.generate_batch(all_prompts, batch_size=32, show_progress=True)
        else:
            # vLLMAPIClient - generate one by one
            all_responses = []
            for prompt in all_prompts:
                response = vllm_client.generate(prompt)
                all_responses.append(response)
        
        del all_prompts
        
        # Parse responses
        print("  Parsing responses...")
        for i, response in enumerate(all_responses):
            p_name, gen_idx = prompt_metadata[i]
            sql_query = extract_sql_from_response(response)
            
            if sql_query:
                generated_candidates.append({
                    "sql": sql_query,
                    "prompt_type": p_name,
                    "gen_idx": gen_idx
                })
        
        del all_responses
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
                "failure_info": {
                    "phase": "EXECUTION",
                    "reason": "No queries executed successfully",
                    "error_count": execution_errors
                }
            }
            results_detail.append(failure_result)
            difficulty_results[difficulty].append(False)
            
            with open(results_file, "w") as f:
                json.dump(results_detail, f, indent=2)
            
            del generated_candidates
            del all_executions
            del sql_to_result_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        # Group by result
        result_groups = {}
        for exec_item in all_executions:
            res_set = exec_item["result_str"]
            if res_set not in result_groups:
                result_groups[res_set] = []
            result_groups[res_set].append({
                "sql": exec_item["sql"],
                "exec_time": exec_item["exec_time"]
            })
        
        group_normalizers = {
            res_set: min(item["exec_time"] for item in items)
            for res_set, items in result_groups.items()
        }
        
        result_counts = Counter(item["result_str"] for item in all_executions)
        most_common_result, top_count = result_counts.most_common(1)[0]
        
        winning_sqls = list(set(
            item["sql"] for item in all_executions 
            if item["result_str"] == most_common_result
        ))
        representative_sql = min(winning_sqls, key=len)
        
        gt_success, gt_res_set, gt_time = execute_sql_with_timeout(db_path, ground_truth)
        winner_confidence = top_count / N_valid
        
        # High confidence candidates
        high_conf_sqls = []
        processed_results = set()
        for res_set, count in result_counts.items():
            conf = count / N_valid
            if conf > 0.2 and res_set not in processed_results:
                processed_results.add(res_set)
                group_sqls = list(set(
                    item["sql"] for item in all_executions 
                    if item["result_str"] == res_set
                ))
                rep = min(group_sqls, key=len)
                high_conf_sqls.append({
                    "sql": rep,
                    "confidence": conf,
                    "count": count,
                    "best_exec_time": group_normalizers[res_set],
                })
        
        high_conf_sqls.sort(key=lambda x: x["confidence"], reverse=True)
        
        # SQL Selection Phase
        high_conf_candidates = [c for c in high_conf_sqls if c['confidence'] > 0.2]
        selected_sql = None
        selection_reasoning = None
        
        if len(high_conf_candidates) == 1:
            selected_sql = high_conf_candidates[0]['sql']
            selection_reasoning = "Single candidate"
        elif len(high_conf_candidates) > 1:
            print("  Running SQL Selection Phase...")
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
            
            # Generate multiple selection responses
            n_selection_samples = config.MAJORITY_VOTE_N
            selection_prompts = [selection_prompt] * n_selection_samples
            
            if hasattr(vllm_client, 'generate_batch'):
                selection_responses = vllm_client.generate_batch(selection_prompts, batch_size=16)
            else:
                selection_responses = [vllm_client.generate(p) for p in selection_prompts]
            
            # Parse and majority vote
            selection_votes = []
            for response in selection_responses:
                sql = extract_sql_from_response(response)
                if sql:
                    selection_votes.append(sql)
            
            if selection_votes:
                sql_counts = Counter(selection_votes)
                most_common_sql, vote_count = sql_counts.most_common(1)[0]
                selected_sql = most_common_sql
                print(f"    Majority vote: {selected_sql[:100]}... ({vote_count}/{len(selection_votes)} votes)")
        
        # Evaluate correctness
        if selected_sql:
            selected_success, selected_res_set, _ = execute_sql_with_timeout(db_path, selected_sql)
            is_correct = (gt_success and selected_success and selected_res_set == gt_res_set)
        else:
            majority_success, majority_res_set, _ = execute_sql_with_timeout(db_path, representative_sql)
            is_correct = (gt_success and majority_success and majority_res_set == gt_res_set)
        
        print(f"    Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        difficulty_results[difficulty].append(is_correct)
        execution_times = [item["exec_time"] for item in all_executions]
        
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
                "selected_sql": selected_sql,
                "reasoning": selection_reasoning,
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
        
        # Cleanup
        del generated_candidates, all_executions, sql_to_result_cache
        del result_groups, group_normalizers, execution_confidences
        del winning_sqls, result_counts, high_conf_sqls, high_conf_candidates
        del selection_prompts, selection_responses, selection_votes
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Generate detailed report
    generate_detailed_report(results_detail, output_dir)
    
    # Cleanup vLLM model
    del vllm_client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_detailed_report(results: List[Dict], output_dir: str):
    """Generate comprehensive benchmark report."""
    import statistics
    
    print("\n" + "="*100)
    print(" " * 30 + "DETAILED BENCHMARK REPORT")
    print("="*100)
    
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    overall_acc = (correct / total * 100) if total > 0 else 0
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total Questions:     {total}")
    print(f"  Correct:             {correct} ({overall_acc:.2f}%)")
    print(f"  Incorrect:           {total - correct} ({100 - overall_acc:.2f}%)")
    
    # Save report
    report_data = {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": overall_acc
        }
    }
    
    report_file = os.path.join(output_dir, "detailed_report.json")
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Report saved to: {report_file}")
    print("="*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCS-SQL Benchmark with vLLM")
    parser.add_argument("--benchmark", required=True, help="Path to mini_dev_sqlite.json")
    parser.add_argument("--db_root", required=True, help="Path to databases directory")
    parser.add_argument("--output", default="outputs/benchmark_vllm")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, 
                        help="Number of GPUs for tensor parallelism (default: 4)")
    parser.add_argument("--vllm-url", type=str, default=None,
                        help="URL to running vLLM API server (optional)")
    parser.add_argument("--start", type=int, default=0, help="Start index for resuming")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    
    args = parser.parse_args()
    
    if not VLLM_AVAILABLE and not args.vllm_url:
        print("\nError: vLLM is not installed and no API URL provided.")
        print("Install with: pip install vllm")
        print("\nOr start a vLLM server:")
        print("  python -m vllm.entrypoints.api_server \\")
        print("      --model openai/gpt-oss-120b \\")
        print("      --tensor-parallel-size 4 \\")
        print("      --port 8000")
        exit(1)
    
    run_vllm_benchmark(
        benchmark_path=args.benchmark,
        db_root=args.db_root,
        output_dir=args.output,
        tensor_parallel_size=args.tensor_parallel_size,
        vllm_url=args.vllm_url,
        start_index=args.start,
        end_index=args.end,
        limit=args.limit,
    )
