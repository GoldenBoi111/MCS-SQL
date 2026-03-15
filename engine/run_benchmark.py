"""
Benchmark script for MCS-SQL pipeline.

This script combines schema linking and multiple SQL generation (5 prompts * 20 queries = 100 queries)
for each question in the BIRD MiniDev benchmark. It then executes all candidates, filters out
those with syntax errors/timeouts, and uses execution-based majority voting to find the best SQL.

Confidence is calculated using: confidence(qi) = 1/N * Σ(exec(qi) = exec(qj))
where N is the number of valid executions (excluding timeouts and syntax errors).
Queries are grouped by execution result, with best execution speed as the normalizer.
Returns queries with confidence > 0.2.
"""

import argparse
import json
import logging
import os
import random
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple

from config import Config
from literal_masker import LiteralMasker
from schema_linking import SchemaLinker, TransformersLLMClient
from training_dataset_indexer import TrainingDatasetIndexer
from training_dataset_indexer_masked import MaskedTrainingDatasetIndexer


logger = logging.getLogger(__name__)


def load_benchmark(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def execute_sql_with_timeout(db_path: str, sql: str, timeout: int = 5) -> Tuple[bool, str, float]:
    """Execute SQL and return (success, result_string_or_error, execution_time)."""
    start_time = time.time()
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        # BIRD evaluation often requires limiting results or executing within strict time
        # Here we just fetch all to get the result set for majority voting
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()

        # Convert results to a canonical string for comparison (handling sorting if needed,
        # though strict BIRD eval might not sort. We sort to group equivalent unordered sets)
        res_str = str(sorted([str(row) for row in results]))
        conn.close()
        exec_time = time.time() - start_time
        return True, res_str, exec_time
    except Exception as e:
        exec_time = time.time() - start_time
        return False, str(e), exec_time


def build_examples_text(examples: List[Dict[str, Any]]) -> str:
    """Format the retrieved examples for the prompt."""
    parts = ["<examples>"]
    for ex in examples:
        q_text = ex.get('orig_question', ex.get('question', ''))
        sql_text = ex.get('orig_sql', ex.get('sql', ''))
        parts.append(f"# Question: {q_text}")
        parts.append(f"# Gold SQL: {sql_text}")
        parts.append("")
    
    parts.append("</examples>")
    return "\n".join(parts)


def run_benchmark(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    limit: int = None
):
    os.makedirs(output_dir, exist_ok=True)
    
    config = Config()
    
    # Need transformers for schema linking & generation
    print("Loading LLM Client...")
    llm_client = TransformersLLMClient(
        model_name=config.LLM_MODEL_NAME,
        device=config.LLM_DEVICE,
        max_new_tokens=1024,
        temperature=0.7
    )
    
    # Setup schema linker
    linker = SchemaLinker(
        pt=config.TABLE_LINKING_ITERATIONS,
        pc=config.COLUMN_LINKING_ITERATIONS,
        n=1, # For benchmark, we might just do 1 pass for speed, or set to MAJORITY_VOTE_N
        llm_client=llm_client
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
    
    # Also need literal masker to mask queries prior to searching masked index
    literal_masker = LiteralMasker(llm_client=llm_client)
    
    # Load prompt template
    with open(config.PROMPTS_DIR / "SQL_generation.txt", "r") as f:
        prompt_template = f.read()
        
    questions = load_benchmark(benchmark_path)
    if limit:
        questions = questions[:limit]
        
    print(f"Loaded {len(questions)} questions")
    
    results_detail = []
    
    for q_idx, q in enumerate(questions):
        db_id = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "")
        ground_truth = q["SQL"]
        difficulty = q.get("difficulty", "unknown")
        
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        print(f"\n[{q_idx+1}/{len(questions)}] Q: {question[:80]}...")
        
        if not os.path.exists(db_path):
            print(f"  Warning: DB not found at {db_path}")
            continue
            
        # 1. Schema Linking
        print("  Running Schema Linking...")
        t0 = time.time()
        full_schema = linker.load_schema(db_path)
        linking_res = linker.link_schema(full_schema, question, evidence)
        
        # Format the linked schema for the generator prompt
        linked_schema_dict = {t: full_schema[t] for t in linking_res.tables if t in full_schema}
        # Filter down columns as well if your linking_res.columns specifies them
        # (For simplicity here, we inject all columns of selected tables, or you can filter exactly)
        schema_text = linker.format_schema_for_prompt(linked_schema_dict)
        print(f"  Schema Linking took {time.time() - t0:.2f}s (Found {len(linking_res.tables)} tables)")
        
        # 2. Retrieve Examples (k=20)
        print("  Retrieving examples from FAISS...")
        k = 20
        standard_results = standard_indexer.search(question, top_k=k)
        
        masked_q = literal_masker.mask_question(question)
        masked_results = masked_indexer.search(masked_q, top_k=k)
        
        # Format examples into dictionaries
        std_examples = [
            {"question": sq, "sql": sql, "metadata": meta}
            for (sq, sql, meta, score) in standard_results
        ]
        
        msk_examples = [
            {"question": mq, "sql": msql, "orig_question": oq, "orig_sql": osql, "metadata": meta}
            for (mq, oq, msql, osql, meta, score) in masked_results
        ]
        
        # 3. Build 5 Prompt Variations
        # 1 MASKED only
        prompt_variations = []
        prompt_variations.append(("masked_only", msk_examples[:10])) # 10 examples per prompt 
        
        # 1 STANDARD only
        prompt_variations.append(("standard_only", std_examples[:10]))
        
        # 3 MIXED
        for i in range(3):
            mixed = std_examples[:5] + msk_examples[:5]
            # Fill rest with random from remainder
            rem_std = std_examples[5:]
            rem_msk = msk_examples[5:]
            rem_pool = rem_std + rem_msk
            if rem_pool:
                mixed += random.sample(rem_pool, min(10 - len(mixed), len(rem_pool)))
            random.shuffle(mixed)
            prompt_variations.append((f"mixed_{i}", mixed))
            
        # 4. Generate 100 Queries (5 prompts * 20 generations)
        print("  Generating SQL candidates (5 x 20)...")
        generated_candidates = []
        
        for p_name, ex_list in prompt_variations:
            ex_text = build_examples_text(ex_list)
            
            prompt = prompt_template.format(
                examples=ex_text,
                schema_text=schema_text,
                question=question,
                evidence=evidence
            )
            
            for gen_idx in range(20):
                try:
                    response = llm_client.generate(prompt)
                    # Extract SQL from JSON
                    start_idx = response.find("{")
                    end_idx = response.rfind("}") + 1
                    sql_query = ""
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        parsed = json.loads(json_str)
                        sql_query = parsed.get("sql", "")
                    
                    if not sql_query:
                        # Regex fallback
                        import re
                        match = re.search(r'SELECT.*?(?:;|$)', response, re.IGNORECASE | re.DOTALL)
                        if match:
                            sql_query = match.group(0).strip()
                            
                    if sql_query:
                        generated_candidates.append({
                            "sql": sql_query,
                            "prompt_type": p_name,
                            "gen_idx": gen_idx
                        })
                except Exception as e:
                    print(f"    Generation error: {e}")
                    
        print(f"  Generated {len(generated_candidates)} valid SQL candidates")
        
        # 5. Execute and perform Majority Voting
        # Store all execution results with timing for each generated candidate
        # Structure: list of {sql, result_str, exec_time, success, prompt_type, gen_idx}
        all_executions = []
        execution_errors = 0
        sql_to_result_cache = {}  # Cache to avoid re-executing same SQL

        print("  Executing candidates...")
        for cand in generated_candidates:
            sql = cand["sql"]
            
            # Check cache first
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
                    "prompt_type": cand["prompt_type"],
                    "gen_idx": cand["gen_idx"]
                })
            else:
                execution_errors += 1

        N_valid = len(all_executions)
        print(f"  Valid executing queries: {N_valid} (Errors: {execution_errors})")

        if N_valid == 0:
            print("  No queries executed successfully.")
            results_detail.append({
                "question_id": q.get("question_id", q_idx),
                "question": question,
                "winner_sql": "SELECT 1",
                "correct": False,
                "confidence": 0.0
            })
            continue

        # Group executions by result_str and track best (minimum) execution time per group
        result_groups = {}  # result_str -> list of {sql, exec_time}
        for exec_item in all_executions:
            res_str = exec_item["result_str"]
            if res_str not in result_groups:
                result_groups[res_str] = []
            result_groups[res_str].append({
                "sql": exec_item["sql"],
                "exec_time": exec_item["exec_time"]
            })

        # Find best (minimum) execution time per group - this is the normalizer
        group_normalizers = {}  # result_str -> best_exec_time
        for res_str, items in result_groups.items():
            group_normalizers[res_str] = min(item["exec_time"] for item in items)

        # Calculate confidence for each execution using the formula:
        # confidence(qi) = 1/N * sum from j=1 to N of (exec(qi) = exec(qj))
        # where N is the number of valid executions (excluding timeouts and syntax errors)
        # This simplifies to: confidence = count(result_i) / N
        execution_confidences = []  # list of {sql, result_str, confidence, exec_time}
        for exec_item in all_executions:
            res_str = exec_item["result_str"]
            count_same_result = len(result_groups[res_str])
            confidence = count_same_result / N_valid
            execution_confidences.append({
                "sql": exec_item["sql"],
                "result_str": res_str,
                "confidence": confidence,
                "exec_time": exec_item["exec_time"],
                "normalized_by": group_normalizers[res_str]
            })

        # Find the result group with highest confidence (most common result)
        result_counts = Counter(item["result_str"] for item in all_executions)
        most_common_result, top_count = result_counts.most_common(1)[0]

        # Find all unique SQLs that produced the winning result
        winning_sqls = list(set(item["sql"] for item in all_executions if item["result_str"] == most_common_result))

        # Pick the shortest winning SQL as the representative
        representative_sql = min(winning_sqls, key=len)

        # Execution evaluation against ground truth
        gt_success, gt_res, gt_time = execute_sql_with_timeout(db_path, ground_truth)

        is_correct = (gt_success and most_common_result == gt_res)
        winner_confidence = top_count / N_valid

        print(f"  Confidence: {winner_confidence:.2f} ({top_count}/{N_valid}) -> {'CORRECT' if is_correct else 'INCORRECT'}")

        # Collect queries with confidence > 0.2, grouped by result with best speed as normalizer
        high_conf_sqls = []
        processed_results = set()
        for res_str, count in result_counts.items():
            conf = count / N_valid
            if conf > 0.2 and res_str not in processed_results:
                processed_results.add(res_str)
                # Find all unique SQLs for this result group
                group_sqls = list(set(item["sql"] for item in all_executions if item["result_str"] == res_str))
                # Pick representative (shortest SQL)
                rep = min(group_sqls, key=len)
                high_conf_sqls.append({
                    "sql": rep,
                    "confidence": conf,
                    "count": count,
                    "best_exec_time": group_normalizers[res_str],
                    "all_sqls_in_group": group_sqls
                })
        
        # Sort by confidence descending
        high_conf_sqls.sort(key=lambda x: x["confidence"], reverse=True)
        
        print("\n  Top 5 High-Confidence Queries:")
        for i, q_res in enumerate(high_conf_sqls[:5], 1):
            print(f"    {i}. [Conf: {q_res['confidence']:.2f}, Time: {q_res['best_exec_time']:.3f}s] {q_res['sql'][:150]}...")
            
        results_detail.append({
            "question_id": q.get("question_id", q_idx),
            "question": question,
            "db_id": db_id,
            "ground_truth": ground_truth,
            "winner_sql": representative_sql,
            "is_correct": is_correct,
            "winner_confidence": winner_confidence,
            "high_confidence_alternatives": high_conf_sqls,
            "metrics": {
                "generated": len(generated_candidates),
                "execution_errors": execution_errors,
                "valid_generations": N_valid,
                "unique_valid_sqls": len(set(item["sql"] for item in all_executions))
            }
        })
        
        # Save intermediate
        with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
            json.dump(results_detail, f, indent=2)

    print("\nDone! Results saved to", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Path to mini_dev_sqlite.json")
    parser.add_argument("--db_root", required=True, help="Path to databases dir")
    parser.add_argument("--output", default="outputs/benchmark_results")
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    run_benchmark(args.benchmark, args.db_root, args.output, args.limit)
