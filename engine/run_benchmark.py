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
from schema_linking import SchemaLinker, TransformersLLMClient, MultiModelManager
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
        # Escape curly braces so they don't interfere with .format()
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


def get_sample_table_contents(db_path: str, tables: List[str], sample_size: int = 3) -> str:
    """
    Get sample contents from each table in CSV format.
    
    Args:
        db_path: Path to SQLite database
        tables: List of table names to sample
        sample_size: Number of rows to sample from each table
        
    Returns:
        Formatted string with sample table contents
    """
    parts = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for table in tables:
            try:
                # Get sample rows
                cursor.execute(f"SELECT * FROM {table} LIMIT {sample_size}")
                rows = cursor.fetchall()
                
                # Get column names
                column_names = [desc[0] for desc in cursor.description]
                
                # Format as CSV-like table
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


def run_benchmark(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    limit: int = None
):
    os.makedirs(output_dir, exist_ok=True)
    
    config = Config()

    # Load 2 model copies for parallel batch generation on A100
    # (2 copies fit in 80 GB VRAM with room for overhead)
    print("Loading Multi-Model Manager (2 copies for parallel generation)...")
    multi_model = MultiModelManager(
        model_name=config.LLM_MODEL_NAME,
        device=config.LLM_DEVICE,
        max_new_tokens=512,
        temperature=0.3,  # Balance between diversity and speed
        num_copies=2,
    )
    
    # Use first model for schema linker (single-threaded)
    llm_client = multi_model.models[0]

    # Setup schema linker with 20 iterations for majority voting
    linker = SchemaLinker(
        pt=config.TABLE_LINKING_ITERATIONS,
        pc=config.COLUMN_LINKING_ITERATIONS,
        n=20,  # 20 parallel outputs per iteration for robust schema linking
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

    # Load prompt templates
    with open(config.PROMPTS_DIR / "SQL_generation.txt", "r") as f:
        prompt_template = f.read()
    
    with open(config.PROMPTS_DIR / "SQL_selection.txt", "r") as f:
        selection_template = f.read()

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
        print(f"  Standard examples: {len(std_examples)}, Masked examples: {len(msk_examples)}")
        print(f"  std_examples[0]: {std_examples[0] if std_examples else 'None'}")
        print(f"  msk_examples[0]: {msk_examples[0] if msk_examples else 'None'}")
        
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
        
        print(f"  Built {len(prompt_variations)} prompt variations")
        for pname, pex in prompt_variations:
            print(f"    {pname}: {len(pex)} examples")
            
        # 4. Generate 100 Queries (5 prompts × 20 generations) using parallel batch generation
        print("  Generating SQL candidates (5 × 20 = 100 using parallel batch)...")
        generated_candidates = []

        # Get sample table contents for the linked schema
        sample_contents = get_sample_table_contents(db_path, list(linking_res.tables), sample_size=3)
        print(f"  Sample table contents:\n{sample_contents[:500]}...")

        # Build all 100 prompts first (5 prompt types × 20 generations each)
        all_prompts = []
        prompt_metadata = []  # Track (prompt_type, gen_idx) for each prompt
        
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
            
            # Create 20 copies of this prompt (each will sample independently)
            for gen_idx in range(20):
                all_prompts.append(base_prompt)
                prompt_metadata.append((p_name, gen_idx))
        
        print(f"  Built {len(all_prompts)} prompts for parallel generation...")
        
        # Generate all 100 responses in parallel using 2 model copies with batch size 8
        print("  Running parallel batch generation across 2 models (batch_size=8)...")
        all_responses = multi_model.generate_parallel(all_prompts, stop_sequences=None, batch_size=8)
        
        # Parse responses and extract SQL
        print("  Parsing responses...")
        for i, response in enumerate(all_responses):
            p_name, gen_idx = prompt_metadata[i]
            
            try:
                print(f"    Gen {i+1}/100 - Response length: {len(response)}")
                
                # Extract SQL from JSON - ignore everything that's not JSON
                sql_query = ""

                # Strip markdown code fences
                response_stripped = response.strip()
                if response_stripped.startswith("```json"):
                    response_stripped = response_stripped[7:]
                elif response_stripped.startswith("```"):
                    response_stripped = response_stripped[3:]

                # Find the first { and extract only the JSON object
                start_idx = response_stripped.find("{")
                if start_idx != -1:
                    # Find matching closing brace by counting braces, ignoring content in strings
                    brace_count = 0
                    end_idx = -1
                    in_string = False
                    escape_next = False

                    for j, char in enumerate(response_stripped[start_idx:], start_idx):
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
                                    end_idx = j + 1
                                    break

                    if end_idx > start_idx:
                        json_str = response_stripped[start_idx:end_idx]
                        # Remove trailing code fence if present
                        if json_str.rstrip().endswith("```"):
                            json_str = json_str.rstrip()[:-3]

                        try:
                            parsed = json.loads(json_str)
                            sql_query = parsed.get("sql", "")
                            print(f"      Parsed SQL: {sql_query[:100] if sql_query else 'None'}...")
                        except json.JSONDecodeError as je:
                            print(f"      JSON parse error: {je}")

                if not sql_query:
                    # Method 2: Regex fallback for SQL
                    import re
                    match = re.search(r'SELECT.*?(?:;|$)', response, re.IGNORECASE | re.DOTALL)
                    if match:
                        sql_query = match.group(0).strip()
                        print(f"      Regex extracted SQL: {sql_query[:100]}...")

                if sql_query:
                    generated_candidates.append({
                        "sql": sql_query,
                        "prompt_type": p_name,
                        "gen_idx": gen_idx
                    })
                else:
                    print(f"      No SQL extracted!")
            except Exception as e:
                print(f"    Parse error: {e}")
                    
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

        # SQL Selection Phase: Use LLM to select the best SQL from all high-confidence candidates
        # Following the paper: present candidates as multiple-choice, sample n responses, majority vote
        print("\n  Running SQL Selection Phase...")
        # Use ALL candidates that pass the confidence threshold (> 0.2), not just top 3
        high_conf_candidates = [c for c in high_conf_sqls if c['confidence'] > 0.2]
        
        selected_sql = None
        selection_reasoning = None
        
        if len(high_conf_candidates) > 0:
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
            
            print(f"    Selection prompt: {selection_prompt[:300]}...")
            print(f"    Candidates (confidence > 0.2): {len(selection_candidates)}")

            # Sample n=20 responses from LLM for majority voting using parallel generation
            n_selection_samples = 20
            selection_votes = []

            print(f"    Generating {n_selection_samples} selection responses in parallel...")
            
            # Create 20 copies of the selection prompt
            selection_prompts = [selection_prompt] * n_selection_samples
            
            # Generate all 20 responses in parallel (batch_size=8)
            selection_responses = multi_model.generate_parallel(selection_prompts, stop_sequences=None, batch_size=8)
            
            # Parse all responses
            print(f"    Parsing {len(selection_responses)} selection responses...")
            for sel_idx, selection_response in enumerate(selection_responses):
                try:
                    sql = None
                    reasoning = None

                    # Method 1: Try to parse JSON
                    start_idx = selection_response.find("{")
                    if start_idx != -1:
                        brace_count = 0
                        end_idx = -1
                        in_string = False
                        escape_next = False

                        for i, char in enumerate(selection_response[start_idx:], start_idx):
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
                            json_str = selection_response[start_idx:end_idx]
                            # Remove trailing code fence if present
                            if json_str.rstrip().endswith("```"):
                                json_str = json_str.rstrip()[:-3]

                            try:
                                parsed = json.loads(json_str)
                                sql = parsed.get("sql", "")
                                reasoning = parsed.get("reasoning", "")
                            except json.JSONDecodeError as je:
                                # Method 2: Regex fallback to extract SQL from broken JSON
                                import re
                                # Try to find "sql": "..." pattern, handling multiline
                                sql_match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', selection_response, re.DOTALL)
                                if sql_match:
                                    sql = sql_match.group(1)
                                    # Unescape JSON string
                                    sql = sql.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                                reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', selection_response, re.DOTALL)
                                if reasoning_match:
                                    reasoning = reasoning_match.group(1).replace('\\"', '"')

                                if sql:
                                    print(f"      Sample {sel_idx+1}: Extracted via regex")
                                else:
                                    print(f"      Sample {sel_idx+1}: Could not extract SQL")
                        else:
                            print(f"      Sample {sel_idx+1}: No matching braces")
                    else:
                        print(f"      Sample {sel_idx+1}: No JSON found")

                    if sql:
                        selection_votes.append({"sql": sql, "reasoning": reasoning or ""})
                        print(f"      Sample {sel_idx+1}: {sql[:80]}...")
                except Exception as e:
                    print(f"      Sample {sel_idx+1} error: {e}")
            
            # Majority voting on selection
            if selection_votes:
                sql_counts = Counter(vote["sql"] for vote in selection_votes)
                most_common_sql, vote_count = sql_counts.most_common(1)[0]
                
                # Get reasoning from the vote that selected this SQL
                selected_reasoning = next(
                    (vote["reasoning"] for vote in selection_votes if vote["sql"] == most_common_sql),
                    ""
                )
                
                selected_sql = most_common_sql
                selection_reasoning = selected_reasoning
                
                print(f"    Majority vote: {selected_sql[:150]}... ({vote_count}/{len(selection_votes)} votes)")
                representative_sql = selected_sql
            else:
                print("    No valid selection responses, keeping majority vote result")
        else:
            print("    No high-confidence candidates (confidence > 0.2) for selection")

        # Re-evaluate correctness with the selected SQL
        if selected_sql:
            selected_success, selected_res, _ = execute_sql_with_timeout(db_path, selected_sql)
            is_correct = (gt_success and selected_res == gt_res)
            print(f"    Selected SQL correctness: {'CORRECT' if is_correct else 'INCORRECT'}")

        results_detail.append({
            "question_id": q.get("question_id", q_idx),
            "question": question,
            "db_id": db_id,
            "ground_truth": ground_truth,
            "winner_sql": representative_sql,
            "is_correct": is_correct,
            "winner_confidence": winner_confidence,
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
