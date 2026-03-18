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
from error_logger import ErrorLogger, MemoryManager, check_gpu_memory, clear_gpu_memory

# Multi-GPU setup
def setup_multi_gpu(num_gpus: int = 4):
    """Setup multi-GPU environment and return list of GPU IDs."""
    import torch
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


def load_benchmark(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def execute_sql_with_timeout(db_path: str, sql: str, timeout: int = 5) -> Tuple[bool, str, float]:
    """Execute SQL and return (success, result_string_or_error, execution_time)."""
    start_time = time.time()
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        exec_time = time.time() - start_time
        # Convert to set of tuples for comparison (order-independent, like official BIRD EX)
        res_set = frozenset(results)
        return True, res_set, exec_time
    except Exception as e:
        exec_time = time.time() - start_time
        return False, frozenset(), exec_time


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
    limit: int = None,
    gpu_id: int = None,
    questions_chunk: List[Dict] = None
):
    """
    Run benchmark on a single GPU or all GPUs.
    
    Args:
        benchmark_path: Path to benchmark JSON file
        db_root: Path to database root directory
        output_dir: Output directory for results
        limit: Optional limit on number of questions
        gpu_id: Specific GPU ID to use (None for all)
        questions_chunk: Subset of questions to process (for multi-GPU)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize error logger and memory manager
    error_logger = ErrorLogger(output_dir)
    memory_manager = MemoryManager(threshold_gb=5.0, errors_dir=output_dir)

    # Set GPU device if specified
    if gpu_id is not None:
        import torch
        torch.cuda.set_device(gpu_id)
        print(f"Running on GPU {gpu_id}")

    config = Config()
    
    # Determine number of model copies based on model size and GPU
    # For 20B model: 1 copy per GPU (uses ~40-45 GB VRAM)
    # For 7B model: 2 copies per GPU (uses ~28 GB VRAM)
    model_name = config.LLM_MODEL_NAME.lower()

    # Auto-detect model size from model name
    if "20b" in model_name or "32b" in model_name or "coder" in model_name or "gpt-oss" in model_name:
        num_copies = 1  # Larger model (20B+), 1 copy per GPU
        print("Detected large model (20B+ or gpt-oss-20b), using 1 copy per GPU")
    elif "14b" in model_name or "13b" in model_name:
        num_copies = 1  # Medium model (13-14B), 1 copy per GPU
        print("Detected medium model (13-14B), using 1 copy per GPU")
    else:
        num_copies = 1  # Force 1 copy to avoid OOM (can increase to 2 if VRAM allows)
        print("Using 1 model copy per GPU (safe mode)")

    # Load model copies for parallel batch generation
    print(f"Loading Multi-Model Manager ({num_copies} copies for parallel generation) on GPU {gpu_id if gpu_id is not None else 'all'}...")
    multi_model = MultiModelManager(
        model_name=config.LLM_MODEL_NAME,
        device=config.LLM_DEVICE,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,  # Use config value
        temperature=config.LLM_TEMPERATURE,  # Use config temperature
        num_copies=num_copies,
        gpu_id=gpu_id,  # Pass GPU ID for multi-GPU support
    )

    # Use first model for schema linker (single-threaded)
    # But pass multi_model for batch generation capability
    llm_client = multi_model.models[0]

    # Setup schema linker with 20 iterations for majority voting
    # Pass the multi_model so it can use generate_parallel
    linker = SchemaLinker(
        pt=config.TABLE_LINKING_ITERATIONS,
        pc=config.COLUMN_LINKING_ITERATIONS,
        n=20,  # 20 parallel outputs per iteration for robust schema linking
        llm_client=multi_model,  # Use multi_model for batch generation
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

    # Load questions (or use provided chunk for multi-GPU)
    if questions_chunk is not None:
        questions = questions_chunk
        print(f"Processing chunk of {len(questions)} questions on GPU {gpu_id}")
    else:
        questions = load_benchmark(benchmark_path)
        if limit:
            questions = questions[:limit]

    print(f"Loaded {len(questions)} questions")

    results_detail = []
    difficulty_results = {
        "simple": [],
        "moderate": [],
        "challenging": [],
        "unknown": []
    }
    
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
                    "question_id": q_idx,
                    "db_id": db_id,
                })
                # Continue with empty linking result
                linking_res = type('obj', (object,), {'tables': [], 'columns': []})
            else:
                raise
        
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

        # Generate all 100 responses in parallel using 2 model copies with batch size 4
        print("  Running parallel batch generation across 2 models (batch_size=4)...")
        
        # Check GPU memory before generation
        is_low, free_gb, allocated_gb = check_gpu_memory(threshold_gb=10.0)
        if is_low:
            print(f"  [WARNING] GPU memory low before generation: {free_gb:.2f}GB free")
            clear_gpu_memory(verbose=True)
        
        # Try generation with error handling
        all_responses = []
        generation_error = None
        
        try:
            all_responses = multi_model.generate_parallel(all_prompts, stop_sequences=None, batch_size=4)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                generation_error = e
                print(f"\n[CUDA OOM] Generation failed, attempting recovery...")
                
                # Log the error
                error_logger.log_cuda_error(e, {
                    "phase": "SQL_GENERATION",
                    "question_id": q_idx,
                    "batch_size": 6,
                    "num_prompts": len(all_prompts),
                    "free_gb": free_gb,
                    "allocated_gb": allocated_gb,
                })
                
                # Try with smaller batch size and CPU offloading
                print("  Retrying with batch_size=2 and CPU offloading...")
                try:
                    # Clear memory first
                    clear_gpu_memory(verbose=True)
                    
                    # Offload model weights temporarily (if possible)
                    import torch
                    model_on_cpu = {}
                    for idx, model_wrapper in enumerate(multi_model.models):
                        if hasattr(model_wrapper, 'model'):
                            model_on_cpu[idx] = {
                                'weights': model_wrapper.model.state_dict(),
                                'device': next(model_wrapper.model.parameters()).device,
                            }
                            model_wrapper.model = model_wrapper.model.cpu()
                    
                    torch.cuda.empty_cache()
                    
                    # Retry with smaller batch
                    all_responses = multi_model.generate_parallel(all_prompts, stop_sequences=None, batch_size=2)
                    
                    # Restore model to GPU
                    for idx, model_wrapper in enumerate(multi_model.models):
                        if idx in model_on_cpu and hasattr(model_wrapper, 'model'):
                            model_wrapper.model.to(model_on_cpu[idx]['device'])
                    
                    print("  Recovery successful!")
                    
                except Exception as recovery_error:
                    print(f"  Recovery failed: {recovery_error}")
                    error_logger.log_cuda_error(recovery_error, {
                        "phase": "SQL_GENERATION_RECOVERY",
                        "question_id": q_idx,
                    })
                    all_responses = [""] * len(all_prompts)  # Empty responses
            else:
                raise  # Re-raise non-OOM errors
        
        # Parse responses and extract SQL
        print("  Parsing responses...")
        for i, response in enumerate(all_responses):
            p_name, gen_idx = prompt_metadata[i]
            
            try:
                print(f"    Gen {i+1}/100 - Response length: {len(response)}")
                
                # Extract SQL from JSON - use robust 3-tier parsing
                sql_query = ""
                reasoning = ""

                # Method 1: Try to parse JSON
                response_stripped = response.strip()
                
                # Remove markdown code fences if present
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
                            reasoning = parsed.get("reasoning", "")
                            print(f"      Parsed JSON: {sql_query[:100] if sql_query else 'None'}...")
                        except json.JSONDecodeError as je:
                            print(f"      JSON decode error: {je}")
                            # Method 2: Regex fallback to extract SQL from broken JSON
                            import re
                            sql_match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', response_stripped, re.DOTALL)
                            if sql_match:
                                sql_query = sql_match.group(1)
                                # Unescape JSON string
                                sql_query = sql_query.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                            reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', response_stripped, re.DOTALL)
                            if reasoning_match:
                                reasoning = reasoning_match.group(1).replace('\\"', '"')
                            if sql_query:
                                print(f"      Extracted via regex: {sql_query[:100]}...")
                    else:
                        print(f"      No matching braces found")
                else:
                    print(f"      No opening brace found")

                # Method 3: Last resort - try to extract SQL directly (without JSON)
                if not sql_query:
                    import re
                    # Look for SELECT statement
                    sql_match = re.search(r'(SELECT\s+.*?)(?:\s*[,}\n]|$)', response, re.IGNORECASE | re.DOTALL)
                    if sql_match:
                        sql_query = sql_match.group(1).strip()
                        print(f"      Extracted SQL directly: {sql_query[:100]}...")

                if sql_query:
                    generated_candidates.append({
                        "sql": sql_query,
                        "prompt_type": p_name,
                        "gen_idx": gen_idx
                    })
                    print(f"      SUCCESS: {sql_query[:80]}...")
                else:
                    print(f"      FAILED - No SQL extracted from response")
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
            # Still save results with failure explanation
            results_detail.append({
                "question_id": q.get("question_id", q_idx),
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
                "failure_reason": "No valid SQL queries were generated or all generated queries failed execution (syntax errors/timeouts). This could be due to: (1) LLM outputting malformed JSON, (2) LLM not following output format, (3) Generated SQL having syntax errors, or (4) Schema linking failed to identify correct tables/columns.",
                "generated_candidates_sample": [cand["sql"][:200] for cand in generated_candidates[:5]] if generated_candidates else []
            })
            
            # Save intermediate results even on failure
            with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
                json.dump(results_detail, f, indent=2)
            
            # Track as incorrect for difficulty stats
            difficulty_results[difficulty].append(False)
            
            # Continue to next question with memory cleanup
            import gc
            import torch
            
            if 'generated_candidates' in locals(): del generated_candidates
            if 'all_executions' in locals(): del all_executions
            if 'sql_to_result_cache' in locals(): del sql_to_result_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            continue

        # Group executions by result_set and track best (minimum) execution time per group
        result_groups = {}  # result_frozenset -> list of {sql, exec_time}
        for exec_item in all_executions:
            res_set = exec_item["result_str"]  # This is now a frozenset
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
        winning_sqls = list(set(item["sql"] for item in all_executions if item["result_str"] == most_common_result))

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
                group_sqls = list(set(item["sql"] for item in all_executions if item["result_str"] == res_set))
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
        print("\n  Running SQL Selection Phase...")
        # Use ALL candidates that pass the confidence threshold (> 0.2), not just top 3
        high_conf_candidates = [c for c in high_conf_sqls if c['confidence'] > 0.2]
        
        selected_sql = None
        selection_reasoning = None
        is_correct = False  # Will be set after selection
        
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

            # Generate all 20 responses in parallel with OOM handling
            selection_responses = []
            
            try:
                selection_responses = multi_model.generate_parallel(selection_prompts, stop_sequences=None, batch_size=4)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"\n[CUDA OOM] Selection failed, attempting recovery...")
                    
                    error_logger.log_cuda_error(e, {
                        "phase": "SQL_SELECTION",
                        "question_id": q_idx,
                        "batch_size": 6,
                        "num_prompts": len(selection_prompts),
                    })
                    
                    # Retry with smaller batch
                    try:
                        clear_gpu_memory(verbose=True)
                        selection_responses = multi_model.generate_parallel(selection_prompts, stop_sequences=None, batch_size=2)
                        print("  Selection recovery successful!")
                    except Exception as recovery_error:
                        error_logger.log_cuda_error(recovery_error, {
                            "phase": "SQL_SELECTION_RECOVERY",
                            "question_id": q_idx,
                        })
                        selection_responses = [""] * len(selection_prompts)
                else:
                    raise
            
            # Parse all responses
            print(f"    Parsing {len(selection_responses)} selection responses...")
            for sel_idx, selection_response in enumerate(selection_responses):
                try:
                    sql = None
                    reasoning = None

                    # Method 1: Try to parse JSON
                    response = selection_response.strip()
                    
                    # Remove markdown code fences if present
                    if response.startswith("```json"):
                        response = response[7:]
                    elif response.startswith("```"):
                        response = response[3:]
                    
                    # Find the JSON object - extract only the JSON, ignore everything else
                    start_idx = response.find("{")
                    if start_idx != -1:
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
                            # Remove trailing code fence if present
                            if json_str.rstrip().endswith("```"):
                                json_str = json_str.rstrip()[:-3]

                            try:
                                parsed = json.loads(json_str)
                                sql = parsed.get("sql", "")
                                reasoning = parsed.get("reasoning", "")
                                print(f"      Sample {sel_idx+1}: Parsed JSON successfully")
                            except json.JSONDecodeError as je:
                                print(f"      Sample {sel_idx+1}: JSON decode error: {je}")
                                # Method 2: Regex fallback to extract SQL from broken JSON
                                import re
                                # Try to find "sql": "..." pattern, handling multiline
                                sql_match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
                                if sql_match:
                                    sql = sql_match.group(1)
                                    # Unescape JSON string
                                    sql = sql.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                                reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL)
                                if reasoning_match:
                                    reasoning = reasoning_match.group(1).replace('\\"', '"')

                                if sql:
                                    print(f"      Sample {sel_idx+1}: Extracted via regex")
                                else:
                                    print(f"      Sample {sel_idx+1}: Could not extract SQL")
                        else:
                            print(f"      Sample {sel_idx+1}: No matching braces found")
                    else:
                        print(f"      Sample {sel_idx+1}: No JSON found (no opening brace)")

                    # Method 3: Last resort - try to extract SQL directly (without JSON)
                    if not sql:
                        import re
                        # Look for SELECT statement
                        sql_match = re.search(r'(SELECT\s+.*?)(?:\s*[,}\n]|$)', response, re.IGNORECASE | re.DOTALL)
                        if sql_match:
                            sql = sql_match.group(1).strip()
                            print(f"      Sample {sel_idx+1}: Extracted SQL directly (no JSON)")

                    if sql:
                        selection_votes.append({"sql": sql, "reasoning": reasoning or ""})
                        print(f"      Sample {sel_idx+1}: {sql[:80]}...")
                    else:
                        print(f"      Sample {sel_idx+1}: FAILED - No SQL extracted")
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
            "question_id": q.get("question_id", q_idx),
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

        # Save intermediate
        with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
            json.dump(results_detail, f, indent=2)

        # ========== MEMORY CLEANUP TO PREVENT VRAM LEAKS ==========
        # Clear GPU memory after each question to prevent OOM
        import gc
        import torch

        # Delete large intermediate variables that are no longer needed
        # Note: Don't delete schema_text, question, evidence - used in selection phase
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
        if 'selection_responses' in locals(): del selection_responses
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
            print(f"  [Memory Cleanup] GPU {gpu_id}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            
            # Check if memory is still high and log warning
            if allocated > 60.0:  # More than 60GB allocated
                print(f"  [WARNING] High memory usage detected!")
                error_logger.log_cuda_error(
                    Exception("High memory usage after cleanup"),
                    {
                        "phase": "POST_QUESTION_CLEANUP",
                        "question_id": q_idx,
                        "allocated_gb": allocated,
                        "reserved_gb": reserved,
                    }
                )
        # ===========================================================

    # Final cleanup of persistent resources
    print("\nPerforming final cleanup of persistent resources...")
    import gc
    import torch
    
    # Delete primary controllers
    if 'multi_model' in locals(): del multi_model
    if 'linker' in locals(): del linker
    if 'standard_indexer' in locals(): del standard_indexer
    if 'masked_indexer' in locals(): del masked_indexer
    if 'literal_masker' in locals(): del literal_masker
    
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
        print(f"\n⏱️  EXECUTION TIME STATISTICS (All SQL Queries)")
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
    db_stats = {}
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
    
    # Per-Database Difficulty Breakdown
    print(f"\n🗄️  PER-DATABASE BREAKDOWN BY DIFFICULTY")
    print("-"*100)
    
    for db_id, db_qs in db_results.items():
        db_name = db_id.split('/')[-1] if '/' in db_id else db_id
        print(f"\n  📁 {db_name} ({len(db_qs)} questions)")
        print("  " + "-"*70)
        print(f"    {'Difficulty':<15} {'Correct':>10} {'Total':>8} {'Accuracy':>10} {'Avg Exec Time':>15}")
        print("    " + "-"*70)
        
        for diff in ["simple", "moderate", "challenging", "unknown"]:
            diff_qs = [r for r in db_qs if r.get("difficulty") == diff]
            if diff_qs:
                diff_correct = sum(1 for r in diff_qs if r.get("is_correct", False))
                diff_total = len(diff_qs)
                diff_acc = (diff_correct / diff_total * 100) if diff_total > 0 else 0
                
                # Average execution time for this difficulty
                diff_exec_times = []
                for r in diff_qs:
                    if "execution_times" in r:
                        diff_exec_times.extend(r["execution_times"])
                avg_exec = statistics.mean(diff_exec_times)*1000 if diff_exec_times else 0
                
                print(f"    {diff.capitalize():<15} {diff_correct:>10} {diff_total:>8} {diff_acc:>9.2f}% {avg_exec:>12.2f} ms")
    
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
# Multi-GPU Worker Function (must be at module level for pickling)
# =============================================================================

def gpu_worker(gpu_id, benchmark_path, db_root, output_dir, questions_chunk):
    """
    Worker function to run benchmark on a specific GPU.
    Must be at module level (not nested) for multiprocessing pickling.
    """
    import os
    import gc
    import torch
    
    # Set CUDA visible device BEFORE any torch operations
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # Run benchmark with gpu_id=0 since CUDA_VISIBLE_DEVICES makes it see only one GPU
    run_benchmark(
        benchmark_path=benchmark_path,
        db_root=db_root,
        output_dir=output_dir,
        limit=None,  # Already chunked
        gpu_id=0,  # In worker, we see only 1 GPU (set by CUDA_VISIBLE_DEVICES)
        questions_chunk=questions_chunk
    )


def run_multi_gpu_benchmark(
    benchmark_path: str,
    db_root: str,
    output_dir: str,
    limit: int = None,
    num_gpus: int = 4
):
    """
    Run benchmark across multiple GPUs using data parallelism.
    Each GPU processes a different subset of questions independently.
    
    Args:
        benchmark_path: Path to benchmark JSON file
        db_root: Path to database root directory
        output_dir: Output directory for results
        limit: Optional limit on number of questions
        num_gpus: Number of GPUs to use
    """
    import torch
    import multiprocessing as mp
    
    # CRITICAL: Use 'spawn' method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    # Setup GPUs
    gpu_ids = setup_multi_gpu(num_gpus)
    num_gpus = len(gpu_ids)
    
    # Load all questions
    questions = load_benchmark(benchmark_path)
    if limit:
        questions = questions[:limit]
    
    print(f"\nTotal questions: {len(questions)}")
    print(f"Distributing across {num_gpus} GPUs...")
    
    # Split questions evenly across GPUs
    chunk_size = (len(questions) + num_gpus - 1) // num_gpus
    question_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(questions))
        if start_idx < len(questions):
            question_chunks.append(questions[start_idx:end_idx])
        else:
            question_chunks.append([])
    
    for i, chunk in enumerate(question_chunks):
        print(f"  GPU {i}: {len(chunk)} questions")
    
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
                args=(gpu_ids[i], benchmark_path, db_root, gpu_output_dirs[i], question_chunks[i])
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
    all_difficulty_results = {
        "simple": [],
        "moderate": [],
        "challenging": [],
        "unknown": []
    }
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Path to mini_dev_sqlite.json")
    parser.add_argument("--db_root", required=True, help="Path to databases dir")
    parser.add_argument("--output", default="outputs/benchmark_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use")

    args = parser.parse_args()
    
    if args.multi_gpu:
        run_multi_gpu_benchmark(
            args.benchmark,
            args.db_root,
            args.output,
            args.limit,
            args.num_gpus
        )
    else:
        run_benchmark(args.benchmark, args.db_root, args.output, args.limit)
