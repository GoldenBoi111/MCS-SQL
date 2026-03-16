"""
Schema Linking Module for Text-to-SQL

This module implements a two-stage schema linking approach:
1. Table Linking - Select relevant tables from the database schema
2. Column Linking - Select relevant columns from the chosen tables

Both stages use LLM calls with shuffled prompts to improve robustness through
majority voting.
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SchemaLinkingResult:
    """Result of schema linking process."""

    tables: List[str]
    columns: List[str]
    reasoning: str


class TransformersLLMClient:
    """
    LLM client using Hugging Face Transformers.
    Supports both Qwen models and GPT-OSS 20B with true batch generation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        gpu_id: int = None,
    ):
        """
        Initialize the LLM client.

        Args:
            model_name: Hugging Face model name
            device: Device to run model on ('cuda' or 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            gpu_id: Specific GPU ID to use (None for auto)
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Always use standard model loading for true batch generation
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gpu_id = gpu_id

        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model with appropriate dtype
        # Use expandable_segments to avoid memory fragmentation
        import os
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

        model_kwargs = {
            "trust_remote_code": True,
        }

        # Set dtype based on model and device
        if device == "cuda":
            # Use bfloat16 for A100 GPUs (better for gpt-oss-20b)
            # Use float16 for older GPUs or Qwen models
            if "gpt-oss" in model_name.lower() or "20b" in model_name.lower():
                model_kwargs["torch_dtype"] = torch.bfloat16
                print("  Using bfloat16 for GPT-OSS 20B")
            else:
                model_kwargs["torch_dtype"] = torch.float16
                print("  Using float16")

        # Set specific GPU if provided
        if gpu_id is not None:
            model_kwargs["device_map"] = f"cuda:{gpu_id}"
            print(f"  Loading on GPU {gpu_id}")
        else:
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print(f"Model loaded successfully on {device}")

    def generate(self, prompt: str, stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate response from the model.
        Uses chat template for gpt-oss-20b, direct prompt for Qwen.

        Args:
            prompt: Input prompt string
            stop_sequences: Optional list of sequences to stop generation at

        Returns:
            Generated response string
        """
        import torch

        # Use chat template for gpt-oss-20b
        if "gpt-oss" in self.model_name.lower():
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = prompt

        # Tokenize
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if hasattr(self, 'device') and self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Prepare stop sequences for transformers
        stopping_criteria = None
        if stop_sequences:
            from transformers import StoppingCriteriaList, StoppingCriteria

            class StopOnSequence(StoppingCriteria):
                def __init__(self, sequences, tokenizer):
                    self.sequences = sequences
                    self.tokenizer = tokenizer

                def __call__(self, input_ids, scores, **kwargs):
                    decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    return any(seq in decoded for seq in self.sequences)

            stopping_criteria = StoppingCriteriaList([
                StopOnSequence(stop_sequences, self.tokenizer)
            ])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        # Decode only the generated tokens (not the prompt)
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response.strip()

    def generate_batch(self, prompts: List[str], stop_sequences: Optional[List[str]] = None) -> List[str]:
        """
        Generate responses for multiple prompts in batch (faster on A100).
        Uses chat template for gpt-oss-20b, direct prompts for Qwen.
        TRUE BATCH: All prompts processed in single model.forward() call.

        Args:
            prompts: List of input prompt strings
            stop_sequences: Optional list of sequences to stop generation at

        Returns:
            List of generated response strings
        """
        import torch
        from transformers import StoppingCriteriaList, StoppingCriteria

        if not prompts:
            return []

        # Apply chat template for gpt-oss-20b if needed
        if "gpt-oss" in self.model_name.lower():
            processed_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                processed_prompts.append(prompt_text)
        else:
            processed_prompts = prompts

        # Tokenize all prompts with padding - TRUE BATCH
        inputs = self.tokenizer(processed_prompts, return_tensors="pt", padding=True)
        if hasattr(self, 'device') and self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Prepare stop sequences
        stopping_criteria = None
        if stop_sequences:
            class StopOnSequence(StoppingCriteria):
                def __init__(self, sequences, tokenizer):
                    self.sequences = sequences
                    self.tokenizer = tokenizer

                def __call__(self, input_ids, scores, **kwargs):
                    decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    return any(seq in decoded for seq in self.sequences)

            stopping_criteria = StoppingCriteriaList([
                StopOnSequence(stop_sequences, self.tokenizer)
            ])

        # Batch generation with KV cache enabled - ALL PROMPTS IN ONE FORWARD PASS
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        # Decode each generated sequence
        input_lengths = [len(inp) for inp in inputs["input_ids"]]
        responses = []
        for i in range(len(prompts)):
            input_len = input_lengths[i]
            gen_ids = outputs[i][input_len:]
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(response.strip())

        return responses


class MultiModelManager:
    """
    Manages multiple model copies for parallel batch generation on A100.
    Distributes prompts across model copies for maximum throughput.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str,
        max_new_tokens: int,
        temperature: float,
        num_copies: int = 4,
        gpu_id: int = None,
    ):
        """
        Initialize multiple model copies.

        Args:
            model_name: Hugging Face model name
            device: Device ('cuda')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_copies: Number of model copies to load
            gpu_id: Specific GPU ID to use (None for auto)
        """
        import torch

        self.num_copies = num_copies
        self.device = device
        self.gpu_id = gpu_id
        self.models: List[TransformersLLMClient] = []

        print(f"Loading {num_copies} model copies for parallel generation...")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  GPU ID: {gpu_id if gpu_id is not None else 'auto'}")
        print(f"  Max tokens: {max_new_tokens}")
        print(f"  Temperature: {temperature}")

        for i in range(num_copies):
            print(f"  Loading model copy {i+1}/{num_copies}...")
            model = TransformersLLMClient(
                model_name=model_name,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                gpu_id=gpu_id,
            )
            self.models.append(model)

            # Clear cache after each model load to prevent fragmentation
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

        print(f"All {num_copies} model copies loaded successfully")
        if gpu_id is not None:
            print(f"  Estimated VRAM usage on GPU {gpu_id}: ~{num_copies * 14} GB")
        else:
            print(f"  Estimated VRAM usage: ~{num_copies * 14} GB")
    
    def generate_parallel(self, prompts: List[str], stop_sequences: Optional[List[str]] = None, batch_size: int = 8) -> List[str]:
        """
        Generate responses by distributing prompts across all model copies.
        Each model processes prompts in smaller batches to avoid OOM.

        Args:
            prompts: List of prompts to process
            stop_sequences: Optional stop sequences
            batch_size: Max prompts per batch per model (default 8)

        Returns:
            List of generated responses (same order as input prompts)
        """
        if not prompts:
            return []

        n_prompts = len(prompts)
        n_models = len(self.models)

        # Distribute prompts across models (round-robin for load balancing)
        model_prompts: List[List[tuple]] = [[] for _ in range(n_models)]
        for i, prompt in enumerate(prompts):
            model_idx = i % n_models
            model_prompts[model_idx].append((i, prompt))

        # Generate on each model in parallel
        import threading
        results: Dict[int, str] = {}
        errors: List[Exception] = []

        def worker(model_idx: int):
            try:
                model = self.models[model_idx]
                indices_prompts = model_prompts[model_idx]

                if indices_prompts:
                    # Process in smaller batches to avoid OOM
                    for batch_start in range(0, len(indices_prompts), batch_size):
                        batch_end = min(batch_start + batch_size, len(indices_prompts))
                        batch = indices_prompts[batch_start:batch_end]
                        
                        batch_prompts = [p for _, p in batch]
                        batch_results = model.generate_batch(batch_prompts, stop_sequences)

                        for (idx, _), result in zip(batch, batch_results):
                            results[idx] = result
            except Exception as e:
                errors.append(e)

        # Start all workers
        threads = []
        for i in range(n_models):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # Check for errors
        if errors:
            print(f"Warning: {len(errors)} model errors during parallel generation")
            for e in errors[:3]:  # Show first 3 errors
                print(f"  Error: {e}")

        # Reconstruct results in original order
        responses = [results.get(i, "") for i in range(n_prompts)]
        return responses


class SchemaLinker:
    """
    Schema linker that uses LLM calls to extract relevant schema elements.

    Attributes:
        pt: Number of times to shuffle and call LLM for table linking
        pc: Number of times to shuffle and call LLM for column linking
        n: Number of outputs to generate for majority voting
        llm_client: Client for making LLM API calls
        _current_schema: Current schema for mock LLM responses
    """

    def __init__(
        self, pt: int = 3, pc: int = 3, n: int = 20, llm_client: Optional[Any] = None
    ):
        """
        Initialize the schema linker.

        Args:
            pt: Number of shuffle iterations for table linking
            pc: Number of shuffle iterations for column linking
            n: Number of parallel outputs for majority voting
            llm_client: LLM client instance (must have a `generate` method)
        """
        self.pt = pt
        self.pc = pc
        self.n = n
        self.llm_client = llm_client
        self._current_schema: Optional[Dict[str, List[str]]] = None

    def load_schema(self, db_path: str) -> Dict[str, List[str]]:
        """
        Load database schema from SQLite database.

        Args:
            db_path: Path to the SQLite database file

        Returns:
            Dictionary mapping table names to lists of column names
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )

        tables = [row[0] for row in cursor.fetchall()]
        print(tables)
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            schema[table] = columns

        conn.close()
        return schema

    def format_schema_for_prompt(
        self,
        schema: Dict[str, List[str]],
        tables_to_include: Optional[List[str]] = None,
    ) -> str:
        """
        Format schema as text for LLM prompt.

        Args:
            schema: Full database schema
            tables_to_include: Optional list of tables to include (for column linking)

        Returns:
            Formatted schema string
        """
        lines = []
        tables = tables_to_include if tables_to_include else schema.keys()

        for table in tables:
            if table in schema:
                columns = ", ".join(schema[table])
                lines.append(f"Table: {table}\nColumns: {columns}")

        return "\n\n".join(lines)

    def shuffle_schema_order(
        self,
        schema: Dict[str, List[str]],
        tables_to_include: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Randomly shuffle the order of tables in schema.

        Args:
            schema: Full database schema
            tables_to_include: Optional list of tables to include

        Returns:
            Shuffled schema dictionary
        """
        tables = list(tables_to_include) if tables_to_include else list(schema.keys())
        random.shuffle(tables)

        shuffled = {}
        for table in tables:
            if table in schema:
                shuffled[table] = schema[table]

        return shuffled

    def build_table_linking_prompt(
        self, schema_text: str, question: str, evidence: str = ""
    ) -> str:
        """
        Build prompt for table linking task.

        Args:
            schema_text: Formatted database schema
            question: User's natural language question
            evidence: Optional knowledge evidence

        Returns:
            Complete prompt string for table linking
        """
        prompt = f"""### Given a database schema, question, and knowledge evidence, extract a list of
tables that should be referenced to convert the question into SQL.
### SQLite SQL tables, with their properties:

{schema_text}

### Question:
{question}

### Knowledge Evidence:
{evidence if evidence else "None provided"}

You need to not only select the required tables, but also explain in detail why each
table is needed.
Your answer should strictly follow the following json format.
{{
    "reasoning": "", // The reason for choosing each table.
    "tables": [], // List of selected tables.
}}

### Your Answer:"""
        return prompt

    def build_column_linking_prompt(
        self,
        schema_text: str,
        question: str,
        selected_tables: List[str],
        evidence: str = "",
    ) -> str:
        """
        Build prompt for column linking task.

        Args:
            schema_text: Formatted database schema (only selected tables)
            question: User's natural language question
            selected_tables: List of tables selected from table linking
            evidence: Optional knowledge evidence

        Returns:
            Complete prompt string for column linking
        """
        tables_str = ", ".join(selected_tables)
        prompt = f"""### Given a database schema, question, and knowledge evidence, extract a list of
columns that should be referenced to convert the question into SQL.
### SQLite SQL tables, with their properties:

{schema_text}

### Selected Tables:
{tables_str}

### Question:
{question}

### Knowledge Evidence:
{evidence if evidence else "None provided"}

You need to not only select the required columns, but also explain in detail why
each column is needed.
Your answer should strictly follow the following json format.
{{
    "reasoning": "", // The reason for choosing each column.
    "columns": ["table_name_i.column_name_j", ...], // List of selected columns
}}

### Your Answer:"""
        return prompt

    def parse_llm_response(self, response: str, task_type: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON result.

        Args:
            response: Raw LLM response string
            task_type: Either 'table' or 'column'

        Returns:
            Parsed dictionary with reasoning and tables/columns
        """
        try:
            # Remove markdown code fences if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            
            # Find the JSON object - extract only the JSON, ignore everything else
            start_idx = response.find("{")
            if start_idx == -1:
                return {"reasoning": "", "tables" if task_type == "table" else "columns": []}
            
            # Find matching closing brace by counting braces, ignoring content in strings
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
                
                print(f"  [DEBUG] JSON attempt ({len(json_str)} chars): {json_str[:200]}...")
                result = json.loads(json_str)

                if task_type == "table":
                    return {
                        "reasoning": result.get("reasoning", ""),
                        "tables": result.get("tables", []),
                    }
                else:
                    return {
                        "reasoning": result.get("reasoning", ""),
                        "columns": result.get("columns", []),
                    }
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [DEBUG] Full response: {response[:500]}...")
            print(f"Error parsing LLM response: {e}")

        return {"reasoning": "", "tables" if task_type == "table" else "columns": []}

    def union_results(
        self, results: List[Dict[str, Any]], task_type: str
    ) -> Tuple[List[str], str]:
        """
        Union all items from multiple LLM outputs without duplicates.
        Collects all unique tables/columns from all LLM calls.

        Args:
            results: List of parsed LLM results
            task_type: Either 'table' or 'column'

        Returns:
            Tuple of (all unique items, combined reasoning)
        """
        key = "tables" if task_type == "table" else "columns"

        # Collect all unique items (union without duplicates)
        unique_items: List[str] = []
        seen: set = set()
        all_reasoning = []

        for result in results:
            items = result.get(key, [])
            reasoning = result.get("reasoning", "")
            # Only add non-empty, non-duplicate reasoning
            if reasoning and reasoning not in all_reasoning:
                all_reasoning.append(reasoning)

            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique_items.append(item)

        combined_reasoning = "\n\n".join(all_reasoning)

        return unique_items, combined_reasoning

    def link_tables(
        self, schema: Dict[str, List[str]], question: str, evidence: str = ""
    ) -> Tuple[List[str], str]:
        """
        Perform table linking using shuffled prompts and union of all results.
        Uses batch generation for speed when MultiModelManager is available.

        Args:
            schema: Database schema dictionary
            question: User's natural language question
            evidence: Optional knowledge evidence

        Returns:
            Tuple of (selected tables, reasoning)
        """
        results = []
        self._current_schema = schema

        # Collect all prompts first
        all_prompts = []
        prompt_configs = []  # Track (shuffle_idx, sample_idx) for each prompt
        
        for i in range(self.pt):
            # Shuffle schema order for diversity
            shuffled_schema = self.shuffle_schema_order(schema)
            schema_text = self.format_schema_for_prompt(shuffled_schema)

            # Build prompt
            prompt = self.build_table_linking_prompt(schema_text, question, evidence)
            
            # Generate n outputs
            for j in range(self.n):
                all_prompts.append(prompt)
                prompt_configs.append((i, j))

        # Check if we have batch generation capability
        print(f"    DEBUG: llm_client type = {type(self.llm_client).__name__}")
        print(f"    DEBUG: has generate_parallel = {hasattr(self.llm_client, 'generate_parallel')}")
        
        if hasattr(self.llm_client, 'generate_parallel'):
            # Use parallel batch generation (MultiModelManager)
            print(f"    Table linking: generating {len(all_prompts)} responses in parallel...")
            all_responses = self.llm_client.generate_parallel(all_prompts, stop_sequences=None, batch_size=8)
            
            # Parse all responses
            for idx, response in enumerate(all_responses):
                parsed = self.parse_llm_response(response, "table")
                results.append(parsed)
        else:
            # Sequential generation (fallback)
            print(f"    Table linking: generating {len(all_prompts)} responses sequentially...")
            for idx, prompt in enumerate(all_prompts):
                if self.llm_client:
                    response = self.llm_client.generate(prompt)
                else:
                    response = self._mock_llm_call(prompt, "table", schema)
                parsed = self.parse_llm_response(response, "table")
                results.append(parsed)

        # Union all results (no duplicates)
        tables, reasoning = self.union_results(results, "table")

        return tables, reasoning

    def link_columns(
        self,
        schema: Dict[str, List[str]],
        selected_tables: List[str],
        question: str,
        evidence: str = "",
    ) -> Tuple[List[str], str]:
        """
        Perform column linking using shuffled prompts and union of all results.
        Uses batch generation for speed when MultiModelManager is available.

        Args:
            schema: Database schema dictionary
            selected_tables: Tables selected from table linking
            question: User's natural language question
            evidence: Optional knowledge evidence

        Returns:
            Tuple of (selected columns, reasoning)
        """
        results = []

        # Collect all prompts first
        all_prompts = []
        
        for i in range(self.pc):
            # Shuffle table order for diversity
            shuffled_schema = self.shuffle_schema_order(schema, selected_tables)
            schema_text = self.format_schema_for_prompt(
                shuffled_schema, selected_tables
            )

            # Build prompt
            prompt = self.build_column_linking_prompt(
                schema_text, question, selected_tables, evidence
            )
            
            # Generate n outputs
            for j in range(self.n):
                all_prompts.append(prompt)

        # Check if we have batch generation capability
        if hasattr(self.llm_client, 'generate_parallel'):
            # Use parallel batch generation (MultiModelManager)
            print(f"    Column linking: generating {len(all_prompts)} responses in parallel...")
            all_responses = self.llm_client.generate_parallel(all_prompts, stop_sequences=None, batch_size=8)
            
            # Parse all responses
            for response in all_responses:
                parsed = self.parse_llm_response(response, "column")
                results.append(parsed)
        else:
            # Sequential generation (fallback)
            print(f"    Column linking: generating {len(all_prompts)} responses sequentially...")
            for prompt in all_prompts:
                if self.llm_client:
                    response = self.llm_client.generate(prompt)
                else:
                    response = self._mock_llm_call(prompt, "column", schema)
                parsed = self.parse_llm_response(response, "column")
                results.append(parsed)

        # Union all results (no duplicates)
        columns, reasoning = self.union_results(results, "column")

        return columns, reasoning

    def link_schema(
        self, schema: Dict[str, List[str]], question: str, evidence: str = ""
    ) -> SchemaLinkingResult:
        """
        Perform complete schema linking (tables + columns).

        Args:
            schema: Database schema dictionary
            question: User's natural language question
            evidence: Optional knowledge evidence

        Returns:
            SchemaLinkingResult with tables, columns, and reasoning
        """
        # Stage 1: Table linking
        tables, table_reasoning = self.link_tables(schema, question, evidence)

        # Stage 2: Column linking (only from selected tables)
        columns, column_reasoning = self.link_columns(
            schema, tables, question, evidence
        )

        # Calculate confidence based on agreement rate
        total_items = len(tables) + len(columns)

        return SchemaLinkingResult(
            tables=tables,
            columns=columns,
            reasoning=f"Table Selection:\n{table_reasoning}\n\nColumn Selection:\n{column_reasoning}",
        )

    def _mock_llm_call(
        self, prompt: str, task_type: str, schema: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Mock LLM call for testing purposes.
        Returns schema-aware mock responses based on the actual database schema.

        Args:
            prompt: Input prompt
            task_type: Either 'table' or 'column'
            schema: Optional schema dictionary for context-aware responses

        Returns:
            Mock response string
        """
        if schema:
            tables = list(schema.keys())
            # Build mock response based on actual schema
            if task_type == "table":
                # Select first 1-2 tables as mock response
                selected = tables[: min(2, len(tables))]
                return f"""{{
    "reasoning": "Based on the question and available tables ({', '.join(tables)}), the selected tables contain relevant data.",
    "tables": {json.dumps(selected)}
}}"""
            else:
                # Select columns from the first table
                if tables:
                    first_table = tables[0]
                    columns = schema.get(first_table, [])
                    selected_columns = [
                        f"{first_table}.{col}" for col in columns[: min(3, len(columns))]
                    ]
                    return f"""{{
    "reasoning": "Selected columns from {first_table} table for the query.",
    "columns": {json.dumps(selected_columns)}
}}"""

        # Fallback for when no schema is provided
        if task_type == "table":
            return """{
    "reasoning": "Based on the question, we need to analyze customer data and payment information.",
    "tables": ["customers", "payments"]
}"""
        else:
            return """{
    "reasoning": "We need specific columns to filter and aggregate the data.",
    "columns": ["customers.customer_id", "customers.currency", "payments.amount"]
}"""


# Example usage
if __name__ == "__main__":
    from config import Config
    
    # Load configuration
    config = Config()
    
    # Initialize Qwen LLM client using Transformers
    # Using Qwen2.5-7B-Instruct (closest publicly available to Qwen 3.5 8B)
    llm_client = TransformersLLMClient(
        model_name=config.LLM_MODEL_NAME,
        device=config.LLM_DEVICE,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=config.LLM_TEMPERATURE,
    )

    # Create schema linker with LLM client
    linker = SchemaLinker(
        pt=config.TABLE_LINKING_ITERATIONS,
        pc=config.COLUMN_LINKING_ITERATIONS,
        n=config.MAJORITY_VOTE_N,
        llm_client=llm_client,
    )

    # Load schema from database
    db_path = config.get_database_path()
    schema = linker.load_schema(db_path)

    # Use a question relevant to the california_schools database
    question = "What is the average SAT score of schools in Los Angeles county?"
    evidence = "Average SAT score is calculated by taking the mean of all SAT scores."

    # Perform schema linking
    result = linker.link_schema(schema, question, evidence)
    print(result)

    print("Selected Tables:", result.tables)
    print("Selected Columns:", result.columns)
    print("Reasoning:", result.reasoning)
