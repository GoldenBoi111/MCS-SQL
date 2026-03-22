"""
Schema Linking Module for Text-to-SQL

This module implements a two-stage schema linking approach:
1. Table Linking - Select relevant tables from the database schema
2. Column Linking - Select relevant columns from the chosen tables

Both stages use LLM calls with shuffled prompts to improve robustness through
majority voting.

Uses outlines library for forced JSON output to ensure structured responses.
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import outlines

    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    outlines = None

from json_schemas import (
    TABLE_LINKING_SCHEMA,
    COLUMN_LINKING_SCHEMA,
    SQL_GENERATION_SCHEMA,
    SQL_SELECTION_SCHEMA,
)


@dataclass
class SchemaLinkingResult:
    """Result of schema linking process."""

    tables: List[str]
    columns: List[str]
    reasoning: str


class TransformersLLMClient:
    """
    LLM client using Hugging Face Transformers with outlines for forced JSON output.
    Supports both Qwen models and GPT-OSS with model parallelism for 120B.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        gpu_id: int = None,
        use_model_parallel: bool = False,
        gpu_memory_gb: int = 75,
    ):
        """
        Initialize the LLM client.

        Args:
            model_name: Hugging Face model name
            device: Device to run model on ('cuda' or 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.3 for focused generation)
            gpu_id: Specific GPU ID to use (None for auto)
            use_model_parallel: If True, distribute model across all GPUs (for 120B)
            gpu_memory_gb: Max GPU memory to use per GPU (default 75GB for 120B)
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gpu_id = gpu_id
        self.use_model_parallel = use_model_parallel
        self.gpu_memory_gb = gpu_memory_gb
        self.outlines_model = None  # Lazy-initialised on first generate_json call
        self._outlines_failed = False  # Guard: stop retrying after first failure

        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        import os

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }

        from config import get_config

        cfg = get_config()
        use_4bit = getattr(cfg, "LLM_USE_4BIT", False)
        use_8bit = getattr(cfg, "LLM_USE_8BIT", False)

        if device == "cuda":
            if use_4bit:
                model_kwargs["load_in_4bit"] = True
                print("  Using 4-bit quantization (bitsandbytes)")
            elif use_8bit:
                model_kwargs["load_in_8bit"] = True
                print("  Using 8-bit quantization (bitsandbytes)")
            else:
                if (
                    "gpt-oss" in model_name.lower()
                    or "20b" in model_name.lower()
                    or "120b" in model_name.lower()
                ):
                    model_kwargs["torch_dtype"] = torch.bfloat16
                    print("  Using bfloat16")
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                    print("  Using float16")

        if use_model_parallel or "120b" in model_name.lower():
            n_gpus = torch.cuda.device_count()
            max_memory = {i: f"{gpu_memory_gb}GiB" for i in range(n_gpus)}
            max_memory["cpu"] = "50GiB"
            model_kwargs["max_memory"] = max_memory
            model_kwargs["device_map"] = "auto"
            print(f"  [120B Mode] Distributing across {n_gpus} GPU(s)")
            print(f"    Per-GPU memory cap: {gpu_memory_gb}GiB")
            print(f"    CPU overflow buffer: 50GiB")
            print(f"    Total available: ~{n_gpus * gpu_memory_gb + 50}GiB")
        else:
            if gpu_id is not None:
                model_kwargs["device_map"] = f"cuda:{gpu_id}"
                print(f"  Loading on GPU {gpu_id}")
            else:
                model_kwargs["device_map"] = "auto"

        if "gpt-oss" in model_name.lower():
            model_kwargs["attn_implementation"] = "eager"
            print("  Using eager attention for GPT-OSS")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print(f"Model loaded successfully on {device}")

    def _get_outlines_model(self):
        """
        Lazy-load outlines wrapper on first use.
        Uses outlines 1.2.x API: outlines.from_transformers(model, tokenizer).
        Sets self._outlines_failed = True on any error so we never retry.
        Zero extra VRAM — wraps the already-loaded HF model in place.
        """
        if (
            self.outlines_model is None
            and OUTLINES_AVAILABLE
            and not self._outlines_failed
        ):
            try:
                print("  [Outlines] Wrapping existing model (no extra VRAM)...")
                # outlines 1.2.x top-level factory function
                self.outlines_model = outlines.from_transformers(
                    self.model, self.tokenizer
                )
                print("  [Outlines] Model wrapped successfully")
            except Exception as e:
                print(f"  [Outlines] Wrap failed: {e}")
                self._outlines_failed = True
        return self.outlines_model

    def generate(self, prompt: str, stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate response from the model.

        Args:
            prompt: Input prompt string
            stop_sequences: Optional list of sequences to stop generation at

        Returns:
            Generated response string
        """
        import torch

        messages = [
            {"role": "system", "content": "You are an expert SQL developer."},
            {"role": "user", "content": prompt},
        ]

        if self.tokenizer.chat_template is not None:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = prompt

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        stopping_criteria = None
        if stop_sequences:
            from transformers import StoppingCriteriaList, StoppingCriteria

            class StopOnSequence(StoppingCriteria):
                def __init__(self, sequences, tokenizer):
                    self.sequences = sequences
                    self.tokenizer = tokenizer

                def __call__(self, input_ids, scores, **kwargs):
                    decoded = self.tokenizer.decode(
                        input_ids[0], skip_special_tokens=True
                    )
                    return any(seq in decoded for seq in self.sequences)

            stopping_criteria = StoppingCriteriaList(
                [StopOnSequence(stop_sequences, self.tokenizer)]
            )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )
        return response.strip()

    def generate_batch(
        self, prompts: List[str], stop_sequences: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batch (faster on A100).
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

        if "gpt-oss" in self.model_name.lower():
            processed_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                processed_prompts.append(prompt_text)
        else:
            processed_prompts = prompts

        # CRITICAL: padding_side='left' is required for correct batch generation.
        # Without it, padding tokens are added on the right, which corrupts the
        # auto-regressive generation and causes blank/truncated outputs.
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(processed_prompts, return_tensors="pt", padding=True)
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        stopping_criteria = None
        if stop_sequences:

            class StopOnSequence(StoppingCriteria):
                def __init__(self, sequences, tokenizer):
                    self.sequences = sequences
                    self.tokenizer = tokenizer

                def __call__(self, input_ids, scores, **kwargs):
                    decoded = self.tokenizer.decode(
                        input_ids[0], skip_special_tokens=True
                    )
                    return any(seq in decoded for seq in self.sequences)

            stopping_criteria = StoppingCriteriaList(
                [StopOnSequence(stop_sequences, self.tokenizer)]
            )

        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                from error_logger import ErrorLogger

                error_logger = ErrorLogger()
                error_logger.log_cuda_error(
                    e,
                    {
                        "phase": "MODEL_GENERATION",
                        "model_name": self.model_name,
                        "batch_size": len(prompts),
                    },
                )
                return [""] * len(prompts)
            else:
                raise

        # CRITICAL: Use the padded tensor shape per-row, not len() on tensor rows.
        padded_input_len = inputs["input_ids"].shape[1]
        responses = []
        for i in range(len(prompts)):
            gen_ids = outputs[i][padded_input_len:]
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(response.strip())

        del inputs
        del outputs
        del gen_ids
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return responses

    def generate_json(
        self, prompt: str, json_schema, temperature: Optional[float] = None
    ) -> dict:
        """
        Generate structured JSON output using outlines 1.2.x.

        json_schema must be a Pydantic BaseModel class (preferred) or a raw dict.
        outlines guarantees the output matches the schema at the logit level —
        it is structurally impossible for the model to emit invalid JSON.

        Args:
            prompt: Input prompt string
            json_schema: Pydantic BaseModel class or JSON schema dict
            temperature: Optional temperature override

        Returns:
            Generated response as a plain dict
        """
        import torch

        outlines_model = self._get_outlines_model()

        if outlines_model is not None and not self._outlines_failed:
            try:
                # outlines 1.2.x API:
                #   outlines.Generator(model, outlines.json_schema(schema_string))
                # json_schema() takes a JSON *string*, not a dict or Pydantic class.
                schema_str = (
                    json.dumps(json_schema.model_json_schema())
                    if hasattr(json_schema, "model_json_schema")
                    else json.dumps(json_schema)
                )
                generator = outlines.Generator(
                    outlines_model, outlines.json_schema(schema_str)
                )

                messages = [
                    {"role": "system", "content": "You are an expert SQL developer."},
                    {"role": "user", "content": prompt},
                ]
                prompt_text = (
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if self.tokenizer.chat_template
                    else prompt
                )

                temp = temperature if temperature is not None else self.temperature
                kwargs = {"max_tokens": self.max_new_tokens}
                if temp > 0:
                    kwargs["temperature"] = temp

                print("  [Outlines] Generating forced JSON output...")
                result = generator(prompt_text, **kwargs)
                print("  [Outlines] JSON generated successfully")

                # 1.2.x returns a JSON string — parse it.
                # If it already came back as a dict (future-proofing), pass through.
                if isinstance(result, dict):
                    return result
                if hasattr(result, "model_dump"):
                    return result.model_dump()
                return json.loads(result)

            except Exception as e:
                print(f"  [Outlines] Generation failed: {e}")
                self._outlines_failed = True
                print("  [Outlines] Switching to fallback for all remaining calls")

        # ── Fallback: standard HF generation with JSON prompting ─────────────
        print("  [Fallback] Standard generation with JSON parsing...")

        schema_hint = (
            json_schema.model_json_schema()
            if hasattr(json_schema, "model_json_schema")
            else json_schema
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert SQL developer. Output ONLY valid JSON. "
                    "No markdown, no explanation. ONLY the JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\nIMPORTANT: Output ONLY valid JSON matching this schema: "
                    f"{json.dumps(schema_hint)}"
                ),
            },
        ]

        prompt_text = (
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if self.tokenizer.chat_template
            else prompt
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        temp = temperature if temperature is not None else self.temperature
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temp if temp > 0 else None,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            input_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            print(f"  [Fallback] Generation error: {e}")
            return {}

        # Strip markdown fences
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3].strip()

        # Extract the first complete JSON object by brace-counting
        start = response.find("{")
        if start == -1:
            print("  [Fallback] No '{' found in response")
            return {}

        depth, in_str, escaped, end = 0, False, False, -1
        for i, ch in enumerate(response[start:], start):
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_str:
                escaped = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            print("  [Fallback] No matching closing brace found")
            return {}

        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError as e:
            print(f"  [Fallback] JSON parse error: {e}")
            return {}


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

            if torch.cuda.is_available():
                gpu = gpu_id if gpu_id is not None else 0
                alloc = torch.cuda.memory_allocated(gpu) / 1e9
                print(f"    After copy {i+1}: GPU memory = {alloc:.2f}GB")

            import gc

            gc.collect()
            torch.cuda.empty_cache()

        print(f"All {num_copies} model copies loaded successfully")
        if gpu_id is not None:
            print(f"  Estimated VRAM usage on GPU {gpu_id}: ~{num_copies * 40} GB")
        else:
            print(f"  Estimated VRAM usage: ~{num_copies * 40} GB")

    def generate_parallel(
        self,
        prompts: List[str],
        stop_sequences: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[str]:
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

        model_prompts: List[List[tuple]] = [[] for _ in range(n_models)]
        for i, prompt in enumerate(prompts):
            model_idx = i % n_models
            model_prompts[model_idx].append((i, prompt))

        import threading

        results: Dict[int, str] = {}
        errors: List[Exception] = []

        def worker(model_idx: int):
            try:
                model = self.models[model_idx]
                indices_prompts = model_prompts[model_idx]

                if indices_prompts:
                    for batch_start in range(0, len(indices_prompts), batch_size):
                        batch_end = min(batch_start + batch_size, len(indices_prompts))
                        batch = indices_prompts[batch_start:batch_end]

                        batch_prompts = [p for _, p in batch]
                        batch_results = model.generate_batch(
                            batch_prompts, stop_sequences
                        )

                        for (idx, _), result in zip(batch, batch_results):
                            results[idx] = result

                        import gc
                        import torch

                        del batch_prompts
                        del batch_results
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(n_models):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            print(f"Warning: {len(errors)} model errors during parallel generation")
            for e in errors[:3]:
                print(f"  Error: {e}")

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

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )

        tables = [row[0] for row in cursor.fetchall()]
        print(tables)
        schema = {}
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
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
        Format: # table_name ( col1, col2, col3 )

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
                columns = schema[table]
                col_str = ", ".join(columns)
                lines.append(f"# {table} ( {col_str} )")

        return "\n".join(lines)

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
        Build prompt for table linking task using external template.
        """
        import os
        from config import get_config

        config = get_config()
        prompt_path = os.path.join(config.PROMPTS_DIR, "table_linking.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template.format(
            schema_text=schema_text,
            question=question,
            evidence=evidence if evidence else "None provided",
        )

    def build_column_linking_prompt(
        self,
        schema_text: str,
        question: str,
        selected_tables: List[str],
        evidence: str = "",
    ) -> str:
        """
        Build prompt for column linking task using external template.
        """
        import os
        from config import get_config

        config = get_config()
        prompt_path = os.path.join(config.PROMPTS_DIR, "column_linking.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        tables_str = ", ".join(selected_tables)
        return template.format(
            schema_text=schema_text,
            question=question,
            selected_tables=tables_str,
            evidence=evidence if evidence else "None provided",
        )

    def parse_llm_response(self, response: str, task_type: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON result.
        Only used in the mock/fallback path — outlines path returns a dict directly.

        Args:
            response: Raw LLM response string
            task_type: Either 'table' or 'column'

        Returns:
            Parsed dictionary with reasoning and tables/columns
        """
        print(f"\n  ============== RAW LLM OUTPUT ==============")
        print(response)
        print(f"  ============== END RAW OUTPUT ==============\n")

        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]

            start_idx = response.find("{")
            if start_idx == -1:
                print("  [DEBUG] No '{' found in response!")
            else:
                brace_count = 0
                end_idx = -1
                in_string = False
                escape_next = False

                for i, char in enumerate(response[start_idx:], start_idx):
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
                                end_idx = i + 1
                                break

                if end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    if json_str.rstrip().endswith("```"):
                        json_str = json_str.rstrip()[:-3]

                    print(
                        f"  [DEBUG] JSON attempt ({len(json_str)} chars): {json_str[:300]}..."
                    )
                    result = json.loads(json_str)
                    print(f"  [DEBUG] Parsed successfully! Keys: {list(result.keys())}")

                    if task_type == "table":
                        tables = result.get("tables", [])
                        print(f"  [DEBUG] Tables extracted: {tables}")
                        return {
                            "reasoning": result.get("reasoning", ""),
                            "tables": tables,
                        }
                    else:
                        columns = result.get("columns", [])
                        print(f"  [DEBUG] Columns extracted: {columns}")
                        return {
                            "reasoning": result.get("reasoning", ""),
                            "columns": columns,
                        }
                else:
                    print("  [DEBUG] No matching closing '}' found!")
        except Exception as e:
            print(f"  [DEBUG] Error parsing JSON: {e}")

        # Fallback: extract table/column names from freeform prose using regex
        import re

        if task_type == "table":
            table_mentions = re.findall(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\s+table\b|\btable[s]?\s+([A-Za-z_][A-Za-z0-9_]*)\b|"
                r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)\b|\bjoin\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                response,
                re.IGNORECASE,
            )
            seen = set()
            tables = []
            for groups in table_mentions:
                for g in groups:
                    g = g.strip()
                    if g and g.lower() not in (
                        "the",
                        "a",
                        "an",
                        "and",
                        "or",
                        "in",
                        "on",
                        "to",
                        "table",
                        "tables",
                    ):
                        if g not in seen:
                            seen.add(g)
                            tables.append(g)
            print(f"  [FALLBACK] Extracted tables from prose: {tables}")
            return {"reasoning": response[:200], "tables": tables}
        else:
            col_mentions = re.findall(
                r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", response
            )
            columns = [f"{t}.{c}" for t, c in col_mentions]
            columns = list(dict.fromkeys(columns))
            print(f"  [FALLBACK] Extracted columns from prose: {columns}")
            return {"reasoning": response[:200], "columns": columns}

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

        unique_items: List[str] = []
        seen: set = set()
        all_reasoning = []

        for result in results:
            items = result.get(key, [])
            reasoning = result.get("reasoning", "")
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
        Uses outlines for forced JSON output.

        Args:
            schema: Database schema dictionary
            question: User's natural language question
            evidence: Optional knowledge evidence

        Returns:
            Tuple of (selected tables, reasoning)
        """
        results = []
        self._current_schema = schema

        all_prompts = []
        prompt_configs = []

        for i in range(self.pt):
            shuffled_schema = self.shuffle_schema_order(schema)
            schema_text = self.format_schema_for_prompt(shuffled_schema)
            prompt = self.build_table_linking_prompt(schema_text, question, evidence)

            for j in range(self.n):
                all_prompts.append(prompt)
                prompt_configs.append((i, j))

        print(f"    DEBUG: llm_client type = {type(self.llm_client).__name__}")
        print(
            f"    DEBUG: has generate_json = {hasattr(self.llm_client, 'generate_json')}"
        )
        print(
            f"    Table linking: generating {len(all_prompts)} responses with forced JSON..."
        )

        for idx, prompt in enumerate(all_prompts):
            if self.llm_client:
                result = self.llm_client.generate_json(prompt, TABLE_LINKING_SCHEMA)
                results.append(result)
                if idx < 3:
                    print(f"      Response {idx+1}: tables={result.get('tables', [])}")
            else:
                response = self._mock_llm_call(prompt, "table", schema)
                parsed = self.parse_llm_response(response, "table")
                results.append(parsed)

        tables, reasoning = self.union_results(results, "table")
        print(f"    Table linking complete: selected {len(tables)} tables: {tables}")
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
        Uses outlines for forced JSON output.

        Args:
            schema: Database schema dictionary
            selected_tables: Tables selected from table linking
            question: User's natural language question
            evidence: Optional knowledge evidence

        Returns:
            Tuple of (selected columns, reasoning)
        """
        results = []
        all_prompts = []

        for i in range(self.pc):
            shuffled_schema = self.shuffle_schema_order(schema, selected_tables)
            schema_text = self.format_schema_for_prompt(
                shuffled_schema, selected_tables
            )
            prompt = self.build_column_linking_prompt(
                schema_text, question, selected_tables, evidence
            )

            for j in range(self.n):
                all_prompts.append(prompt)

        print(
            f"    Column linking: generating {len(all_prompts)} responses with forced JSON..."
        )

        for idx, prompt in enumerate(all_prompts):
            if self.llm_client:
                result = self.llm_client.generate_json(prompt, COLUMN_LINKING_SCHEMA)
                results.append(result)
                if idx < 3:
                    print(
                        f"      Response {idx+1}: columns={len(result.get('columns', []))} columns"
                    )
            else:
                response = self._mock_llm_call(prompt, "column", schema)
                parsed = self.parse_llm_response(response, "column")
                results.append(parsed)

        columns, reasoning = self.union_results(results, "column")
        print(f"    Column linking complete: selected {len(columns)} columns")
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
        tables, table_reasoning = self.link_tables(schema, question, evidence)
        columns, column_reasoning = self.link_columns(
            schema, tables, question, evidence
        )

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
            if task_type == "table":
                selected = tables[: min(2, len(tables))]
                return f"""{{
    "reasoning": "Based on the question and available tables ({', '.join(tables)}), the selected tables contain relevant data.",
    "tables": {json.dumps(selected)}
}}"""
            else:
                if tables:
                    first_table = tables[0]
                    columns = schema.get(first_table, [])
                    selected_columns = [
                        f"{first_table}.{col}"
                        for col in columns[: min(3, len(columns))]
                    ]
                    return f"""{{
    "reasoning": "Selected columns from {first_table} table for the query.",
    "columns": {json.dumps(selected_columns)}
}}"""

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
