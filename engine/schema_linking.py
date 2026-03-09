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


@dataclass
class SchemaLinkingResult:
    """Result of schema linking process."""

    tables: List[str]
    columns: List[str]
    reasoning: str


class TransformersLLMClient:
    """LLM client using Hugging Face Transformers for Qwen models."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Initialize the Transformers LLM client.

        Args:
            model_name: Hugging Face model name
            device: Device to run model on ('cuda' or 'cpu')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "Please install transformers and torch: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cuda":
            self.model = self.model.to(device)
        print(f"Model loaded successfully on {device}")

    def generate(self, prompt: str) -> str:
        """
        Generate response from the model.

        Args:
            prompt: Input prompt string

        Returns:
            Generated response string
        """
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt) :]
        return response.strip()


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
            # Try to find JSON in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
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

        Args:
            schema: Database schema dictionary
            question: User's natural language question
            evidence: Optional knowledge evidence

        Returns:
            Tuple of (selected tables, reasoning)
        """
        results = []
        self._current_schema = schema

        for i in range(self.pt):
            # Shuffle schema order for diversity
            shuffled_schema = self.shuffle_schema_order(schema)
            schema_text = self.format_schema_for_prompt(shuffled_schema)

            # Build prompt
            prompt = self.build_table_linking_prompt(schema_text, question, evidence)

            # Generate n outputs
            for _ in range(self.n):
                if self.llm_client:
                    response = self.llm_client.generate(prompt)
                else:
                    # Placeholder for testing - replace with actual LLM call
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

        Args:
            schema: Database schema dictionary
            selected_tables: Tables selected from table linking
            question: User's natural language question
            evidence: Optional knowledge evidence

        Returns:
            Tuple of (selected columns, reasoning)
        """
        results = []

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
            for _ in range(self.n):
                if self.llm_client:
                    response = self.llm_client.generate(prompt)
                else:
                    # Placeholder for testing - replace with actual LLM call
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
    # Initialize Qwen LLM client using Transformers
    # Using Qwen2.5-7B-Instruct (closest publicly available to Qwen 3.5 8B)
    llm_client = TransformersLLMClient(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="cuda",  # Change to "cpu" if no GPU available
        max_new_tokens=512,
        temperature=0.7,
    )

    # Create schema linker with LLM client
    linker = SchemaLinker(pt=3, pc=3, n=1, llm_client=llm_client)

    # Load schema from database
    db_path = r"C:\Repos\MCS-SQL\mini_dev\minidev\minidev\dev_databases\california_schools\california_schools.sqlite"
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
