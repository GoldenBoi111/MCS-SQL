"""
Multiple SQL Generation for Text-to-SQL

This module generates multiple SQL query candidates for a given natural language question
using few-shot learning with examples sampled from the training dataset.

Prompt Strategy (5 prompts total):
1. MASKED-only: Examples retrieved using masked question similarity (literals replaced)
2. UNMASKED-only: Examples retrieved using standard question similarity
3-5. MIXED: Random samples from both masked and unmasked examples, shuffled

Each prompt generates n=20 responses, for a total of 100 SQL candidates.

Two sampling methods for examples:
1. Question Similarity-based: Select top-k questions with closest sentence embeddings
2. Masked Question Similarity-based: Uses masked questions (literals replaced with 
   special tokens) to prevent collusion based on similar variable/literal names.
   Uses LLM to replace table names, column names, and values.

Includes table schema and sample CSV rows to help LLM understand column/table contents.
LLM provides reasoning for each generated SQL.
"""

import json
import os
import pickle
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import faiss
import numpy as np
from tqdm import tqdm

from config import Config
from literal_masker import LiteralMasker
from schema_linking import TransformersLLMClient


@dataclass
class GenerationConfig:
    """Configuration for multiple SQL generation."""
    
    # Number of similar examples to retrieve
    top_k: int = 20
    
    # Number of examples to use per prompt (sampled from top_k)
    examples_per_prompt: int = 10
    
    # Number of different prompts
    num_prompts: int = 5
    # Breakdown:
    #   - 1 prompt: only masked SQL examples
    #   - 1 prompt: only unmasked SQL examples
    #   - 3 prompts: mixed random examples
    
    # Number of responses per prompt
    responses_per_prompt: int = 20
    
    # Total responses = num_prompts * responses_per_prompt
    # Default: 5 * 20 = 100
    
    # Include table schema in prompts
    include_schema: bool = True
    
    # Include sample rows in prompts
    include_sample_rows: bool = True
    
    # Number of sample rows per table
    num_sample_rows: int = 3


class MultipleSQLGenerator:
    """
    Generates multiple SQL query candidates using few-shot learning.
    
    Attributes:
        config: Generation configuration
        llm_client: LLM client for generating responses
        literal_masker: For masking questions
        embedding_model: Sentence transformer for question embeddings
        faiss_index: FAISS index for similarity search
        training_data: Loaded training dataset
    """
    
    def __init__(
        self,
        config: Config,
        llm_client: Optional[TransformersLLMClient] = None,
        use_masked_similarity: bool = False,
    ):
        """
        Initialize the multiple SQL generator.
        
        Args:
            config: Configuration object
            llm_client: Optional LLM client (created if not provided)
            use_masked_similarity: Whether to use masked questions for similarity
        """
        self.config = config
        self.generation_config = GenerationConfig(
            use_masked_similarity=use_masked_similarity
        )
        
        # Initialize components
        self.llm_client = llm_client
        self.literal_masker = None
        self.embedding_model = None
        self.faiss_index = None
        self.training_data = None
        
        # Load FAISS index and training data
        self._load_faiss_index()
        
    def _load_faiss_index(self):
        """Load FAISS index and training data for similarity search."""
        from sentence_transformers import SentenceTransformer
        
        # Determine which index to use
        if self.generation_config.use_masked_similarity:
            index_path = self.config.FAISS_INDEX_MASKED
            print(f"Loading masked FAISS index from: {index_path}")
        else:
            index_path = self.config.FAISS_INDEX
            print(f"Loading standard FAISS index from: {index_path}")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run training_dataset_indexer.py or training_dataset_indexer_masked.py first."
            )
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(
            os.path.join(index_path, "faiss_index.bin")
        )
        
        # Load question store
        if self.generation_config.use_masked_similarity:
            with open(os.path.join(index_path, "masked_question_store.pkl"), "rb") as f:
                self.question_store = pickle.load(f)
        else:
            with open(os.path.join(index_path, "question_store.pkl"), "rb") as f:
                self.question_store = pickle.load(f)
        
        # Load SQL store
        with open(os.path.join(index_path, "original_sql_store.pkl"), "rb") as f:
            self.sql_store = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(index_path, "metadata_store.pkl"), "rb") as f:
            self.metadata_store = pickle.load(f)
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            self.config.EMBEDDING_MODEL_NAME
        )
        
        print(f"Loaded {len(self.question_store)} training examples")
        
    def _init_masker(self):
        """Initialize literal masker if using masked similarity."""
        if self.generation_config.use_masked_similarity and self.literal_masker is None:
            self.literal_masker = LiteralMasker(llm_client=self.llm_client)
    
    def find_similar_examples(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find top-k similar questions from training set.
        
        Args:
            question: Natural language question
            top_k: Number of examples to retrieve (default: config.top_k)
            
        Returns:
            List of dicts with question, sql, metadata, and similarity score
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not loaded")
        
        k = top_k or self.generation_config.top_k
        k = min(k, len(self.question_store))
        
        # Mask question if using masked similarity
        if self.generation_config.use_masked_similarity:
            self._init_masker()
            query_text = self.literal_masker.mask_question(question)
            print(f"  Original: {question}")
            print(f"  Masked:   {query_text}")
        else:
            query_text = question
        
        # Get embedding
        query_embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.question_store):
                results.append({
                    "question": self.question_store[idx],
                    "sql": self.sql_store[idx],
                    "metadata": self.metadata_store[idx],
                    "similarity_score": float(score),
                })
        
        return results
    
    def load_database_schema(self, db_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load database schema with sample rows.
        
        Args:
            db_path: Path to SQLite database
            
        Returns:
            Dictionary with table info including columns and sample rows
        """
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            # Get column info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [
                {"name": row[1], "type": row[2]}
                for row in cursor.fetchall()
            ]
            
            # Get sample rows
            sample_rows = []
            if self.generation_config.include_sample_rows:
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT {self.generation_config.num_sample_rows}")
                    rows = cursor.fetchall()
                    col_names = [col["name"] for col in columns]
                    sample_rows = [dict(zip(col_names, row)) for row in rows]
                except Exception as e:
                    print(f"  Warning: Could not get sample rows for {table}: {e}")
            
            schema[table] = {
                "columns": columns,
                "sample_rows": sample_rows,
            }
        
        conn.close()
        return schema
    
    def format_schema_for_prompt(
        self,
        schema: Dict[str, Dict[str, Any]],
        tables_to_include: Optional[List[str]] = None,
    ) -> str:
        """
        Format schema as text for LLM prompt.
        
        Args:
            schema: Database schema dictionary
            tables_to_include: Optional list of tables to include
            
        Returns:
            Formatted schema string
        """
        lines = []
        tables = tables_to_include if tables_to_include else schema.keys()
        
        for table in tables:
            if table in schema:
                table_info = schema[table]
                columns = table_info["columns"]
                sample_rows = table_info["sample_rows"]
                
                # Table header
                col_defs = ", ".join([f"{col['name']}: {col['type']}" for col in columns])
                lines.append(f"Table: {table} ({col_defs})")
                
                # Sample rows as CSV
                if sample_rows:
                    lines.append(f"  Sample rows ({len(sample_rows)}):")
                    
                    # CSV header
                    col_names = [col["name"] for col in columns]
                    lines.append(f"    {','.join(col_names)}")
                    
                    # CSV rows
                    for row in sample_rows:
                        row_values = [str(row.get(col, "")) for col in col_names]
                        lines.append(f"    {','.join(row_values)}")
                
                lines.append("")
        
        return "\n".join(lines)
    
    def build_few_shot_prompt(
        self,
        question: str,
        examples: List[Dict[str, Any]],
        schema: Optional[str] = None,
        evidence: str = "",
        shuffle_examples: bool = True,
    ) -> str:
        """
        Build a few-shot prompt with examples.
        
        Args:
            question: Target natural language question
            examples: List of example dicts with question, sql, metadata
            schema: Optional database schema text
            evidence: Optional knowledge evidence
            shuffle_examples: Whether to shuffle example order
            
        Returns:
            Complete prompt string
        """
        # Shuffle examples if requested
        if shuffle_examples:
            examples = examples.copy()
            random.shuffle(examples)
        
        # Build prompt
        prompt_parts = []
        
        # Header
        prompt_parts.append("""### Task: Given a database schema, example question-SQL pairs, and a new question,
generate the correct SQLite SQL query for the new question.

Provide detailed reasoning for your answer.

Your answer should strictly follow the following JSON format:
{
    "reasoning": "", // Detailed reasoning steps
    "sql": "" // The final SQL query
}

""")
        
        # Schema (if provided)
        if schema:
            prompt_parts.append("### Database Schema:\n")
            prompt_parts.append(schema)
            prompt_parts.append("\n")
        
        # Examples
        prompt_parts.append("### Example Question-SQL Pairs:\n")
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Question: {example['question']}")
            if example.get('metadata', {}).get('evidence'):
                prompt_parts.append(f"Knowledge Evidence: {example['metadata']['evidence']}")
            prompt_parts.append(f"Gold SQL: {example['sql']}")
            prompt_parts.append("")
        
        # Target question
        prompt_parts.append("### Your Task:\n")
        prompt_parts.append(f"Question: {question}")
        if evidence:
            prompt_parts.append(f"Knowledge Evidence: {evidence}")
        prompt_parts.append("\n### Your Answer (JSON format):")
        
        return "\n".join(prompt_parts)
    
    def _sample_examples_for_prompt(
        self,
        question: str,
        prompt_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Sample examples based on prompt type.
        
        Args:
            question: Target question
            prompt_type: "masked", "unmasked", or "mixed"
            
        Returns:
            List of example dicts
        """
        # Get both standard and masked examples
        print("  Retrieving standard similarity examples...")
        self.generation_config.use_masked_similarity = False
        self._load_faiss_index()
        standard_examples = self.find_similar_examples(
            question,
            top_k=self.generation_config.top_k
        )
        
        print("  Retrieving masked similarity examples...")
        self.generation_config.use_masked_similarity = True
        self._load_faiss_index()
        masked_examples = self.find_similar_examples(
            question,
            top_k=self.generation_config.top_k
        )
        
        n = self.generation_config.examples_per_prompt
        
        if prompt_type == "masked":
            # Use only masked examples
            print("  Using only masked SQL examples")
            return masked_examples[:n]
        
        elif prompt_type == "unmasked":
            # Use only standard (unmasked) examples
            print("  Using only unmasked SQL examples")
            return standard_examples[:n]
        
        else:  # mixed
            # Randomly sample from both, ensuring diversity
            print("  Mixing standard and masked examples")
            mixed = []
            
            # Take top 5 from each (ensures both types represented)
            mixed.extend(standard_examples[:5])
            mixed.extend(masked_examples[:5])
            
            # Randomly sample remaining from top 20 of each
            remaining = n - 10
            if remaining > 0:
                standard_remaining = standard_examples[5:]
                masked_remaining = masked_examples[5:]
                additional = random.sample(
                    standard_remaining + masked_remaining,
                    min(remaining, len(standard_remaining) + len(masked_remaining))
                )
                mixed.extend(additional)
            
            # Shuffle to avoid ordering bias
            random.shuffle(mixed)
            return mixed[:n]
    
    def generate_sql(
        self,
        question: str,
        db_path: str,
        evidence: str = "",
        prompt_type: str = "mixed",
    ) -> List[Dict[str, str]]:
        """
        Generate multiple SQL candidates for a question.
        
        Args:
            question: Natural language question
            db_path: Path to SQLite database
            evidence: Optional knowledge evidence
            prompt_type: Type of prompt - "masked", "unmasked", or "mixed"
            
        Returns:
            List of dicts with sql, reasoning, and prompt_id
        """
        if self.llm_client is None:
            raise ValueError("LLM client not initialized")
        
        # Load schema
        print(f"Loading database schema from: {db_path}")
        schema = self.load_database_schema(db_path)
        schema_text = self.format_schema_for_prompt(schema)
        
        # Sample examples based on prompt type
        print(f"Finding similar examples for question (type: {prompt_type})...")
        examples = self._sample_examples_for_prompt(question, prompt_type)
        print(f"Sampled {len(examples)} examples for {prompt_type} prompt")
        
        # Generate multiple prompts with shuffled examples
        all_responses = []
        
        for prompt_id in range(self.generation_config.num_prompts):
            print(f"\nGenerating responses for prompt {prompt_id + 1}/{self.generation_config.num_prompts}")
            
            # Build prompt with shuffled examples
            prompt = self.build_few_shot_prompt(
                question=question,
                examples=examples,
                schema=schema_text,
                evidence=evidence,
                shuffle_examples=True,  # Always shuffle for diversity
            )
            
            # Generate multiple responses
            for response_id in range(self.generation_config.responses_per_prompt):
                try:
                    response = self.llm_client.generate(prompt)
                    parsed = self._parse_sql_response(response)
                    
                    if parsed:
                        parsed["prompt_id"] = prompt_id
                        parsed["response_id"] = response_id
                        parsed["prompt_type"] = prompt_type
                        parsed["examples_used"] = len(examples)
                        all_responses.append(parsed)
                        
                except Exception as e:
                    print(f"  Error generating response {response_id}: {e}")
        
        print(f"\nGenerated {len(all_responses)} SQL candidates")
        return all_responses
    
    def _parse_sql_response(self, response: str) -> Optional[Dict[str, str]]:
        """
        Parse LLM response to extract SQL and reasoning.

        Args:
            response: Raw LLM response

        Returns:
            Dict with sql and reasoning, or None if parsing fails
        """
        try:
            # Try to find JSON in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)

                return {
                    "sql": result.get("sql", ""),
                    "reasoning": result.get("reasoning", ""),
                }
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Error parsing response: {e}")

        # Fallback: try to extract SQL with regex
        import re
        sql_match = re.search(r'SELECT.*?(?:;|$)', response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return {
                "sql": sql_match.group(0).strip(),
                "reasoning": response[:200] + "...",  # Truncate reasoning
            }

        return None


def main():
    """Example usage of MultipleSQLGenerator."""
    # Load configuration
    config = Config()
    
    print("MCS-SQL Multiple SQL Generation")
    print("="*60)
    
    # Initialize LLM client
    print("Initializing LLM client...")
    llm_client = TransformersLLMClient(
        model_name=config.LLM_MODEL_NAME,
        device=config.LLM_DEVICE,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=config.LLM_TEMPERATURE,
    )
    
    # Create generator
    generator = MultipleSQLGenerator(
        config=config,
        llm_client=llm_client,
    )
    
    # Example question
    question = "What is the average SAT score of schools in Los Angeles county?"
    evidence = "Average SAT score is calculated by taking the mean of all SAT scores."
    
    # Get database path
    db_path = config.get_database_path()
    
    print(f"\nQuestion: {question}")
    print(f"Database: {db_path}")
    print(f"\nPrompt Strategy:")
    print(f"  - 1 prompt with only MASKED SQL examples")
    print(f"  - 1 prompt with only UNMASKED SQL examples")
    print(f"  - 3 prompts with MIXED random examples")
    print(f"  - {generator.generation_config.examples_per_prompt} examples per prompt")
    print(f"  - {generator.generation_config.responses_per_prompt} responses per prompt")
    print(f"  - Total: {generator.generation_config.num_prompts * generator.generation_config.responses_per_prompt} SQL candidates")
    
    # Generate SQL candidates for each prompt type
    all_responses = []
    
    # 1. Masked-only prompt
    print("\n" + "="*60)
    print("Generating MASKED-only prompt responses...")
    print("="*60)
    masked_responses = generator.generate_sql(
        question=question,
        db_path=db_path,
        evidence=evidence,
        prompt_type="masked",
    )
    all_responses.extend(masked_responses)
    
    # 2. Unmasked-only prompt
    print("\n" + "="*60)
    print("Generating UNMASKED-only prompt responses...")
    print("="*60)
    unmasked_responses = generator.generate_sql(
        question=question,
        db_path=db_path,
        evidence=evidence,
        prompt_type="unmasked",
    )
    all_responses.extend(unmasked_responses)
    
    # 3. Mixed prompts (3 different prompts)
    for i in range(3):
        print("\n" + "="*60)
        print(f"Generating MIXED prompt {i+1}/3 responses...")
        print("="*60)
        mixed_responses = generator.generate_sql(
            question=question,
            db_path=db_path,
            evidence=evidence,
            prompt_type="mixed",
        )
        all_responses.extend(mixed_responses)
    
    # Save results
    output_dir = config.PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "multiple_generation_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "question": question,
            "evidence": evidence,
            "db_path": str(db_path),
            "config": {
                "num_prompts": generator.generation_config.num_prompts,
                "examples_per_prompt": generator.generation_config.examples_per_prompt,
                "responses_per_prompt": generator.generation_config.responses_per_prompt,
                "top_k": generator.generation_config.top_k,
            },
            "summary": {
                "total_responses": len(all_responses),
                "masked_only": len(masked_responses),
                "unmasked_only": len(unmasked_responses),
                "mixed": len(all_responses) - len(masked_responses) - len(unmasked_responses),
            },
            "responses": all_responses,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    print(f"Total SQL candidates generated: {len(all_responses)}")
    
    # Show unique SQL queries
    unique_sqls = list(set(r["sql"] for r in all_responses if r["sql"]))
    print(f"Unique SQL queries: {len(unique_sqls)}")
    
    # Show breakdown by prompt type
    print(f"\nBreakdown by prompt type:")
    print(f"  Masked-only:   {len(masked_responses)} responses")
    print(f"  Unmasked-only: {len(unmasked_responses)} responses")
    print(f"  Mixed:         {len(all_responses) - len(masked_responses) - len(unmasked_responses)} responses")
    
    # Show first few examples
    print("\nFirst 3 generated SQL queries:")
    for i, response in enumerate(all_responses[:3], 1):
        print(f"\n{i}. Prompt Type: {response.get('prompt_type', 'unknown')}")
        print(f"   SQL: {response['sql'][:100]}...")
        print(f"   Reasoning: {response['reasoning'][:100]}...")


if __name__ == "__main__":
    main()
