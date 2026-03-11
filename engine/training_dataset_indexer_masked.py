"""
Masked Training Dataset Indexer for Text-to-SQL

This script builds a FAISS index using MASKED questions where literals
(numbers, strings, dates, etc.) are replaced with placeholders.

This helps generalize better - semantically similar questions with different
values will map to the same embedding.

Example:
  Original: "What is the ratio of customers who pay in EUR against CZK?"
  Masked:   "What is the ratio of customers who pay in [CURRENCY] against [CURRENCY]?"

This version uses LLM-based masking (Qwen) to identify and replace:
- Table names → [TABLE]
- Column names → [COLUMN]
- Literal values → [VALUE]

Note: This is slower than regex-based masking but provides better generalization.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
import faiss
import numpy as np
from tqdm import tqdm

from literal_masker import LiteralMasker
from prompt_manager import PromptManager
from config import Config
from schema_linking import TransformersLLMClient


class MaskedTrainingDatasetIndexer:
    """
    Indexer for training dataset using masked questions for better generalization.
    Uses LLM-based masking to identify table names, column names, and values.
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        index_type: str = "HNSW",
        llm_client: Optional[Any] = None,
        prompts_dir: Optional[str] = None,
        databases_path: Optional[str] = None,
        use_llm_masking: bool = True,
    ):
        """
        Initialize the masked training dataset indexer.

        Args:
            embedding_model_name: Name of the sentence transformer model for embeddings
            index_type: FAISS index type (Flat, IVF, HNSW)
            llm_client: LLM client for masking (created if not provided)
            prompts_dir: Path to prompts directory
            databases_path: Path to database directory (for schema lookup)
            use_llm_masking: Whether to use LLM-based masking (slower but better)
        """
        self.embedding_model_name = embedding_model_name
        self.index_type = index_type
        self.embedding_model: Optional[Any] = None
        self.index: Optional[faiss.Index] = None
        self.masked_question_store: List[str] = []
        self.original_question_store: List[str] = []
        self.masked_sql_store: List[str] = []
        self.original_sql_store: List[str] = []
        self.metadata_store: List[Dict[str, Any]] = []
        self.dimension: int = 0
        self.databases_path = databases_path
        self.use_llm_masking = use_llm_masking
        self.schema_cache: Dict[str, str] = {}  # Cache schemas by db_id

        # Initialize LLM client for masking
        self.llm_client = llm_client
        self.prompt_manager = None
        if prompts_dir and os.path.exists(prompts_dir):
            self.prompt_manager = PromptManager(prompts_dir)
            print(f"Loaded prompt templates from: {prompts_dir}")

        self.literal_masker = LiteralMasker(
            llm_client=self.llm_client, prompt_manager=self.prompt_manager
        )

    def load_model(self):
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")

    def _get_schema_for_db(self, db_id: str) -> Optional[str]:
        """
        Load and format schema for a given database with column types.
        Uses JSON cache to avoid reloading schemas on subsequent runs.

        Args:
            db_id: Database identifier

        Returns:
            Formatted schema string with column types or None if not found
        """
        # Check in-memory cache first
        if db_id in self.schema_cache:
            return self.schema_cache[db_id]

        # Try to load from JSON cache file
        cache_file = (
            Path(self.databases_path) / "schema_cache.json"
            if self.databases_path
            else None
        )
        if cache_file and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    all_schemas = json.load(f)
                    if db_id in all_schemas:
                        schema_text = all_schemas[db_id]
                        self.schema_cache[db_id] = schema_text  # Also cache in memory
                        return schema_text
            except Exception as e:
                print(f"  Warning: Could not read schema cache: {e}")

        # Not in cache - load from database and save it
        if not self.databases_path:
            return None

        import sqlite3

        db_dir = Path(self.databases_path) / db_id
        if not db_dir.exists():
            return None

        db_path = db_dir / f"{db_id}.sqlite"
        if not db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get all tables (including those with special names)
            cursor.execute(
                'SELECT name FROM sqlite_master WHERE type="table" AND name NOT LIKE "sqlite_%"'
            )
            tables = [row[0] for row in cursor.fetchall()]

            schema_lines = []
            for table in tables:
                # Escape table names with special characters using brackets
                # This handles: reserved words, dashes, spaces, etc.
                try:
                    cursor.execute(f'PRAGMA table_info("[{table}]")')
                    columns = cursor.fetchall()

                    # Format: table ( col1: type, col2: type, ... )
                    col_defs = []
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2] if col[2] else "text"
                        col_defs.append(f"{col_name}: {col_type}")

                    schema_lines.append(f"# {table} ( {', '.join(col_defs)} )")
                except Exception as col_err:
                    print(
                        f"  Warning: Could not get columns for {table} in {db_id}: {col_err}"
                    )
                    continue

            conn.close()
            schema_text = (
                "\n\n".join(schema_lines) if schema_lines else "# No tables found"
            )

            # Cache in memory
            self.schema_cache[db_id] = schema_text

            # Save to JSON cache file
            if cache_file:
                self._save_schema_to_cache(db_id, schema_text, cache_file)

            return schema_text
        except Exception as e:
            print(f"  Warning: Could not load schema for {db_id}: {e}")
            return None

    def preload_schemas(self, db_ids: List[str]):
        """
        Preload schemas for all unique database IDs.
        Checks cache first, only loads missing schemas from databases.

        Args:
            db_ids: List of unique database IDs to preload
        """
        if not self.databases_path:
            return

        cache_file = Path(self.databases_path) / "schema_cache.json"

        # Load existing cache
        all_schemas = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    all_schemas = json.load(f)
                    print(f"Loaded {len(all_schemas)} schemas from cache")
            except Exception as e:
                print(f"  Warning: Could not read schema cache: {e}")

        # Find which schemas are missing
        missing_schemas = []
        for db_id in db_ids:
            if db_id in all_schemas:
                self.schema_cache[db_id] = all_schemas[db_id]
            else:
                missing_schemas.append(db_id)

        # Load missing schemas
        if missing_schemas:
            print(f"Loading {len(missing_schemas)} missing schemas from databases...")
            for db_id in tqdm(missing_schemas, desc="Loading schemas"):
                schema = self._get_schema_for_db(db_id)
                if schema:
                    all_schemas[db_id] = schema

            # Save complete cache
            if cache_file and all_schemas:
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(all_schemas, f, indent=2, ensure_ascii=False)
                    print(f"Saved {len(all_schemas)} schemas to cache: {cache_file}")
                except Exception as e:
                    print(f"  Warning: Could not save schema cache: {e}")
        else:
            print("All schemas found in cache!")

    def _save_schema_to_cache(self, db_id: str, schema_text: str, cache_file: Path):
        """
        Save a schema to the JSON cache file.

        Args:
            db_id: Database identifier
            schema_text: Formatted schema string
            cache_file: Path to cache JSON file
        """
        # Load existing cache
        all_schemas = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    all_schemas = json.load(f)
            except Exception:
                all_schemas = {}

        # Add new schema
        all_schemas[db_id] = schema_text

        # Save back to file
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(all_schemas, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Warning: Could not save schema cache: {e}")

    def save_checkpoint(
        self,
        checkpoint_path: str,
        processed_count: int,
        start_index: int = 0,
        masked_questions: Optional[List[str]] = None,
        original_questions: Optional[List[str]] = None,
        masked_sqls: Optional[List[str]] = None,
        original_sqls: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Save a checkpoint that can be resumed later.

        Args:
            checkpoint_path: Directory to save checkpoint
            processed_count: Number of entries processed so far
            start_index: Index in dataset where processing stopped
            masked_questions: List of masked questions (optional, uses self.masked_question_store if not provided)
            original_questions: List of original questions (optional, uses self.original_question_store if not provided)
            masked_sqls: List of masked SQLs (optional, uses self.masked_sql_store if not provided)
            original_sqls: List of original SQLs (optional, uses self.original_sql_store if not provided)
            metadata: List of metadata (optional, uses self.metadata_store if not provided)
        """
        os.makedirs(checkpoint_path, exist_ok=True)

        # Use provided lists or fall back to instance stores
        mq = (
            self.masked_question_store + masked_questions
            if masked_questions is not None
            else self.masked_question_store
        )
        oq = (
            self.original_question_store + original_questions
            if original_questions is not None
            else self.original_question_store
        )
        ms = (
            self.masked_sql_store + masked_sqls
            if masked_sqls is not None
            else self.masked_sql_store
        )
        osql = (
            self.original_sql_store + original_sqls
            if original_sqls is not None
            else self.original_sql_store
        )
        md = (
            self.metadata_store + metadata
            if metadata is not None
            else self.metadata_store
        )

        # Save partial data
        with open(os.path.join(checkpoint_path, "masked_questions.pkl"), "wb") as f:
            pickle.dump(mq, f)
        with open(os.path.join(checkpoint_path, "original_questions.pkl"), "wb") as f:
            pickle.dump(oq, f)
        with open(os.path.join(checkpoint_path, "masked_sqls.pkl"), "wb") as f:
            pickle.dump(ms, f)
        with open(os.path.join(checkpoint_path, "original_sqls.pkl"), "wb") as f:
            pickle.dump(osql, f)
        with open(os.path.join(checkpoint_path, "metadata.pkl"), "wb") as f:
            pickle.dump(md, f)

        # Save progress
        checkpoint_data = {
            "processed_count": processed_count,
            "start_index": start_index,  # Where to resume from
            "embedding_model_name": self.embedding_model_name,
            "index_type": self.index_type,
        }
        with open(os.path.join(checkpoint_path, "checkpoint.json"), "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        print(
            f"  Checkpoint saved: {processed_count} entries processed (index {start_index})"
        )

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str, config: Config
    ) -> Optional[
        Tuple["MaskedTrainingDatasetIndexer", int, int, List, List, List, List, List]
    ]:
        """
        Load a checkpoint to resume processing.

        Args:
            checkpoint_path: Directory containing checkpoint files
            config: Configuration object

        Returns:
            Tuple of (indexer, processed_count, resume_index, masked_q, orig_q, masked_sql, orig_sql, metadata)
            or None if no checkpoint exists
        """
        if not os.path.exists(checkpoint_path):
            return None

        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.json")
        if not os.path.exists(checkpoint_file):
            return None

        try:
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            processed_count = checkpoint_data.get("processed_count", 0)
            resume_index = checkpoint_data.get("start_index", processed_count)

            # Create indexer
            indexer = cls(
                embedding_model_name=checkpoint_data.get(
                    "embedding_model_name", config.EMBEDDING_MODEL_NAME
                ),
                index_type=checkpoint_data.get("index_type", config.FAISS_INDEX_TYPE),
            )

            # Load partial data
            with open(os.path.join(checkpoint_path, "masked_questions.pkl"), "rb") as f:
                masked_q = pickle.load(f)
            with open(
                os.path.join(checkpoint_path, "original_questions.pkl"), "rb"
            ) as f:
                orig_q = pickle.load(f)
            with open(os.path.join(checkpoint_path, "masked_sqls.pkl"), "rb") as f:
                masked_sql = pickle.load(f)
            with open(os.path.join(checkpoint_path, "original_sqls.pkl"), "rb") as f:
                orig_sql = pickle.load(f)
            with open(os.path.join(checkpoint_path, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)

            # Assign loaded data to indexer's stores
            indexer.masked_question_store = masked_q
            indexer.original_question_store = orig_q
            indexer.masked_sql_store = masked_sql
            indexer.original_sql_store = orig_sql
            indexer.metadata_store = metadata

            print(f"Checkpoint loaded: {processed_count} entries already processed")
            print(f"Resume index: {resume_index}")
            return (
                indexer,
                processed_count,
                resume_index,
                masked_q,
                orig_q,
                masked_sql,
                orig_sql,
                metadata,
            )
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            return None

    def create_index(self):
        """Create FAISS index."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVF":
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "HNSW":
            M = 32
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        print(f"Created {self.index_type} FAISS index")

    def load_dataset(
        self,
        dataset_path: str,
        max_entries: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 500,
        resume_index: int = 0,
    ) -> Tuple[List[str], List[str], List[str], List[str], List[Dict[str, Any]]]:
        """
        Load dataset and mask questions/SQL.
        Preloads all schemas from cache, only loading missing ones from databases.
        Supports checkpointing for resumable processing.

        Args:
            dataset_path: Path to JSON array file
            max_entries: Maximum entries to load
            checkpoint_path: Optional directory to save/load checkpoints
            checkpoint_every: Save checkpoint every N entries
            resume_index: Index to start from (for resuming)

        Returns:
            Tuple of (masked_questions, original_questions, masked_sqls, original_sqls, metadata)
        """
        masked_questions = []
        original_questions = []
        masked_sqls = []
        original_sqls = []
        metadata = []

        print(f"Loading dataset from: {dataset_path}")
        file_size_gb = os.path.getsize(dataset_path) / (1024**3)
        print(f"File size: {file_size_gb:.2f} GB")

        # First pass: collect all unique db_ids
        print("\nScanning dataset for unique databases...")
        unique_db_ids = set()
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for entry in data[:max_entries]:
                    if "db_id" in entry:
                        unique_db_ids.add(entry["db_id"])

        print(
            f"Found {len(unique_db_ids)} unique databases: {', '.join(sorted(unique_db_ids)[:10])}{'...' if len(unique_db_ids) > 10 else ''}"
        )

        # Preload all schemas (checks cache first, loads missing)
        if self.use_llm_masking and self.databases_path:
            print("\nPreloading schemas...")
            self.preload_schemas(list(unique_db_ids))

        # Second pass: load and mask
        print(f"\nLoading and masking entries (starting from index {resume_index})...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                # Skip already processed entries
                for idx, entry in tqdm(
                    enumerate(data[resume_index:max_entries], start=resume_index),
                    total=len(data[:max_entries]),
                    initial=resume_index,
                    desc="Loading & Masking",
                ):

                    if "question" in entry and "SQL" in entry:
                        orig_question = entry["question"]
                        orig_sql = entry["SQL"]
                        db_id = entry.get("db_id", "")
                        evidence = entry.get("evidence", "")

                        # Get schema for this database (from cache or load)
                        schema = None
                        if self.use_llm_masking and self.databases_path:
                            schema = self._get_schema_for_db(db_id)

                        # Mask question using LLM (with schema) or regex (without)
                        masked_question = self.literal_masker.mask_question(
                            orig_question, schema=schema, evidence=evidence
                        )
                        masked_sql = self.literal_masker.mask_sql(orig_sql)

                        masked_questions.append(masked_question)
                        original_questions.append(orig_question)
                        masked_sqls.append(masked_sql)
                        original_sqls.append(orig_sql)
                        metadata.append(
                            {
                                "db_id": db_id,
                                "evidence": evidence,
                                "original_sql": orig_sql,
                                "masked_sql": masked_sql,
                                "difficulty": entry.get("difficulty", "unknown"),
                            }
                        )

                        # Save checkpoint periodically
                        if (
                            checkpoint_path
                            and len(masked_questions) % checkpoint_every == 0
                        ):
                            self.save_checkpoint(
                                checkpoint_path,
                                len(masked_questions) + resume_index,
                                idx + 1,
                                masked_questions=masked_questions,
                                original_questions=original_questions,
                                masked_sqls=masked_sqls,
                                original_sqls=original_sqls,
                                metadata=metadata,
                            )

        print(f"Extracted and masked {len(masked_questions)} question-SQL pairs")
        return (
            masked_questions,
            original_questions,
            masked_sqls,
            original_sqls,
            metadata,
        )

    def build_index(
        self,
        masked_questions: List[str],
        original_questions: List[str],
        masked_sqls: List[str],
        original_sqls: List[str],
        metadata: List[Dict[str, Any]],
        batch_size: int = 10000,
    ):
        """Build FAISS index from masked questions."""
        if self.embedding_model is None:
            self.load_model()
        if self.index is None:
            self.create_index()

        print(f"Building index for {len(masked_questions)} masked questions...")

        if self.index_type == "IVF":
            print("Training IVF index...")
            sample_size = min(10000, len(masked_questions))
            sample_embeddings = self.embedding_model.encode(
                masked_questions[:sample_size],
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            sample_embeddings = np.asarray(sample_embeddings, dtype=np.float32)
            self.index.train(sample_embeddings)

        self.masked_question_store = []
        self.original_question_store = []
        self.masked_sql_store = []
        self.original_sql_store = []
        self.metadata_store = []

        for i in tqdm(range(0, len(masked_questions), batch_size), desc="Indexing"):
            batch_masked = masked_questions[i : i + batch_size]
            batch_orig = original_questions[i : i + batch_size]
            batch_masked_sql = masked_sqls[i : i + batch_size]
            batch_orig_sql = original_sqls[i : i + batch_size]
            batch_metadata = metadata[i : i + batch_size]

            embeddings = self.embedding_model.encode(
                batch_masked,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            self.index.add(embeddings)

            self.masked_question_store.extend(batch_masked)
            self.original_question_store.extend(batch_orig)
            self.masked_sql_store.extend(batch_masked_sql)
            self.original_sql_store.extend(batch_orig_sql)
            self.metadata_store.extend(batch_metadata)

        print(f"Index built with {self.index.ntotal} entries")

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, str, str, str, Dict[str, Any], float]]:
        """
        Search for similar questions using masked query.

        Args:
            query: Search query string (will be masked automatically)
            top_k: Number of results to return

        Returns:
            List of (masked_question, original_question, masked_sql, original_sql, metadata, score) tuples
        """
        if self.index is None or self.embedding_model is None:
            raise ValueError("Index not built.")

        # Mask the query using LLM
        masked_query = self.literal_masker.mask_question(query)
        print(f"  Original query: {query}")
        print(f"  Masked query:   {masked_query}")

        query_embedding = self.embedding_model.encode(
            [masked_query], normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.masked_question_store):
                results.append(
                    (
                        self.masked_question_store[idx],
                        self.original_question_store[idx],
                        self.masked_sql_store[idx],
                        self.original_sql_store[idx],
                        self.metadata_store[idx],
                        float(score),
                    )
                )
        return results

    def save(self, save_path: str):
        """Save index and data to disk."""
        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_path, "faiss_index.bin"))

        with open(os.path.join(save_path, "masked_question_store.pkl"), "wb") as f:
            pickle.dump(self.masked_question_store, f)
        with open(os.path.join(save_path, "original_question_store.pkl"), "wb") as f:
            pickle.dump(self.original_question_store, f)
        with open(os.path.join(save_path, "masked_sql_store.pkl"), "wb") as f:
            pickle.dump(self.masked_sql_store, f)
        with open(os.path.join(save_path, "original_sql_store.pkl"), "wb") as f:
            pickle.dump(self.original_sql_store, f)
        with open(os.path.join(save_path, "metadata_store.pkl"), "wb") as f:
            pickle.dump(self.metadata_store, f)

        config = {
            "embedding_model_name": self.embedding_model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "masked": True,  # Flag to indicate this is a masked index
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Masked index saved to: {save_path}")

    def load(self, load_path: str):
        """Load index and data from disk."""
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)

        self.embedding_model_name = config["embedding_model_name"]
        self.index_type = config["index_type"]
        self.dimension = config["dimension"]

        self.load_model()
        self.index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))

        with open(os.path.join(load_path, "masked_question_store.pkl"), "rb") as f:
            self.masked_question_store = pickle.load(f)
        with open(os.path.join(load_path, "original_question_store.pkl"), "rb") as f:
            self.original_question_store = pickle.load(f)
        with open(os.path.join(load_path, "masked_sql_store.pkl"), "rb") as f:
            self.masked_sql_store = pickle.load(f)
        with open(os.path.join(load_path, "original_sql_store.pkl"), "rb") as f:
            self.original_sql_store = pickle.load(f)
        with open(os.path.join(load_path, "metadata_store.pkl"), "rb") as f:
            self.metadata_store = pickle.load(f)

        print(f"Masked index loaded from: {load_path}")
        print(f"Total entries: {len(self.masked_question_store)}")


def main():
    """Main function - builds or loads masked index with checkpoint support."""
    import shutil

    # Load configuration
    config = Config()

    dataset_path = config.TRAIN_DATASET
    index_path = config.FAISS_INDEX_MASKED
    prompts_dir = config.PROMPTS_DIR
    databases_path = config.DEV_DATABASES

    # Checkpoint configuration
    checkpoint_path = str(index_path) + ".checkpoint"  # Save checkpoint next to index
    checkpoint_every = 500  # Save every 500 entries

    # Determine target entry count
    max_entries = config.MAX_ENTRIES if config.MAX_ENTRIES else None

    print(f"Configuration loaded:")
    print(f"  Train Dataset: {dataset_path}")
    print(f"  FAISS Index Path: {index_path}")
    print(f"  Prompts Directory: {prompts_dir}")
    print(f"  Databases Path: {databases_path}")
    print(f"  Checkpoint Path: {checkpoint_path}")
    print(f"  Checkpoint Interval: Every {checkpoint_every} entries")
    print(f"  Target entries: {max_entries if max_entries else 'ALL'}")

    # Try to load existing checkpoint
    checkpoint_data = MaskedTrainingDatasetIndexer.load_checkpoint(
        checkpoint_path, config
    )

    # Determine if we should resume from checkpoint
    should_resume = False
    if checkpoint_data:
        (
            indexer,
            processed_count,
            resume_index,
            masked_q,
            orig_q,
            masked_sql,
            orig_sql,
            metadata,
        ) = checkpoint_data

        # Check if checkpoint is incomplete (has fewer entries than target)
        target_count = max_entries if max_entries else float("inf")
        if processed_count < target_count:
            should_resume = True
            print(f"\n{'='*60}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"{'='*60}")
            print(f"Already processed: {processed_count} entries")
            print(f"Resuming from index: {resume_index}")
            print(f"Target: {target_count} entries")
            print(
                f"Remaining: {target_count - processed_count if max_entries else 'unknown'} entries"
            )
            print(f"Loading remaining entries and continuing...\n")

            # Initialize LLM client for masking (needed for resume)
            llm_client = TransformersLLMClient(
                model_name=config.LLM_MODEL_NAME,
                device=config.LLM_DEVICE,
                max_new_tokens=256,
                temperature=0.0,  # Deterministic for consistent masking
            )
            # Update indexer's llm_client and literal_masker
            indexer.llm_client = llm_client
            indexer.literal_masker = LiteralMasker(
                llm_client=llm_client, prompt_manager=indexer.prompt_manager
            )

            # Load remaining entries and append to checkpoint data
            (
                remaining_masked_q,
                remaining_orig_q,
                remaining_masked_sql,
                remaining_orig_sql,
                remaining_metadata,
            ) = indexer.load_dataset(
                dataset_path,
                max_entries=max_entries,
                checkpoint_path=checkpoint_path,
                checkpoint_every=checkpoint_every,
                resume_index=resume_index,
            )

            # Combine checkpoint data with newly loaded data
            masked_q = masked_q + remaining_masked_q
            orig_q = orig_q + remaining_orig_q
            masked_sql = masked_sql + remaining_masked_sql
            orig_sql = orig_sql + remaining_orig_sql
            metadata = metadata + remaining_metadata
        else:
            # Checkpoint is complete or exceeds target - treat as already built
            print(f"\n{'='*60}")
            print(f"CHECKPOINT COMPLETE OR EXCEEDS TARGET")
            print(f"{'='*60}")
            print(f"Checkpoint has {processed_count} entries (target: {target_count})")
            print(f"Treating as already built - skipping index generation.\n")
            # Keep checkpoint folder (do not delete)
            return  # Exit early - index is already built

    if not should_resume:
        # Initialize LLM client for masking
        print("\nInitializing LLM client for masking...")
        llm_client = TransformersLLMClient(
            model_name=config.LLM_MODEL_NAME,
            device=config.LLM_DEVICE,
            max_new_tokens=256,
            temperature=0.0,  # Deterministic for consistent masking
        )

        # Build new masked index with LLM-based masking
        print("\nBuilding masked index with LLM-based masking...")
        print("Note: This will take time as each question is processed by the LLM.")
        print(
            "      Schema will be loaded for each database and cached in schema_cache.json"
        )
        print("      Subsequent runs will be faster as schemas are cached.")
        print("      Checkpoints will be saved every 500 entries for resumability.")

        indexer = MaskedTrainingDatasetIndexer(
            embedding_model_name=config.EMBEDDING_MODEL_NAME,
            index_type=config.FAISS_INDEX_TYPE,
            llm_client=llm_client,
            prompts_dir=prompts_dir,
            databases_path=databases_path,
            use_llm_masking=True,
        )

        if max_entries:
            print(f"Processing {max_entries} entries...")
        else:
            print(f"Processing ALL entries...")

        masked_q, orig_q, masked_sql, orig_sql, metadata = indexer.load_dataset(
            dataset_path,
            max_entries=max_entries,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume_index=0,
        )

    # Build index
    indexer.build_index(
        masked_q,
        orig_q,
        masked_sql,
        orig_sql,
        metadata,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )

    # Save final index
    print(f"\nSaving masked index to: {index_path}")
    indexer.save(index_path)

    # Keep checkpoint folder (do not delete)
    print(f"Checkpoint preserved: {checkpoint_path}")

    print("Masked index built and saved!")


if __name__ == "__main__":
    main()
