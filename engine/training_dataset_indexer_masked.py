"""
Masked Training Dataset Indexer for Text-to-SQL

This script builds a FAISS index using MASKED questions where literals
(numbers, strings, dates, etc.) are replaced with placeholders.

This helps generalize better - semantically similar questions with different
values will map to the same embedding.

Example:
  Original: "What is the ratio of customers who pay in EUR against CZK?"
  Masked:   "What is the ratio of customers who pay in [CURRENCY] against [CURRENCY]?"

This version uses a transformer-based LLM (Qwen) to identify and replace
literals instead of regex-based masking, providing better semantic understanding.
"""

import json
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple, Generator
import faiss
import numpy as np
from tqdm import tqdm

from literal_masker import LiteralMasker
from prompt_manager import PromptManager


class MaskedTrainingDatasetIndexer:
    """
    Indexer for training dataset using masked questions for better generalization.
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        index_type: str = "HNSW",
        llm_client: Optional[Any] = None,
        prompts_dir: str = None,
    ):
        """
        Initialize the masked training dataset indexer.

        Args:
            embedding_model_name: Name of the sentence transformer model for embeddings
            index_type: FAISS index type (Flat, IVF, HNSW)
            llm_client: Optional LLM client for transformer-based masking.
                       If None, falls back to regex-based masking.
            prompts_dir: Path to the prompts directory for loading prompt templates.
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
        
        # Initialize prompt manager if prompts directory is provided
        self.prompt_manager = None
        if prompts_dir and os.path.exists(prompts_dir):
            self.prompt_manager = PromptManager(prompts_dir)
            print(f"Loaded prompt templates from: {prompts_dir}")
        
        self.literal_masker = LiteralMasker(
            llm_client=llm_client,
            prompt_manager=self.prompt_manager
        )

    def load_model(self):
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")

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
    ) -> Tuple[List[str], List[str], List[str], List[str], List[Dict[str, Any]]]:
        """
        Load dataset and mask questions/SQL.

        Args:
            dataset_path: Path to JSON array file
            max_entries: Maximum entries to load

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

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for entry in tqdm(data[:max_entries], desc="Loading & Masking"):
                    if "question" in entry and "SQL" in entry:
                        orig_question = entry["question"]
                        orig_sql = entry["SQL"]

                        masked_questions.append(self.literal_masker.mask_question(orig_question))
                        original_questions.append(orig_question)
                        masked_sqls.append(self.literal_masker.mask_sql(orig_sql))
                        original_sqls.append(orig_sql)
                        metadata.append({
                            "db_id": entry.get("db_id", ""),
                            "evidence": entry.get("evidence", ""),
                            "original_sql": orig_sql,
                            "masked_sql": masked_sqls[-1],
                            "difficulty": entry.get("difficulty", "unknown"),
                        })

        print(f"Extracted and masked {len(masked_questions)} question-SQL pairs")
        return masked_questions, original_questions, masked_sqls, original_sqls, metadata

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

        # Mask the query using the transformer model
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
                results.append((
                    self.masked_question_store[idx],
                    self.original_question_store[idx],
                    self.masked_sql_store[idx],
                    self.original_sql_store[idx],
                    self.metadata_store[idx],
                    float(score)
                ))
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
    """Main function - builds or loads masked index."""
    dataset_path = r"/project/ss797/at978/MCS-SQL/MCS-SQL/engine/train/train/train.json"
    index_path = r"/project/ss797/at978/MCS-SQL/MCS-SQL/engine/faiss-index-masked"
    prompts_dir = r"C:\Repos\MCS-SQL\prompts"

    if os.path.exists(index_path):
        # Load existing masked index
        indexer = MaskedTrainingDatasetIndexer()
        print(f"Loading existing masked index from: {index_path}")
        indexer.load(index_path)
        print(f"Index loaded with {len(indexer.masked_question_store)} entries")
    else:
        # Initialize Qwen LLM client for transformer-based masking
        from schema_linking import TransformersLLMClient

        print("Initializing transformer model for literal masking...")
        llm_client = TransformersLLMClient(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            device="cuda",  # Change to "cpu" if no GPU available
            max_new_tokens=256,
            temperature=0.0,  # Use deterministic output for masking
        )

        # Build new masked index with transformer-based masking
        indexer = MaskedTrainingDatasetIndexer(
            embedding_model_name="BAAI/bge-base-en-v1.5",
            index_type="HNSW",
            llm_client=llm_client,
            prompts_dir=prompts_dir,
        )

        masked_q, orig_q, masked_sql, orig_sql, metadata = indexer.load_dataset(
            dataset_path, max_entries=100  # Limit for testing, remove for full dataset
        )
        indexer.build_index(masked_q, orig_q, masked_sql, orig_sql, metadata, batch_size=1000)

        print(f"Saving masked index to: {index_path}")
        indexer.save(index_path)
        print("Masked index built and saved!")


if __name__ == "__main__":
    main()
