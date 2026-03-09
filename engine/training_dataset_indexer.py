"""
Training Dataset Indexer for Text-to-SQL

This script loads a large training dataset (JSON array) with fields:
- db_id, question, evidence, SQL

It extracts question and SQL, creates FAISS embeddings for questions,
and stores SQL as associated values for semantic search.
"""

import json
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple, Generator
import faiss
import numpy as np
from tqdm import tqdm


class TrainingDatasetIndexer:
    """
    Indexer for training dataset using FAISS for semantic search.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "Flat",
    ):
        self.embedding_model_name = embedding_model_name
        self.index_type = index_type
        self.embedding_model: Optional[Any] = None
        self.index: Optional[faiss.Index] = None
        self.question_store: List[str] = []  # Store questions for retrieval
        self.sql_store: List[str] = []
        self.metadata_store: List[Dict[str, Any]] = []
        self.dimension: int = 0

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

    def stream_json_array(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream JSON array items one by one using ijson.
        Memory efficient for very large files.
        """
        try:
            import ijson
        except ImportError:
            raise ImportError("Install ijson: pip install ijson")

        print(f"Streaming JSON array from: {file_path}")
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        print(f"File size: {file_size_gb:.2f} GB")

        with open(file_path, "rb") as f:
            for item in ijson.items(f, "item"):
                yield item

    def load_dataset(
        self,
        dataset_path: str,
        max_entries: Optional[int] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load dataset from JSON file into memory.
        Only use for files < 2GB. For larger files, use build_index_streaming().

        Args:
            dataset_path: Path to JSON array file
            max_entries: Maximum entries to load

        Returns:
            Tuple of (questions list, metadata list)
        """
        questions = []
        metadata = []

        print(f"Loading dataset from: {dataset_path}")
        file_size_gb = os.path.getsize(dataset_path) / (1024**3)
        print(f"File size: {file_size_gb:.2f} GB")

        if file_size_gb > 2:
            print("WARNING: Large file. Consider using build_index_streaming() instead.")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for entry in tqdm(data[:max_entries], desc="Extracting"):
                    if "question" in entry and "SQL" in entry:
                        questions.append(entry["question"])
                        metadata.append({
                            "db_id": entry.get("db_id", ""),
                            "evidence": entry.get("evidence", ""),
                            "sql": entry["SQL"],
                            "difficulty": entry.get("difficulty", "unknown"),
                        })

        print(f"Extracted {len(questions)} question-SQL pairs")
        return questions, metadata

    def build_index_streaming(
        self,
        dataset_path: str,
        embedding_batch_size: int = 1000,
        index_batch_size: int = 10000,
        max_entries: Optional[int] = None,
    ):
        """
        Build index by streaming the dataset - memory efficient.

        Args:
            dataset_path: Path to JSON array file
            embedding_batch_size: Batch size for embeddings
            index_batch_size: Add to index after this many entries
            max_entries: Maximum entries to process
        """
        if self.embedding_model is None:
            self.load_model()
        if self.index is None:
            self.create_index()

        questions_batch = []
        metadata_batch = []
        count = 0

        print("Building index from streaming data...")

        for entry in self.stream_json_array(dataset_path):
            if max_entries and count >= max_entries:
                break

            if "question" in entry and "SQL" in entry:
                questions_batch.append(entry["question"])
                metadata_batch.append({
                    "db_id": entry.get("db_id", ""),
                    "evidence": entry.get("evidence", ""),
                    "sql": entry["SQL"],
                    "difficulty": entry.get("difficulty", "unknown"),
                })
                count += 1

            # Process batch
            if len(questions_batch) >= index_batch_size:
                self._process_batch(questions_batch, metadata_batch, embedding_batch_size)
                questions_batch = []
                metadata_batch = []

                if count % 100000 == 0:
                    print(f"Processed {count} entries, index size: {self.index.ntotal}")

        # Process remaining
        if questions_batch:
            self._process_batch(questions_batch, metadata_batch, embedding_batch_size)

        print(f"Index built with {self.index.ntotal} entries from {count} total")

    def _process_batch(
        self,
        questions: List[str],
        metadata: List[Dict[str, Any]],
        batch_size: int,
    ):
        """Process a batch of questions."""
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            questions,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Add to index
        self.index.add(embeddings)

        # Store questions, SQL and metadata
        self.question_store.extend(questions)
        self.sql_store.extend([m["sql"] for m in metadata])
        self.metadata_store.extend(metadata)

    def build_index(
        self,
        questions: List[str],
        metadata: List[Dict[str, Any]],
        batch_size: int = 10000,
    ):
        """Build FAISS index from loaded questions (loads all in memory)."""
        if self.embedding_model is None:
            self.load_model()
        if self.index is None:
            self.create_index()

        print(f"Building index for {len(questions)} questions...")

        if self.index_type == "IVF":
            print("Training IVF index...")
            sample_size = min(10000, len(questions))
            sample_embeddings = self.embedding_model.encode(
                questions[:sample_size],
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            sample_embeddings = np.asarray(sample_embeddings, dtype=np.float32)
            self.index.train(sample_embeddings)

        self.question_store = []
        self.sql_store = []
        self.metadata_store = []

        for i in tqdm(range(0, len(questions), batch_size), desc="Indexing"):
            batch_questions = questions[i : i + batch_size]
            batch_metadata = metadata[i : i + batch_size]

            embeddings = self.embedding_model.encode(
                batch_questions,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            self.index.add(embeddings)
            self.question_store.extend(batch_questions)
            self.sql_store.extend([m["sql"] for m in batch_metadata])
            self.metadata_store.extend(batch_metadata)

        print(f"Index built with {self.index.ntotal} entries")

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """
        Search for similar questions.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (question, SQL, metadata, similarity_score) tuples
        """
        if self.index is None or self.embedding_model is None:
            raise ValueError("Index not built.")

        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        print(f"  Debug: Index has {self.index.ntotal} entries")
        print(f"  Debug: Query embedding shape: {query_embedding.shape}")
        
        k = min(top_k, self.index.ntotal)
        print(f"  Debug: Searching for top {k} results")
        
        scores, indices = self.index.search(query_embedding, k)
        
        print(f"  Debug: Scores shape: {scores.shape}, Indices shape: {indices.shape}")
        print(f"  Debug: Scores: {scores}")
        print(f"  Debug: Indices: {indices}")

        results = []
        for score, idx in zip(scores[0], indices[0]):
            print(f"  Debug: Checking idx={idx}, score={score}")
            if idx >= 0 and idx < len(self.question_store):
                results.append((
                    self.question_store[idx],
                    self.sql_store[idx],
                    self.metadata_store[idx],
                    float(score)
                ))
        return results

    def save(self, save_path: str):
        """Save index and data to disk."""
        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_path, "faiss_index.bin"))
        with open(os.path.join(save_path, "question_store.pkl"), "wb") as f:
            pickle.dump(self.question_store, f)
        with open(os.path.join(save_path, "sql_store.pkl"), "wb") as f:
            pickle.dump(self.sql_store, f)
        with open(os.path.join(save_path, "metadata_store.pkl"), "wb") as f:
            pickle.dump(self.metadata_store, f)
        config = {
            "embedding_model_name": self.embedding_model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Index saved to: {save_path}")

    def load(self, load_path: str):
        """Load index and data from disk."""
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)
        self.embedding_model_name = config["embedding_model_name"]
        self.index_type = config["index_type"]
        self.dimension = config["dimension"]
        self.load_model()
        self.index = faiss.read_index(os.path.join(load_path, "faiss_index.bin"))
        with open(os.path.join(load_path, "question_store.pkl"), "rb") as f:
            self.question_store = pickle.load(f)
        with open(os.path.join(load_path, "sql_store.pkl"), "rb") as f:
            self.sql_store = pickle.load(f)
        with open(os.path.join(load_path, "metadata_store.pkl"), "rb") as f:
            self.metadata_store = pickle.load(f)
        print(f"Index loaded from: {load_path}")
        print(f"Total entries: {len(self.question_store)}")


def main():
    """Main function - builds or loads index."""
    # Default paths
    dataset_path = r"/project/ss797/at978/MCS-SQL/MCS-SQL/engine/train/train/train.json"
    index_path = r"/project/ss797/at978/MCS-SQL/MCS-SQL/engine/faiss-index"
    
    # Check if index already exists
    if os.path.exists(index_path):
        # Load existing index
        indexer = TrainingDatasetIndexer()
        print(f"Loading existing index from: {index_path}")
        indexer.load(index_path)
        print(f"Index loaded with {len(indexer.question_store)} entries")
    else:
        # Build new index
        indexer = TrainingDatasetIndexer(
            embedding_model_name="BAAI/bge-base-en-v1.5",
            index_type="HNSW",
        )

        print(f"Loading dataset from: {dataset_path}")
        questions, metadata = indexer.load_dataset(dataset_path)
        indexer.build_index(questions, metadata, batch_size=1000)

        print(f"Saving index to: {index_path}")
        indexer.save(index_path)
        print("Index built and saved!")


if __name__ == "__main__":
    main()