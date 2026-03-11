"""
Create a checkpoint from an existing FAISS index.
Use this to resume processing from an existing index.
"""

import os
import sys
import json
import pickle
from pathlib import Path

# Add engine directory to path
engine_dir = Path(__file__).parent
sys.path.insert(0, str(engine_dir))

from config import Config


def create_checkpoint_from_index():
    """Create a checkpoint from existing FAISS index."""
    config = Config()
    
    index_path = config.FAISS_INDEX_MASKED
    checkpoint_path = str(index_path) + ".checkpoint"
    
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        print("Run training_dataset_indexer_masked.py first to create the index.")
        return
    
    print(f"Loading existing index from: {index_path}")
    
    # Load config from index
    with open(os.path.join(index_path, "config.json"), "r") as f:
        index_config = json.load(f)
    
    # Load stores (matching names from training_dataset_indexer_masked.py)
    with open(os.path.join(index_path, "masked_question_store.pkl"), "rb") as f:
        masked_questions = pickle.load(f)
    with open(os.path.join(index_path, "original_question_store.pkl"), "rb") as f:
        original_questions = pickle.load(f)
    with open(os.path.join(index_path, "masked_sql_store.pkl"), "rb") as f:
        masked_sqls = pickle.load(f)
    with open(os.path.join(index_path, "original_sql_store.pkl"), "rb") as f:
        original_sqls = pickle.load(f)
    with open(os.path.join(index_path, "metadata_store.pkl"), "rb") as f:
        metadata = pickle.load(f)

    processed_count = len(masked_questions)
    print(f"Found {processed_count} entries in existing index")

    # Create checkpoint directory
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save checkpoint data (matching names expected by load_checkpoint)
    with open(os.path.join(checkpoint_path, "masked_questions.pkl"), "wb") as f:
        pickle.dump(masked_questions, f)
    with open(os.path.join(checkpoint_path, "original_questions.pkl"), "wb") as f:
        pickle.dump(original_questions, f)
    with open(os.path.join(checkpoint_path, "masked_sqls.pkl"), "wb") as f:
        pickle.dump(masked_sqls, f)
    with open(os.path.join(checkpoint_path, "original_sqls.pkl"), "wb") as f:
        pickle.dump(original_sqls, f)
    with open(os.path.join(checkpoint_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    # Save checkpoint metadata
    checkpoint_data = {
        "processed_count": processed_count,
        "start_index": processed_count,  # Resume from this index
        "embedding_model_name": index_config.get("embedding_model_name", "BAAI/bge-base-en-v1.5"),
        "index_type": index_config.get("index_type", "HNSW"),
    }
    with open(os.path.join(checkpoint_path, "checkpoint.json"), "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"\nCheckpoint created: {checkpoint_path}")
    print(f"Processed count: {processed_count}")
    print(f"Resume index: {processed_count}")
    print(f"\nNow run training_dataset_indexer_masked.py to continue!")
    print(f"Set MAX_ENTRIES= in .env to process all remaining entries.")


if __name__ == "__main__":
    create_checkpoint_from_index()
