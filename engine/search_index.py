"""
Search Script for Masked Training Dataset Indexer

Quick script to search the FAISS index for similar questions.
"""

import os
import sys
from pathlib import Path

# Add engine directory to path
engine_dir = Path(__file__).parent
sys.path.insert(0, str(engine_dir))

from config import Config
from training_dataset_indexer_masked import MaskedTrainingDatasetIndexer


def main():
    """Interactive search script."""
    # Load configuration
    config = Config()
    
    # Determine which index to use
    index_path = config.FAISS_INDEX_MASKED
    
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        print("Run training_dataset_indexer_masked.py first to build the index.")
        return
    
    # Load the indexer
    print(f"Loading masked index from: {index_path}")
    indexer = MaskedTrainingDatasetIndexer(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        index_type=config.FAISS_INDEX_TYPE,
    )
    indexer.load(index_path)
    
    print(f"\nIndex loaded with {len(indexer.masked_question_store)} entries")
    print("Enter questions to search (or 'quit' to exit)\n")
    
    while True:
        try:
            # Get query from user
            query = input("Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Search
            print("\nSearching...")
            results = indexer.search(query, top_k=5)
            
            if not results:
                print("No results found.\n")
                continue
            
            # Display results
            print(f"\nFound {len(results)} similar questions:\n")
            print("=" * 80)
            
            for i, (masked_q, orig_q, masked_sql, orig_sql, metadata, score) in enumerate(results, 1):
                print(f"\n{i}. Score: {score:.4f}")
                print(f"   Question: {orig_q}")
                print(f"   SQL: {orig_sql[:200]}{'...' if len(orig_sql) > 200 else ''}")
                print(f"   Database: {metadata.get('db_id', 'unknown')}")
                if metadata.get('evidence'):
                    print(f"   Evidence: {metadata['evidence'][:150]}{'...' if len(metadata['evidence']) > 150 else ''}")
                print("-" * 80)
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
