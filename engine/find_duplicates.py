"""
Find duplicate questions in the training dataset and compare against the index.
"""
import json
import pickle
import sys
from collections import Counter

from config import Config


def check_duplicates(index_path: str = None):
    config = Config()
    dataset_path = str(config.TRAIN_DATASET)

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = [(i, e) for i, e in enumerate(data) if "question" in e and "SQL" in e]
    print(f"Dataset has {len(entries)} valid entries")

    # Find duplicate questions
    q_counts = Counter(e["question"] for _, e in entries)
    dupes = [(q, count) for q, count in q_counts.items() if count > 1]
    print(f"\nDuplicate questions (same text): {len(dupes)}")
    for q, count in dupes:
        print(f"\n  [{count}x] {q[:120]}")
        # Show each occurrence with its SQL
        for idx, e in entries:
            if e["question"] == q:
                print(f"    Index {idx} | db_id={e.get('db_id','')} | SQL: {e['SQL'][:100]}")

    # Find exact duplicates (same question AND same SQL)
    full_counts = Counter((e["question"], e["SQL"]) for _, e in entries)
    exact_dupes = [(k, count) for k, count in full_counts.items() if count > 1]
    print(f"\nExact duplicates (same question + same SQL): {len(exact_dupes)}")
    for (q, sql), count in exact_dupes:
        print(f"  [{count}x] Q: {q[:80]}")
        print(f"         SQL: {sql[:80]}")

    # Compare against index if path provided
    if index_path:
        print(f"\n--- Comparing against index at: {index_path} ---")
        with open(f"{index_path}/original_question_store.pkl", "rb") as f:
            indexed_orig = pickle.load(f)
        with open(f"{index_path}/masked_question_store.pkl", "rb") as f:
            indexed_masked = pickle.load(f)

        print(f"Index has {len(indexed_orig)} entries")
        print(f"Dataset has {len(entries)} entries")
        print(f"Difference: {len(entries) - len(indexed_orig)}")

        # Check for masked duplicates that got deduped
        masked_counts = Counter(zip(indexed_masked, indexed_orig))
        index_dupes = [(k, c) for k, c in masked_counts.items() if c > 1]
        print(f"Duplicates in index: {len(index_dupes)}")


if __name__ == "__main__":
    idx_path = sys.argv[1] if len(sys.argv) > 1 else None
    check_duplicates(idx_path)
