import json
import os
import pickle
import sys

from config import Config
from training_dataset_indexer_masked import MaskedTrainingDatasetIndexer


def load_pkl_from_checkpoint(checkpoint_path: str):
    """Load all .pkl data from a single checkpoint directory."""
    print(f"  Loading from: {checkpoint_path}")
    
    with open(os.path.join(checkpoint_path, "masked_questions.pkl"), "rb") as f:
        masked_q = pickle.load(f)
    with open(os.path.join(checkpoint_path, "original_questions.pkl"), "rb") as f:
        orig_q = pickle.load(f)
    with open(os.path.join(checkpoint_path, "masked_sqls.pkl"), "rb") as f:
        masked_sql = pickle.load(f)
    with open(os.path.join(checkpoint_path, "original_sqls.pkl"), "rb") as f:
        orig_sql = pickle.load(f)
    with open(os.path.join(checkpoint_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    print(f"  Loaded {len(masked_q)} items")
    return masked_q, orig_q, masked_sql, orig_sql, metadata


def rebuild_from_checkpoints(checkpoint_paths: list, output_index_path: str):
    """Merge multiple checkpoint directories into one FAISS index."""
    config = Config()
    
    # Accumulate all items from all checkpoints
    all_masked_q, all_orig_q, all_masked_sql, all_orig_sql, all_metadata = [], [], [], [], []
    
    print(f"Merging {len(checkpoint_paths)} checkpoint(s)...\n")
    for chk_path in checkpoint_paths:
        masked_q, orig_q, masked_sql, orig_sql, metadata = load_pkl_from_checkpoint(chk_path)
        all_masked_q.extend(masked_q)
        all_orig_q.extend(orig_q)
        all_masked_sql.extend(masked_sql)
        all_orig_sql.extend(orig_sql)
        all_metadata.extend(metadata)
    
    print(f"\nTotal items loaded: {len(all_masked_q)}")
    
    # Deduplicate based on (masked_question, original_sql)
    seen = set()
    dedup_mq, dedup_oq, dedup_ms, dedup_os, dedup_md = [], [], [], [], []
    for mq, oq, ms, os_, md in zip(all_masked_q, all_orig_q, all_masked_sql, all_orig_sql, all_metadata):
        key = (mq, os_)
        if key not in seen:
            seen.add(key)
            dedup_mq.append(mq)
            dedup_oq.append(oq)
            dedup_ms.append(ms)
            dedup_os.append(os_)
            dedup_md.append(md)
    
    num_dupes = len(all_masked_q) - len(dedup_mq)
    print(f"Removed {num_dupes} duplicates.")
    print(f"Building index with {len(dedup_mq)} unique items.\n")
    
    # Create indexer and build
    indexer = MaskedTrainingDatasetIndexer(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        index_type=config.FAISS_INDEX_TYPE,
        use_llm_masking=False,
    )
    
    indexer.build_index(
        dedup_mq, dedup_oq, dedup_ms, dedup_os, dedup_md,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    
    print(f"\nSaving finalized index to: {output_index_path}")
    indexer.save(output_index_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <output_dir> <checkpoint_dir1> [checkpoint_dir2] ...")
        print(f"Example: python {sys.argv[0]} ./output ./checkpoint1 ./checkpoint2")
        sys.exit(1)
    
    output_path = sys.argv[1]
    checkpoint_dirs = sys.argv[2:]
    rebuild_from_checkpoints(checkpoint_dirs, output_path)
