# MCS-SQL Configuration Guide

## Overview

The MCS-SQL project uses a centralized configuration system via `.env` files and a `Config` class to manage all paths and settings.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Paths

Copy the example environment file and update paths for your system:

```bash
cp .env.example .env
```

Edit `.env` and update the following key paths:

```ini
# Base paths
PROJECT_ROOT=C:/Repos/MCS-SQL
DATA_BASE=C:/Repos/MCS-SQL/data

# Dataset paths (update to match your actual data locations)
TRAIN_DATASET=/path/to/your/train.json
DEV_DATABASES=/path/to/your/databases

# FAISS indices
FAISS_INDEX=/path/to/faiss-index
FAISS_INDEX_MASKED=/path/to/faiss-index-masked
```

### 3. Verify Configuration

```bash
cd engine
python config.py
```

This will display all configuration values.

## Configuration Options

### Base Paths

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ROOT` | Root directory of MCS-SQL project | Auto-detected |
| `DATA_BASE` | Base directory for all data files | `${PROJECT_ROOT}/data` |

### Dataset Paths

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_DATASET` | Training dataset JSON file | `${DATA_BASE}/train/train/train.json` |
| `DEV_DATABASES` | Development databases directory | `${DATA_BASE}/dev_databases` |

### FAISS Index Paths

| Variable | Description | Default |
|----------|-------------|---------|
| `FAISS_INDEX` | Standard (unmasked) questions index | `${DATA_BASE}/faiss-index` |
| `FAISS_INDEX_MASKED` | Masked questions index | `${DATA_BASE}/faiss-index-masked` |

### Model Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL_NAME` | Sentence transformer model | `BAAI/bge-base-en-v1.5` |
| `LLM_MODEL_NAME` | LLM for text generation | `Qwen/Qwen2.5-7B-Instruct` |
| `MODELS_DIR` | Directory for downloaded models | `${DATA_BASE}/models` |

### LLM Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_DEVICE` | Device for inference (`cuda` or `cpu`) | `cuda` |
| `LLM_MAX_NEW_TOKENS` | Maximum tokens to generate | `512` |
| `LLM_TEMPERATURE` | Sampling temperature (0.0 = deterministic) | `0.7` |
| `TABLE_LINKING_ITERATIONS` | Shuffle iterations for table linking | `3` |
| `COLUMN_LINKING_ITERATIONS` | Shuffle iterations for column linking | `3` |
| `MAJORITY_VOTE_N` | Parallel outputs for majority voting | `20` |

### Processing Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_BATCH_SIZE` | Batch size for embeddings | `1000` |
| `INDEX_BATCH_SIZE` | Batch size for FAISS indexing | `10000` |
| `MAX_ENTRIES` | Max entries to process (empty = all) | Empty |

### FAISS Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `FAISS_INDEX_TYPE` | Index type (`Flat`, `IVF`, `HNSW`) | `HNSW` |

## Using Configuration in Code

### Basic Usage

```python
from config import Config

config = Config()

# Access paths
print(config.TRAIN_DATASET)
print(config.FAISS_INDEX)

# Access settings
print(config.LLM_DEVICE)
print(config.EMBEDDING_BATCH_SIZE)

# Get database path
db_path = config.get_database_path("my_database/my_database.sqlite")
```

### Using Global Config

```python
from config import get_config

config = get_config()
print(config.PROMPTS_DIR)
```

### In Scripts

```python
from config import Config

def main():
    config = Config()
    
    indexer = MaskedTrainingDatasetIndexer(
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        index_type=config.FAISS_INDEX_TYPE,
        prompts_dir=config.PROMPTS_DIR,
    )
    
    # ... rest of your code
```

## Directory Structure

```
MCS-SQL/
├── .env                  # Your local configuration (not in git)
├── .env.example          # Example configuration (committed)
├── requirements.txt      # Python dependencies
├── prompts/              # Prompt templates
│   ├── question_masking.txt
│   ├── table_linking.txt
│   ├── column_linking.txt
│   ├── SQL_generation.txt
│   └── SQL_selection.txt
└── engine/
    ├── config.py         # Configuration module
    ├── training_dataset_indexer.py
    ├── training_dataset_indexer_masked.py
    ├── schema_linking.py
    ├── literal_masker.py
    └── prompt_manager.py
```

## Environment-Specific Configuration

You can create different `.env` files for different environments:

```bash
# Development
cp .env.example .env.dev

# Production
cp .env.example .env.prod
```

Then load a specific config:

```python
config = Config(env_path=".env.prod")
```

## Troubleshooting

### Configuration Not Loading

1. Ensure `.env` file exists in the project root
2. Check file permissions
3. Verify path format (use forward slashes `/` or escaped backslashes `\\`)

### Paths Not Found

1. Run `python config.py` to see resolved paths
2. Ensure directories exist or will be created automatically
3. Check that environment variables are properly formatted

### Model Loading Issues

1. Verify `LLM_DEVICE` matches your system (`cuda` for GPU, `cpu` for CPU-only)
2. Check that `MODELS_DIR` has sufficient disk space
3. Ensure internet connection for initial model download

## Migration from Hardcoded Paths

If you have scripts with hardcoded paths, update them to use config:

**Before:**
```python
dataset_path = r"/project/ss797/at978/MCS-SQL/MCS-SQL/engine/train/train/train.json"
index_path = r"/project/ss797/at978/MCS-SQL/MCS-SQL/engine/faiss-index"
```

**After:**
```python
from config import Config

config = Config()
dataset_path = config.TRAIN_DATASET
index_path = config.FAISS_INDEX
```
