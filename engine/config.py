"""
Configuration Module for MCS-SQL

This module loads configuration from .env file and provides
centralized access to all paths and settings.

Usage:
    from config import Config
    
    config = Config()
    print(config.TRAIN_DATASET)
    print(config.FAISS_INDEX)
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """
    Centralized configuration manager for MCS-SQL.
    
    Loads environment variables from .env file and provides
    typed access to all configuration values.
    """

    def __init__(self, env_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env_path: Path to .env file. If None, searches for .env in
                     project root and current directory.
        """
        # Find and load .env file
        if env_path:
            load_dotenv(env_path)
        else:
            # Try to find .env in project root
            project_root = Path(__file__).parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
            else:
                # Try current directory
                load_dotenv()

        # Base paths
        self.PROJECT_ROOT = self._get_path("PROJECT_ROOT", Path(__file__).parent.parent)

        # Dataset paths
        self.TRAIN_DATASET = self._get_path("TRAIN_DATASET", self.PROJECT_ROOT / "train" / "train" / "train.json")
        self.DEV_DATABASES = self._get_path("DEV_DATABASES", self.PROJECT_ROOT / "minidev")

        # FAISS index paths
        self.FAISS_INDEX = self._get_path("FAISS_INDEX", self.PROJECT_ROOT / "faiss-index")
        self.FAISS_INDEX_MASKED = self._get_path("FAISS_INDEX_MASKED", self.PROJECT_ROOT / "faiss-index-masked")

        # Model configuration
        self.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
        self.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

        # Prompts path
        self.PROMPTS_DIR = self._get_path("PROMPTS_DIR", self.PROJECT_ROOT / "prompts")

        # Processing configuration
        self.EMBEDDING_BATCH_SIZE = self._get_int("EMBEDDING_BATCH_SIZE", 1000)
        self.INDEX_BATCH_SIZE = self._get_int("INDEX_BATCH_SIZE", 10000)
        self.MAX_ENTRIES = self._get_optional_int("MAX_ENTRIES")

        # LLM configuration
        self.LLM_DEVICE = os.getenv("LLM_DEVICE", "cuda")
        self.LLM_MAX_NEW_TOKENS = self._get_int("LLM_MAX_NEW_TOKENS", 512)
        self.LLM_TEMPERATURE = self._get_float("LLM_TEMPERATURE", 0.7)
        self.TABLE_LINKING_ITERATIONS = self._get_int("TABLE_LINKING_ITERATIONS", 3)
        self.COLUMN_LINKING_ITERATIONS = self._get_int("COLUMN_LINKING_ITERATIONS", 3)
        self.MAJORITY_VOTE_N = self._get_int("MAJORITY_VOTE_N", 20)

        # FAISS configuration
        self.FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "HNSW")

        # Database configuration
        self.DEFAULT_DATABASE = os.getenv("DEFAULT_DATABASE", "california_schools/california_schools.sqlite")

        # Create necessary directories
        self._ensure_directories()

    def _get_path(self, key: str, default: Path) -> Path:
        """Get a path value from environment or use default."""
        value = os.getenv(key)
        if value:
            return Path(value)
        return default

    def _get_int(self, key: str, default: int) -> int:
        """Get an integer value from environment or use default."""
        value = os.getenv(key)
        if value:
            try:
                return int(value)
            except ValueError:
                return default
        return default

    def _get_float(self, key: str, default: float) -> float:
        """Get a float value from environment or use default."""
        value = os.getenv(key)
        if value:
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _get_optional_int(self, key: str) -> Optional[int]:
        """Get an optional integer value from environment."""
        value = os.getenv(key)
        if value:
            try:
                return int(value)
            except ValueError:
                return None
        return None

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        # No auto-creation needed - all directories are at PROJECT_ROOT level
        pass

    def get_database_path(self, db_name: Optional[str] = None) -> Path:
        """
        Get the full path to a database.

        Args:
            db_name: Database name (e.g., 'california_schools/california_schools.sqlite').
                    If None, uses DEFAULT_DATABASE.

        Returns:
            Full path to the database file
        """
        if db_name is None:
            db_name = self.DEFAULT_DATABASE
        return self.DEV_DATABASES / db_name

    def to_dict(self) -> dict:
        """Return configuration as a dictionary."""
        return {
            "PROJECT_ROOT": str(self.PROJECT_ROOT),
            "TRAIN_DATASET": str(self.TRAIN_DATASET),
            "DEV_DATABASES": str(self.DEV_DATABASES),
            "FAISS_INDEX": str(self.FAISS_INDEX),
            "FAISS_INDEX_MASKED": str(self.FAISS_INDEX_MASKED),
            "EMBEDDING_MODEL_NAME": self.EMBEDDING_MODEL_NAME,
            "LLM_MODEL_NAME": self.LLM_MODEL_NAME,
            "PROMPTS_DIR": str(self.PROMPTS_DIR),
            "EMBEDDING_BATCH_SIZE": self.EMBEDDING_BATCH_SIZE,
            "INDEX_BATCH_SIZE": self.INDEX_BATCH_SIZE,
            "MAX_ENTRIES": self.MAX_ENTRIES,
            "LLM_DEVICE": self.LLM_DEVICE,
            "LLM_MAX_NEW_TOKENS": self.LLM_MAX_NEW_TOKENS,
            "LLM_TEMPERATURE": self.LLM_TEMPERATURE,
            "TABLE_LINKING_ITERATIONS": self.TABLE_LINKING_ITERATIONS,
            "COLUMN_LINKING_ITERATIONS": self.COLUMN_LINKING_ITERATIONS,
            "MAJORITY_VOTE_N": self.MAJORITY_VOTE_N,
            "FAISS_INDEX_TYPE": self.FAISS_INDEX_TYPE,
            "DEFAULT_DATABASE": self.DEFAULT_DATABASE,
        }

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


# Example usage
if __name__ == "__main__":
    config = Config()
    print("MCS-SQL Configuration:")
    print("-" * 50)
    for key, value in config.to_dict().items():
        print(f"{key}: {value}")
