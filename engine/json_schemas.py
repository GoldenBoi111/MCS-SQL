"""
JSON Schema Definitions for Outlines Library

This module defines Pydantic models for use with the outlines library
for forced structured output, using outlines utilities to convert JSON schemas.
"""

from typing import Dict, Any
from pydantic import BaseModel

try:
    from outlines.types.json_schema_utils import json_schema_dict_to_pydantic
except ImportError:
    # Fallback if outlines is not available
    def json_schema_dict_to_pydantic(schema: Dict[str, Any], name: str = None) -> type:
        """Dummy function if outlines is not available."""

        # Return a basic BaseModel for compatibility
        class DummyModel(BaseModel):
            pass

        return DummyModel


# Schema for table linking task (from table_linking.txt)
TABLE_LINKING_SCHEMA_DICT: Dict[str, Any] = {
    "title": "TableLinkingResult",
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "The reason for choosing each table",
        },
        "tables": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "List of selected tables",
        },
    },
    "required": ["reasoning", "tables"],
    "additionalProperties": False,
}

# Convert to Pydantic model
TableLinkingResult = json_schema_dict_to_pydantic(
    TABLE_LINKING_SCHEMA_DICT, "TableLinkingResult"
)


# Schema for column linking task (from column_linking.txt)
COLUMN_LINKING_SCHEMA_DICT: Dict[str, Any] = {
    "title": "ColumnLinkingResult",
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "The reason for choosing each column",
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "List of selected columns in format table_name.column_name",
        },
    },
    "required": ["reasoning", "columns"],
    "additionalProperties": False,
}

# Convert to Pydantic model
ColumnLinkingResult = json_schema_dict_to_pydantic(
    COLUMN_LINKING_SCHEMA_DICT, "ColumnLinkingResult"
)


# Schema for SQL generation task (from SQL_generation.txt)
SQL_GENERATION_SCHEMA_DICT: Dict[str, Any] = {
    "title": "SQLGenerationResult",
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "The reasoning steps for generating SQL",
        },
        "sql": {"type": "string", "description": "The final generated SQL query"},
    },
    "required": ["reasoning", "sql"],
    "additionalProperties": False,
}

# Convert to Pydantic model
SQLGenerationResult = json_schema_dict_to_pydantic(
    SQL_GENERATION_SCHEMA_DICT, "SQLGenerationResult"
)


# Schema for SQL selection task (from SQL_selection.txt)
SQL_SELECTION_SCHEMA_DICT: Dict[str, Any] = {
    "title": "SQLSelectionResult",
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "The reasoning steps for choosing the best SQL",
        },
        "sql": {"type": "string", "description": "The final chosen SQL query"},
    },
    "required": ["reasoning", "sql"],
    "additionalProperties": False,
}

# Convert to Pydantic model
SQLSelectionResult = json_schema_dict_to_pydantic(
    SQL_SELECTION_SCHEMA_DICT, "SQLSelectionResult"
)


# Schema for question masking task (from question_masking.txt)
QUESTION_MASKING_SCHEMA_DICT: Dict[str, Any] = {
    "title": "QuestionMaskingResult",
    "type": "object",
    "properties": {
        "masked_question": {
            "type": "string",
            "description": "The question with table names, column names, and values replaced by placeholders",
        }
    },
    "required": ["masked_question"],
    "additionalProperties": False,
}

# Convert to Pydantic model
QuestionMaskingResult = json_schema_dict_to_pydantic(
    QUESTION_MASKING_SCHEMA_DICT, "QuestionMaskingResult"
)


# Schema for SQL masking task
SQL_MASKING_SCHEMA_DICT: Dict[str, Any] = {
    "title": "SQLMaskingResult",
    "type": "object",
    "properties": {
        "masked_text": {
            "type": "string",
            "description": "The SQL query with literals replaced by placeholders",
        }
    },
    "required": ["masked_text"],
    "additionalProperties": False,
}

# Convert to Pydantic model
SQLMaskingResult = json_schema_dict_to_pydantic(
    SQL_MASKING_SCHEMA_DICT, "SQLMaskingResult"
)


# Dictionary mapping task names to their Pydantic models
SCHEMA_MODELS = {
    "table_linking": TableLinkingResult,
    "column_linking": ColumnLinkingResult,
    "sql_generation": SQLGenerationResult,
    "sql_selection": SQLSelectionResult,
    "question_masking": QuestionMaskingResult,
    "sql_masking": SQLMaskingResult,
}


def get_schema_for_task(task_name: str) -> type:
    """
    Get Pydantic model for a specific task.

    Args:
        task_name: Name of the task (e.g., 'table_linking', 'column_linking',
                   'sql_generation', 'sql_selection', 'question_masking', 'sql_masking')

    Returns:
        Pydantic model for the task
    """
    if task_name not in SCHEMA_MODELS:
        raise ValueError(
            f"Unknown task: {task_name}. Available tasks: {list(SCHEMA_MODELS.keys())}"
        )

    return SCHEMA_MODELS[task_name]
