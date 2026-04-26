"""
JSON Schema Definitions for Outlines Library

This module defines JSON schemas for use with the outlines library
for forced structured output.
"""

from typing import Dict, Any

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

# Export schema dictionaries for use in other modules
TABLE_LINKING_SCHEMA = TABLE_LINKING_SCHEMA_DICT
COLUMN_LINKING_SCHEMA = COLUMN_LINKING_SCHEMA_DICT
SQL_GENERATION_SCHEMA = SQL_GENERATION_SCHEMA_DICT
SQL_SELECTION_SCHEMA = SQL_SELECTION_SCHEMA_DICT

# Dictionary mapping task names to their schema dictionaries
SCHEMA_DICTS = {
    "table_linking": TABLE_LINKING_SCHEMA_DICT,
    "column_linking": COLUMN_LINKING_SCHEMA_DICT,
    "sql_generation": SQL_GENERATION_SCHEMA_DICT,
    "sql_selection": SQL_SELECTION_SCHEMA_DICT,
    "question_masking": QUESTION_MASKING_SCHEMA_DICT,
    "sql_masking": SQL_MASKING_SCHEMA_DICT,
}

def get_schema_for_task(task_name: str) -> Dict[str, Any]:
    """
    Get JSON schema dictionary for a specific task.

    Args:
        task_name: Name of the task (e.g., 'table_linking', 'column_linking',
                   'sql_generation', 'sql_selection', 'question_masking', 'sql_masking')

    Returns:
        JSON schema dictionary for the task
    """
    if task_name not in SCHEMA_DICTS:
        raise ValueError(
            f"Unknown task: {task_name}. Available tasks: {list(SCHEMA_DICTS.keys())}"
        )

    return SCHEMA_DICTS[task_name]
