"""
Literal Masker using Transformer Models

This module uses a transformer-based LLM to identify and replace literals
in natural language questions and SQL queries with semantic placeholders.

This provides better generalization than regex-based masking by understanding
context and semantics.

Uses outlines library for forced JSON output to ensure structured responses.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

try:
    import outlines
    from outlines.models import Transformers as OutlinesTransformers
    from outlines.types.json_schema_utils import json_schema_dict_to_pydantic

    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    outlines = None

from json_schemas import get_schema_for_task


class LiteralMasker:
    """
    Uses a transformer model to mask literals in text and SQL.
    """

    def __init__(
        self, llm_client: Optional[Any] = None, prompt_manager: Optional[Any] = None
    ):
        """
        Initialize the literal masker.

        Args:
            llm_client: LLM client instance (must have a `generate` method).
                       If None, falls back to regex-based masking.
            prompt_manager: PromptManager instance for loading prompt templates.
                           If None, uses built-in prompts.
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self._use_llm = llm_client is not None

    def mask_question(
        self,
        question: str,
        schema: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> str:
        """
        Mask literals in a natural language question.

        Args:
            question: Input question string
            schema: Optional database schema text for context
            evidence: Optional knowledge evidence

        Returns:
            Question with literals replaced by placeholders
        """
        if self._use_llm and schema:
            # Use LLM when schema is provided (can identify table/column names)
            return self._mask_with_llm(question, "question", schema, evidence)
        else:
            # Use regex when no schema (can only mask values, not table/column names)
            return mask_literals_regex(question)

    def mask_sql(self, sql: str) -> str:
        """
        Mask literals in a SQL query.

        Args:
            sql: SQL query string

        Returns:
            SQL with literals replaced by placeholders
        """
        if self._use_llm:
            return self._mask_with_llm(sql, "sql")
        else:
            return mask_sql_regex(sql)

    def _mask_with_llm(
        self,
        text: str,
        text_type: str,
        schema: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> str:
        """
        Use LLM to mask literals in text.

        Args:
            text: Input text to mask
            text_type: Either "question" or "sql"
            schema: Optional database schema text
            evidence: Optional knowledge evidence

        Returns:
            Masked text
        """
        if text_type == "question":
            prompt = self._build_question_masking_prompt(text, schema, evidence)
            json_schema = get_schema_for_task("question_masking")
        else:
            prompt = self._build_sql_masking_prompt(text)
            json_schema = get_schema_for_task("sql_masking")

        # Try outlines if available
        if OUTLINES_AVAILABLE and self.llm_client:
            try:
                # Use outlines.from_transformers to properly wrap the existing model
                outlines_model = outlines.from_transformers(
                    self.llm_client.model,
                    self.llm_client.tokenizer,
                )

                # Apply chat template
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert SQL developer. Output valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ]

                if self.llm_client.tokenizer.chat_template is not None:
                    prompt_text = self.llm_client.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_text = prompt

                # Call the wrapped model directly with output_type parameter
                if text_type == "question":
                    # Convert dictionary schemas to Pydantic models using outlines utility
                    question_model = json_schema_dict_to_pydantic(
                        json_schema, "QuestionMaskingResult"
                    )
                    try:
                        result = outlines_model(prompt_text, output_type=question_model)
                        # Check if result is a proper Pydantic model with the expected attribute
                        if hasattr(result, 'masked_question'):
                            return result.masked_question
                        else:
                            # If we get a string or other type, return it directly
                            return str(result)
                    except Exception:
                        # If any error occurs during outlines processing, fall back to standard generation
                        response = self.llm_client.generate(prompt)
                        masked_text = self._parse_masking_response(response, text_type)
                        if masked_text and len(masked_text) > 0:
                            return masked_text
                        # If parsing failed, fall back to regex
                        return (
                            mask_literals_regex(text)
                            if text_type == "question"
                            else mask_sql_regex(text)
                        )
                else:
                    # Convert dictionary schemas to Pydantic models using outlines utility
                    sql_model = json_schema_dict_to_pydantic(
                        json_schema, "SQLMaskingResult"
                    )
                    try:
                        result = outlines_model(prompt_text, output_type=sql_model)
                        # Check if result is a proper Pydantic model with the expected attribute
                        if hasattr(result, 'masked_text'):
                            return result.masked_text
                        else:
                            # If we get a string or other type, return it directly
                            return str(result)
                    except Exception:
                        # If any error occurs during outlines processing, fall back to standard generation
                        response = self.llm_client.generate(prompt)
                        masked_text = self._parse_masking_response(response, text_type)
                        if masked_text and len(masked_text) > 0:
                            return masked_text
                        # If parsing failed, fall back to regex
                        return (
                            mask_literals_regex(text)
                            if text_type == "question"
                            else mask_sql_regex(text)
                        )
                    except Exception:
                        # If any error occurs during outlines processing, fall back to standard generation
                        response = self.llm_client.generate(prompt)
                        masked_text = self._parse_masking_response(response, text_type)
                        if masked_text and len(masked_text) > 0:
                            return masked_text
                        # If parsing failed, fall back to regex
                        return (
                            mask_literals_regex(text)
                            if text_type == "question"
                            else mask_sql_regex(text)
                        )
                else:
                    # Convert dictionary schemas to Pydantic models using outlines utility
                    sql_model = json_schema_dict_to_pydantic(
                        json_schema, "SQLMaskingResult"
                    )
                    try:
                        result = outlines_model(prompt_text, output_type=sql_model)
                        # Check if result is a proper Pydantic model with the expected attribute
                        if hasattr(result, 'masked_text'):
                            return result.masked_text
                        else:
                            # If we get a string or other type, return it directly
                            return str(result)
                    except Exception:
                        # If any error occurs during outlines processing, fall back to standard generation
                        response = self.llm_client.generate(prompt)
                        masked_text = self._parse_masking_response(response, text_type)
                        if masked_text and len(masked_text) > 0:
                            return masked_text
                        # If parsing failed, fall back to regex
                        return (
                            mask_literals_regex(text)
                            if text_type == "question"
                            else mask_sql_regex(text)
                        )
                else:
                    # Convert dictionary schemas to Pydantic models using outlines utility
                    sql_model = json_schema_dict_to_pydantic(
                        json_schema, "SQLMaskingResult"
                    )
                    result = outlines_model(prompt_text, output_type=sql_model)
                    # Ensure we get a proper result object
                    if hasattr(result, "masked_text"):
                        return result.masked_text
                    else:
                        # If result is not a proper object, try to extract from string
                        return str(result)

            except Exception as e:
                print(
                    f"  Warning: outlines masking failed: {e}, falling back to standard generation"
                )
                # Fall back to standard generation approach
                response = self.llm_client.generate(prompt)
                masked_text = self._parse_masking_response(response, text_type)
                if masked_text and len(masked_text) > 0:
                    return masked_text
                # If parsing failed, fall back to regex
                return (
                    mask_literals_regex(text)
                    if text_type == "question"
                    else mask_sql_regex(text)
                )

        try:
            response = self.llm_client.generate(prompt)
            masked_text = self._parse_masking_response(response, text_type)
            if masked_text and len(masked_text) > 0:
                return masked_text
            # If parsing returned empty/None, fall back to regex
            return (
                mask_literals_regex(text)
                if text_type == "question"
                else mask_sql_regex(text)
            )
        except Exception as e:
            # On any error, fall back to regex
            return (
                mask_literals_regex(text)
                if text_type == "question"
                else mask_sql_regex(text)
            )

    def _build_question_masking_prompt(
        self,
        question: str,
        schema: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> str:
        """Build prompt for masking a natural language question using prompt template."""
        import os
        from config import get_config

        config = get_config()
        prompt_path = os.path.join(config.PROMPTS_DIR, "question_masking.txt")

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()

            schema_text = schema if schema else "Schema not provided"
            evidence_text = evidence if evidence else "None provided"

            return template.format(
                schema_text=schema_text, question=question, evidence=evidence_text
            )
        except FileNotFoundError:
            # Fallback to built-in prompt with examples from template
            schema_text = schema if schema else "Schema not provided"
            evidence_text = evidence if evidence else "None provided"

            prompt = f"""### Given a DB schema and a question, mask the table name, column name, and values in the question.

Use these placeholder types:
- [TABLE] for table names
- [COLUMN] for column names
- [VALUE] for literal values (numbers, strings, dates, etc.)

<example1>
### SQLite SQL tables, with their properties:
# customers ( CustomerID, Segment, Currency )
# products ( ProductID, Description )
### Question: For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012.
### Masked Question: For all the [TABLE] who paid more than [VALUE] per unit of [COLUMN] [VALUE]. Give their consumption status in the [VALUE].
</example1>

### SQLite SQL tables, with their properties:
{schema_text}

### Question:
{question}

### Knowledge Evidence:
{evidence_text}

### Masked Question:"""
            return prompt

    def _build_sql_masking_prompt(self, sql: str) -> str:
        """Build prompt for masking a SQL query."""
        prompt = f"""### Task: Replace specific literals in the following SQL query with generic placeholders.

Use these placeholder types:
- [NUMBER] for numeric values
- [STRING] for string literals (keep the quotes, replace content with [STRING])
- [DATE] for date literals

Do NOT replace column names, table names, SQL keywords, or function names. Only replace literal values.

### SQL Query:
{sql}

### Masked SQL (JSON format):
{{
    "masked_text": "your masked SQL query here"
}}

### Your Answer:"""
        return prompt

    def _parse_masking_response(self, response: str, text_type: str) -> str:
        """
        Parse LLM response to extract masked text.

        Args:
            response: Raw LLM response
            text_type: Either "question" or "sql"

        Returns:
            Extracted masked text
        """
        response = response.strip()

        # Remove markdown code fences
        if response.startswith("```"):
            response = response[3:]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

        if text_type == "question":
            # For question masking, look for "### Masked Question:" or just extract the masked text
            if "### Masked Question:" in response:
                parts = response.split("### Masked Question:")
                if len(parts) > 1:
                    return parts[1].strip()

            # Try to find text after "Masked Question:"
            if "Masked Question:" in response:
                parts = response.split("Masked Question:")
                if len(parts) > 1:
                    return parts[1].strip()

            # Fallback: return the whole response trimmed
            return response
        else:
            # For SQL masking, try to parse JSON
            try:
                start_idx = response.find("{")
                if start_idx != -1:
                    end_idx = response.rfind("}") + 1
                    if end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        result = json.loads(json_str)
                        return result.get("masked_text", response)
            except:
                pass

            # Fallback: look for "### Masked SQL:"
            if "### Masked SQL:" in response:
                parts = response.split("### Masked SQL:")
                if len(parts) > 1:
                    return parts[1].strip()

            return response


def mask_literals_regex(text: str) -> str:
    """
    Fallback regex-based literal masking.

    Replace literals in text with placeholders.

    Masks:
    - Numbers (integers, floats, percentages)
    - Quoted strings (single and double quotes)
    - Dates (various formats)
    - Currency codes (EUR, USD, CZK, etc.)
    - Boolean values
    - NULL/None values
    """
    masked = text

    # Mask currency codes (3-letter uppercase, common pattern)
    currency_pattern = r"\b(AED|AFN|ALL|AMD|ANG|AOA|ARS|AUD|AWG|AZN|BAM|BBD|BDT|BGN|BHD|BIF|BMD|BND|BOB|BRL|BSD|BTN|BWP|BYN|BZD|CAD|CDF|CHF|CLP|CNY|COP|CRC|CUP|CVE|CZK|DJF|DKK|DOP|DZD|EGP|ERN|ETB|EUR|FJD|FKP|GBP|GEL|GHS|GIP|GMD|GNF|GTQ|GYD|HKD|HNL|HRK|HTG|HUF|IDR|ILS|INR|IQD|IRR|ISK|JMD|JOD|JPY|KES|KGS|KHR|KMF|KPW|KRW|KWD|KYD|KZT|LAK|LBP|LKR|LRD|LSL|LYD|MAD|MDL|MGA|MKD|MMK|MNT|MOP|MRU|MUR|MVR|MWK|MXN|MYR|MZN|NAD|NGN|NIO|NOK|NPR|NZD|OMR|PAB|PEN|PGK|PHP|PKR|PLN|PYG|QAR|RON|RSD|RUB|RWF|SAR|SBD|SCR|SDG|SEK|SGD|SHP|SLL|SOS|SRD|SSP|STN|SYP|SZL|THB|TJS|TMT|TND|TOP|TRY|TTD|TWD|TZS|UAH|UGX|USD|UYU|UZS|VES|VND|VUV|WST|XAF|XCD|XOF|XPF|YER|ZAR|ZMW|ZWL)\b"
    masked = re.sub(currency_pattern, "[CURRENCY]", masked)

    # Mask quoted strings (double quotes)
    masked = re.sub(r'"[^"]*"', "[STRING]", masked)

    # Mask quoted strings (single quotes)
    masked = re.sub(r"'[^']*'", "[STRING]", masked)

    # Mask dates (YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY, etc.)
    date_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",  # 2023-01-15
        r"\b\d{4}/\d{2}/\d{2}\b",  # 2023/01/15
        r"\b\d{2}/\d{2}/\d{4}\b",  # 01/15/2023
        r"\b\d{2}-\d{2}-\d{4}\b",  # 01-15-2023
        r"\b\d{4}\d{2}\d{2}\b",  # 20230115
    ]
    for pattern in date_patterns:
        masked = re.sub(pattern, "[DATE]", masked)

    # Mask percentages
    masked = re.sub(r"\b\d+\.?\d*\s*%", "[PERCENT]", masked)

    # Mask floating point numbers
    masked = re.sub(r"\b\d+\.\d+\b", "[NUMBER]", masked)

    # Mask integers (standalone)
    masked = re.sub(r"\b\d+\b", "[NUMBER]", masked)

    # Mask boolean values
    masked = re.sub(r"\b(TRUE|FALSE|True|False|true|false)\b", "[BOOL]", masked)

    # Mask NULL/None values
    masked = re.sub(
        r"\b(NULL|None|null|none|N/A|NA)\b", "[NULL]", masked, flags=re.IGNORECASE
    )

    return masked


def mask_sql_regex(sql: str) -> str:
    """
    Fallback regex-based SQL literal masking.

    Mask literals in SQL queries.

    Masks:
    - String literals in WHERE clauses
    - Numeric literals
    - Dates in SQL
    """
    masked = sql

    # Mask string literals in SQL (single quotes)
    masked = re.sub(r"'[^']*'", "'[STRING]'", masked)

    # Mask numbers in SQL (but be careful with column names like col1)
    # Only mask standalone numbers or after operators
    masked = re.sub(r"(=|>|<|>=|<=|!=|<>|\s)\d+\.?\d*", r"\g<1>[NUMBER]", masked)

    # Mask dates in SQL
    date_patterns = [
        r"'\d{4}-\d{2}-\d{2}'",
        r"'\d{4}/\d{2}/\d{2}'",
    ]
    for pattern in date_patterns:
        masked = re.sub(pattern, "'[DATE]'", masked)

    return masked


def batch_mask(
    texts: List[str],
    masker: LiteralMasker,
    text_type: str = "question",
    batch_size: int = 10,
) -> List[str]:
    """
    Mask literals in a batch of texts.

    Args:
        texts: List of texts to mask
        masker: LiteralMasker instance
        text_type: Either "question" or "sql"
        batch_size: Size of batches for processing

    Returns:
        List of masked texts
    """
    masked_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            if text_type == "question":
                masked = masker.mask_question(text)
            else:
                masked = masker.mask_sql(text)
            masked_texts.append(masked)

    return masked_texts
