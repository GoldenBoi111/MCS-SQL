"""
Literal Masker using Transformer Models

This module uses a transformer-based LLM to identify and replace literals
in natural language questions and SQL queries with semantic placeholders.

This provides better generalization than regex-based masking by understanding
context and semantics.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple


class LiteralMasker:
    """
    Uses a transformer model to mask literals in text and SQL.
    """

    def __init__(self, llm_client: Optional[Any] = None, prompt_manager: Optional[Any] = None):
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

    def mask_question(self, question: str, schema: Optional[str] = None, evidence: Optional[str] = None) -> str:
        """
        Mask literals in a natural language question.

        Args:
            question: Input question string
            schema: Optional database schema text for context
            evidence: Optional knowledge evidence

        Returns:
            Question with literals replaced by placeholders
        """
        if self._use_llm:
            return self._mask_with_llm(question, "question", schema, evidence)
        else:
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

    def _mask_with_llm(self, text: str, text_type: str, schema: Optional[str] = None, evidence: Optional[str] = None) -> str:
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
        else:
            prompt = self._build_sql_masking_prompt(text)

        try:
            response = self.llm_client.generate(prompt)
            masked_text = self._parse_masking_response(response, text_type)
            return masked_text if masked_text else text
        except Exception as e:
            print(f"LLM masking failed: {e}, falling back to regex")
            if text_type == "question":
                return mask_literals_regex(text)
            else:
                return mask_sql_regex(text)

    def _build_question_masking_prompt(self, question: str, schema: Optional[str] = None, evidence: Optional[str] = None) -> str:
        """Build prompt for masking a natural language question using prompt template."""
        if self.prompt_manager and self.prompt_manager.templates.get("question_masking"):
            # Use the prompt template from file
            return self.prompt_manager.build_prompt(
                name="question_masking",
                schema=schema if schema else "",
                question=question,
                evidence=evidence if evidence else "",
            )
        else:
            # Fallback to built-in prompt
            schema_text = schema if schema else "Schema not provided"
            evidence_text = evidence if evidence else "None provided"
            
            prompt = f"""### Given a DB schema and a question, mask the table name, column name, and values
in the question.

Use these placeholder types:
- [TABLE] for table names
- [COLUMN] for column names
- [VALUE] for literal values (numbers, strings, dates, etc.)

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

    def _parse_masking_response(self, response: str, text_type: str) -> Optional[str]:
        """
        Parse LLM response to extract masked text.

        Args:
            response: Raw LLM response
            text_type: Either "question" or "sql"

        Returns:
            Extracted masked text or None if parsing fails
        """
        # First, try to extract from "### Masked Question:" or "### Masked SQL:" format
        # (used by question_masking.txt prompt template)
        import re
        
        if text_type == "question":
            # Look for content after "### Masked Question:"
            match = re.search(r"### Masked Question:\s*(.+?)(?=###|$)", response, re.DOTALL)
            if match:
                masked_text = match.group(1).strip()
                if masked_text:
                    return masked_text
        
        # Try to find JSON format (fallback for backward compatibility)
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                masked_text = result.get("masked_text", "")

                if masked_text:
                    return masked_text
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing masking response: {e}")

        # Try to extract masked text directly if JSON parsing fails
        lines = response.strip().split("\n")
        for line in lines:
            if '"' in line and ("masked" in line.lower() or "text" in line.lower()):
                # Extract content between quotes after colon
                match = re.search(r':\s*"([^"]+)"', line)
                if match:
                    return match.group(1)

        # If all else fails, return the entire response (trimmed)
        # This handles cases where LLM just outputs the masked text directly
        if response.strip():
            return response.strip()

        return None


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
    currency_pattern = r'\b(AED|AFN|ALL|AMD|ANG|AOA|ARS|AUD|AWG|AZN|BAM|BBD|BDT|BGN|BHD|BIF|BMD|BND|BOB|BRL|BSD|BTN|BWP|BYN|BZD|CAD|CDF|CHF|CLP|CNY|COP|CRC|CUP|CVE|CZK|DJF|DKK|DOP|DZD|EGP|ERN|ETB|EUR|FJD|FKP|GBP|GEL|GHS|GIP|GMD|GNF|GTQ|GYD|HKD|HNL|HRK|HTG|HUF|IDR|ILS|INR|IQD|IRR|ISK|JMD|JOD|JPY|KES|KGS|KHR|KMF|KPW|KRW|KWD|KYD|KZT|LAK|LBP|LKR|LRD|LSL|LYD|MAD|MDL|MGA|MKD|MMK|MNT|MOP|MRU|MUR|MVR|MWK|MXN|MYR|MZN|NAD|NGN|NIO|NOK|NPR|NZD|OMR|PAB|PEN|PGK|PHP|PKR|PLN|PYG|QAR|RON|RSD|RUB|RWF|SAR|SBD|SCR|SDG|SEK|SGD|SHP|SLL|SOS|SRD|SSP|STN|SYP|SZL|THB|TJS|TMT|TND|TOP|TRY|TTD|TWD|TZS|UAH|UGX|USD|UYU|UZS|VES|VND|VUV|WST|XAF|XCD|XOF|XPF|YER|ZAR|ZMW|ZWL)\b'
    masked = re.sub(currency_pattern, '[CURRENCY]', masked)

    # Mask quoted strings (double quotes)
    masked = re.sub(r'"[^"]*"', '[STRING]', masked)

    # Mask quoted strings (single quotes)
    masked = re.sub(r"'[^']*'", '[STRING]', masked)

    # Mask dates (YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY, etc.)
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # 2023-01-15
        r'\b\d{4}/\d{2}/\d{2}\b',  # 2023/01/15
        r'\b\d{2}/\d{2}/\d{4}\b',  # 01/15/2023
        r'\b\d{2}-\d{2}-\d{4}\b',  # 01-15-2023
        r'\b\d{4}\d{2}\d{2}\b',    # 20230115
    ]
    for pattern in date_patterns:
        masked = re.sub(pattern, '[DATE]', masked)

    # Mask percentages
    masked = re.sub(r'\b\d+\.?\d*\s*%', '[PERCENT]', masked)

    # Mask floating point numbers
    masked = re.sub(r'\b\d+\.\d+\b', '[NUMBER]', masked)

    # Mask integers (standalone)
    masked = re.sub(r'\b\d+\b', '[NUMBER]', masked)

    # Mask boolean values
    masked = re.sub(r'\b(TRUE|FALSE|True|False|true|false)\b', '[BOOL]', masked)

    # Mask NULL/None values
    masked = re.sub(r'\b(NULL|None|null|none|N/A|NA)\b', '[NULL]', masked, flags=re.IGNORECASE)

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
