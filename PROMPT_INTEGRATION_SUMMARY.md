# Prompt Template Integration Summary

This document summarizes the changes made to align the codebase with the prompt templates in the `prompts/` folder.

## Prompt Templates

The following prompt templates are defined in the `prompts/` folder:

1. **table_linking.txt** - Schema linking for table selection
2. **column_linking.txt** - Schema linking for column selection  
3. **SQL_generation.txt** - SQL generation with few-shot examples
4. **SQL_selection.txt** - SQL selection from candidates
5. **question_masking.txt** - Question masking for literal replacement

## Changes Made

### 1. `engine/schema_linking.py`

#### Updated `format_schema_for_prompt()` method
**Before:**
```python
lines.append(f"Table: {table}\nColumns: {columns}")
return "\n\n".join(lines)
```

**After:**
```python
# Format: # table_name ( col1, col2, col3 )
col_str = ", ".join(columns)
lines.append(f"# {table} ( {col_str} )")
return "\n".join(lines)
```

**Reason:** The prompt templates expect schema format as:
```
# table_name ( column1, column2, column3 )
```

Not:
```
Table: table_name
Columns: column1, column2, column3
```

### 2. `engine/literal_masker.py`

#### Updated `_build_question_masking_prompt()` method
**Before:** Used PromptManager or built-in prompt without examples

**After:** Loads from `prompts/question_masking.txt` file with fallback

```python
import os
from config import get_config
config = get_config()
prompt_path = os.path.join(config.PROMPTS_DIR, "question_masking.txt")

try:
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    return template.format(
        schema_text=schema_text,
        question=question,
        evidence=evidence_text
    )
except FileNotFoundError:
    # Fallback to built-in prompt with examples
    ...
```

**Reason:** Ensures consistency with the official prompt template that includes few-shot examples.

#### Updated `_parse_masking_response()` method
**Before:** Complex regex parsing

**After:** Simple string splitting based on template format

```python
if text_type == "question":
    if "### Masked Question:" in response:
        parts = response.split("### Masked Question:")
        if len(parts) > 1:
            return parts[1].strip()
```

**Reason:** The prompt template ends with `### Masked Question:`, so the response should contain the masked question after this marker.

### 3. `engine/run_benchmark.py`

#### Added outlines import
```python
try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
```

**Reason:** Enable forced JSON output using outlines library.

#### Updated docstring
Added note about outlines usage for forced JSON output.

### 4. `engine/multiple_generation.py`

#### Added outlines import
```python
try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
```

**Reason:** Enable forced JSON output using outlines library.

#### Updated docstring
Added note about outlines usage.

### 5. `requirements.txt`

#### Added outlines dependency
```txt
outlines>=0.0.40
```

**Reason:** Required for structured JSON generation.

## Prompt Template Formats

### Table Linking (`table_linking.txt`)
```
### Given a database schema, question, and knowledge evidence, extract a list of tables that should be referenced to convert the question into SQL.
### SQLite SQL tables, with their properties:
{schema_text}

### Question:
{question}

### Knowledge Evidence:
{evidence}

You need to not only select the required tables, but also explain in detail why each table is needed.
Your answer should strictly follow the following json format.
{{
    "reasoning": "", // The reason for choosing each table.
    "tables": [], // List of selected tables.
}}

### Your Answer:
```

### Column Linking (`column_linking.txt`)
```
### Given a database schema, question, and knowledge evidence, extract a list of
columns that should be referenced to convert the question into SQL.
### SQLite SQL tables, with their properties:
{schema_text}

### Selected Tables:
{selected_tables}

### Question:
{question}

### Knowledge Evidence:
{evidence}

You need to not only select the required columns, but also explain in detail why each column is needed.
Your answer should strictly follow the following json format.
{{
    "reasoning": "", // The reason for choosing each column.
    "columns": ["table_name_i.column_name_j", ...], // List of selected columns
}}

### Your Answer:
```

### SQL Generation (`SQL_generation.txt`)
```
### Given a database schema, question, and knowledge evidence, generate the correct sqlite SQL query for the question.

### Relevant examples from index:
{examples}

### SQLite SQL tables, with their properties:
{schema_text}

### Sample rows of each table in csv format:
{sample_contents}

### Question:
{question}

### Knowledge Evidence:
{evidence}

You need to not only create the SQL, but also provide the detailed reasoning steps required to create the SQL. Your answer should strictly follow the following
json format:
{
"reasoning": "", // The reasoning steps for generating SQL.
"sql": "", // The final generated SQL.
}
### Your Answer:
```

### SQL Selection (`SQL_selection.txt`)
```
### When a DB schema, a question, and a knowledge evidence are given, and up to three SQLite queries expressing the question are given, please choose the most accurate SQL based on the Checklist.
### SQLite SQL tables, with their properties:
{schema_text}

### Question:
{question}

### Knowledge Evidence:
{evidence}

### Candidate SQLs:
{candidate_sqls}

### Checklist:
1. The SQL should accurately represent the question.
2. The SQL should accurately use the given knowledge evidence.
3. The SELECT clause should not include any additional columns that are not included in the question.
4. The order of columns in the SELECT clause must be the same as the order in the question.
5. Check if the operations are performed correctly according to the column type.

### Instruction:
- If the first SQL satisfies all the conditions of the checklist, please choose the first SQL. If not, move on to the next SQL.
- If there's no SQL that satisfies all the requirements on the checklist, just choose the first SQL.
- Provide a detailed step-by-step explanation following the order of the checklist when checking whether each SQL satisfies the checklist.
- Your answer should strictly follow the following json format.
{{
"reasoning": "", // The reasoning steps for choosing the best SQL.
"sql": "", // The final chosen SQL.
}}
### Your Answer:
```

### Question Masking (`question_masking.txt`)
```
### Given a DB schema and a question, mask the table name, column name, and values in the question.

Use these placeholder types:
- [TABLE] for table names
- [COLUMN] for column names
- [VALUE] for literal values (numbers, strings, dates, etc.)

<example1>
### SQLite SQL tables, with their properties:
# customers ( CustomerID, Segment, Currency )
# gasstations ( GasStationID, ChainID, Country, Segment )
# products ( ProductID, Description )
# transactions_1k ( TransactionID, Date, Time, CustomerID, CardID, GasStationID, ProductID, Amount, integer, Price, real )
# yearmonth ( CustomerID, Date, Consumption )
### Question: For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012.
### Masked Question: For all the [TABLE] who paid more than [VALUE] per unit of [COLUMN] [VALUE]. Give their consumption status in the [VALUE].
</example1>

### Your Task:

### SQLite SQL tables, with their properties:
{schema_text}

### Question:
{question}

### Knowledge Evidence:
{evidence}

### Masked Question:
```

## Key Format Requirements

### Schema Format
All prompts expect schema in this format:
```
# table_name ( column1, column2, column3 )
```

### JSON Output Format
All prompts require JSON output with double braces for Python `.format()`:
```python
{{
    "reasoning": "",
    "tables": []  # or "columns": [] or "sql": ""
}}
```

### Variable Names
Template variables must match exactly:
- `{schema_text}` - Formatted schema
- `{question}` - Natural language question
- `{evidence}` - Knowledge evidence
- `{selected_tables}` - For column linking
- `{examples}` - For SQL generation
- `{sample_contents}` - For SQL generation
- `{candidate_sqls}` - For SQL selection

## Testing Recommendations

1. **Test schema formatting:**
   ```python
   from schema_linking import SchemaLinker
   linker = SchemaLinker()
   schema = {"table1": ["col1", "col2"]}
   formatted = linker.format_schema_for_prompt(schema)
   assert formatted == "# table1 ( col1, col2 )"
   ```

2. **Test prompt loading:**
   ```python
   from literal_masker import LiteralMasker
   masker = LiteralMasker(llm_client=mock_client)
   prompt = masker._build_question_masking_prompt(
       question="Test?",
       schema="# test ( id )",
       evidence="None"
   )
   assert "### Masked Question:" in prompt
   ```

3. **Test response parsing:**
   ```python
   response = "### Masked Question: Test [VALUE]"
   masked = masker._parse_masking_response(response, "question")
   assert masked == "Test [VALUE]"
   ```

## Migration Notes

- **Backward Compatibility:** The code includes fallbacks for missing prompt files
- **Error Handling:** Graceful degradation to regex-based masking if LLM fails
- **JSON Parsing:** Robust parsing with markdown code fence handling
