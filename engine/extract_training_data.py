#!/usr/bin/env python3
"""
Script to extract and transform training data from JSON file.
Converts "query" to "SQL" and keeps only db_id, SQL, and question fields.
"""

import json
import sys
from typing import List, Dict, Any


def transform_training_data(input_file: str, output_file: str) -> None:
    """
    Transform training data by renaming 'query' to 'SQL' and filtering fields.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Transform data
    transformed_data = []
    for item in data:
        # Create new item with required fields only
        transformed_item = {
            "db_id": item.get("db_id", ""),
            "SQL": item.get("query", ""),
            "question": item.get("question", ""),
        }
        transformed_data.append(transformed_item)

    # Write output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)

    print(f"Transformed {len(transformed_data)} records")
    print(f"Output saved to: {output_file}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 3:
        print("Usage: python extract_training_data.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        transform_training_data(input_file, output_file)
        print("Data transformation completed successfully!")
    except Exception as e:
        print(f"Error during transformation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
