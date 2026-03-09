"""
Prompt Manager for Text-to-SQL

This module loads and manages prompt templates from the prompts folder.
It provides methods to build complete prompts by filling in placeholders.
"""

import os
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Represents a loaded prompt template."""
    name: str
    content: str
    placeholders: List[str]


class PromptManager:
    """
    Manages prompt templates from the prompts folder.
    
    Attributes:
        prompts_dir: Directory containing prompt .txt files
        templates: Dictionary of loaded prompt templates
    """
    
    def __init__(self, prompts_dir: str):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Path to the directory containing prompt templates
        """
        self.prompts_dir = prompts_dir
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all .txt prompt files from the prompts directory."""
        if not os.path.exists(self.prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith(".txt"):
                name = filename[:-4]  # Remove .txt extension
                filepath = os.path.join(self.prompts_dir, filename)
                
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Extract placeholders (### markers indicate sections to fill)
                placeholders = self._extract_placeholders(content)
                
                self.templates[name] = PromptTemplate(
                    name=name,
                    content=content,
                    placeholders=placeholders
                )
    
    def _extract_placeholders(self, content: str) -> List[str]:
        """
        Extract placeholder markers from prompt content.
        Placeholders are indicated by {placeholder_name} format.
        
        Args:
            content: Prompt template content
            
        Returns:
            List of placeholder names
        """
        import re
        # Find patterns like {schema_text}, {question}, {evidence}, etc.
        pattern = r"\{([^}]+)\}"
        matches = re.findall(pattern, content)
        return list(set([match.strip() for match in matches]))  # Remove duplicates
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.
        
        Args:
            name: Name of the prompt template (without .txt extension)
            
        Returns:
            PromptTemplate object or None if not found
        """
        return self.templates.get(name)
    
    def build_prompt(
        self,
        name: str,
        schema: Optional[str] = None,
        question: Optional[str] = None,
        evidence: Optional[str] = None,
        additional_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build a complete prompt by filling in placeholders.
        
        Args:
            name: Name of the prompt template
            schema: Database schema text (for {schema_text} placeholder)
            question: User question
            evidence: Knowledge evidence
            additional_context: Additional key-value pairs for custom placeholders
            
        Returns:
            Complete prompt string
        """
        template = self.templates.get(name)
        if not template:
            raise ValueError(f"Prompt template '{name}' not found")
        
        content = template.content
        
        # Replace {schema_text} placeholder
        if schema:
            content = content.replace("{schema_text}", schema)
        
        # Replace {question} placeholder
        if question:
            content = content.replace("{question}", question)
        
        # Replace {evidence} placeholder
        if evidence:
            content = content.replace("{evidence}", evidence)
        else:
            content = content.replace("{evidence}", "None provided")
        
        # Replace {selected_tables} placeholder (for column_linking)
        if additional_context:
            if "selected_tables" in additional_context:
                tables = additional_context["selected_tables"]
                if isinstance(tables, list):
                    content = content.replace("{selected_tables}", ", ".join(tables))
                else:
                    content = content.replace("{selected_tables}", str(tables))
            
            # Replace {candidate_sqls} placeholder (for SQL_selection)
            if "candidate_sqls" in additional_context:
                candidate_sqls = additional_context["candidate_sqls"]
                if isinstance(candidate_sqls, list):
                    formatted_sqls = "\n".join(
                        f"{i+1}. {sql}" for i, sql in enumerate(candidate_sqls)
                    )
                    content = content.replace("{candidate_sqls}", formatted_sqls)
                else:
                    content = content.replace("{candidate_sqls}", str(candidate_sqls))
            
            # Replace any other custom placeholders
            for key, value in additional_context.items():
                if key not in ["selected_tables", "candidate_sqls"]:
                    content = content.replace(f"{{{key}}}", str(value))
        
        return content
    
    def list_templates(self) -> List[str]:
        """Return list of available template names."""
        return list(self.templates.keys())


# Example usage
if __name__ == "__main__":
    from config import Config
    
    # Load configuration
    config = Config()
    prompts_dir = config.PROMPTS_DIR
    
    manager = PromptManager(prompts_dir)

    print("Available templates:", manager.list_templates())

    # Example: Build a question masking prompt
    schema_text = """# customers ( CustomerID: integer, Segment: text, Currency: text )
# transactions ( TransactionID: integer, Date: date, CustomerID: integer, Amount: real )"""

    prompt = manager.build_prompt(
        name="question_masking",
        schema=schema_text,
        question="What is the average amount paid by customers in 2023?",
    )

    print("\n=== Built Prompt ===")
    print(prompt)
