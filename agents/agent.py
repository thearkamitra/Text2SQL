"""
SQL Generation Agent using LangChain and LLM Generator
This agent takes natural language queries and converts them to SQL using table schema information.
"""
import re
import json
import argparse
import os
from typing import Dict, List, Union, Optional, Any
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from llm_generator import LLMAgentFactory, LLMProvider

# Pydantic models for structured output
class SQLResult(BaseModel):
    """Individual SQL query result"""
    question: str = Field(description="The natural language question")
    sql: str = Field(description="The generated SQL query")

class SQLOutputList(BaseModel):
    """List of SQL query results with no keys"""
    outputs: List[SQLResult] = Field(description="List of question-SQL pairs")

    def to_dict_list(self) -> List[Dict[str, str]]:
        return [{"question": item.question, "sql": item.sql} for item in self.outputs]


class SQLGenerationAgent:
    """
    Agent that converts natural language queries to SQL using LangChain RunnableSequence
    and enforces strict JSON schema output.
    """

    def __init__(self, 
                 provider: LLMProvider, 
                 model_name: Optional[str] = None,
                 env_path_name: Optional[str] = None,
                 **kwargs):
        self.llm_agent = LLMAgentFactory.create_agent(
            provider=provider,
            model_name=model_name,
            env_path_name=env_path_name,
            **kwargs
        )
        self.llm = self.llm_agent.create_llm()

        self.table_schemas = {}
        self._setup_prompts()
        self._load_default_schemas()

    def _load_default_schemas(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            tables_file = os.path.join(project_root, "sciencebenchmark_dataset", "cordis", "tables.json")
            columns_file = os.path.join(project_root, "datasets", "columns.json")
            columns_desc_file = os.path.join(project_root, "datasets", "columns_description.json")

            if os.path.exists(tables_file) and os.path.exists(columns_file):
                self._load_schemas_from_files(tables_file, columns_file, columns_desc_file)
            else:
                print(f"⚠️ Schema files missing: {tables_file}, {columns_file}")
        except Exception as e:
            print(f"⚠️ Could not load default schemas: {e}")

    def _load_schemas_from_files(self, tables_file: str, columns_file: str, columns_desc_file: str):
        try:
            with open(tables_file, 'r') as f:
                self.tables_data = json.load(f)
            with open(columns_file, 'r') as f:
                self.columns_data = json.load(f)
            self.columns_descriptions = {}
            if os.path.exists(columns_desc_file):
                with open(columns_desc_file, 'r') as f:
                    self.columns_descriptions = json.load(f)
            print(f"✅ Schema data loaded")
        except Exception as e:
            print(f"❌ Error loading schemas: {e}")
            raise

    def _setup_prompts(self):
        """Setup strict prompts enforcing JSON output"""
        strict_guidelines = """
IMPORTANT: Follow these rules:
1. Output only valid JSON — no markdown, no extra text.
2. JSON must be a list of dictionaries.
3. Each element must be a dictionary with two keys: "question" and "sql".
4. Do not modify or reword the input questions.
5. Ensure all SQL queries use proper syntax and only the given schema.
6. Avoid duplicates by using DISTINCT where appropriate.
7. Absolutely no commentary or extra explanation — only JSON.
8. Always output a single valid JSON array.
9. The JSON must be enclosed in square brackets: [ ... ].
10. Each element must be a dictionary with two keys: "question" and "sql".
11. Use ONLY double quotes (") for JSON strings.
12. Inside SQL, single quotes (') are allowed but must NOT be escaped as \'.
13. Do NOT wrap the output in markdown code blocks (no ```json).
14. Do NOT output any text or commentary outside the JSON array.
"""

        self.single_query_prompt = PromptTemplate(
            input_variables=["table_info", "column_info", "question"],
            template=(
                "You are an expert SQL generator. "
                "Convert the following natural language question into SQL.\n\n"
                f"{strict_guidelines}\n\n"
                "Table Schema:\n{{table_info}}\n\n"
                "Columns:\n{{column_info}}\n\n"
                "Question:\n{{question}}\n\n"
                'Respond ONLY with valid JSON (no code blocks):\n'
                '{{"outputs": [{{"question": "<same question>", "sql": "<generated SQL query>"}}]}}'
            )
        )

        self.multiple_queries_prompt = PromptTemplate(
            input_variables=["table_info", "column_info", "questions"],
            template=(
                "You are an expert SQL generator. "
                "Convert the following natural language questions into SQL queries.\n\n"
                f"{strict_guidelines}\n\n"
                "Table Schema:\n{table_info}\n\n"
                "Columns:\n{column_info}\n\n"
                "Questions:\n{questions}\n\n"
                "Your output must strictly follow this JSON format (this is just an example, "
                "replace with real questions and queries):\n"
                '{{['
                '{{"question": "Show me project member roles for the project member INSTITUTO SUPERIOR TECNICO", "sql": "SELECT project_member_roles.description FROM project_member_roles JOIN project_members ON project_member_roles.code = project_members.member_role WHERE project_members.member_name = \'INSTITUTO SUPERIOR TECNICO\'"}}, '
                '{{"question": "Show me all projects members that are not in the same location as 16.5972215.", "sql": "SELECT projects.homepage FROM projects JOIN project_members ON projects.unics_id = project_members.project WHERE project_members.longitude != 16.5972215"}}'
                '.....'
                ']}}'),
        )

    def _run_llm(self, prompt: PromptTemplate, variables: Dict[str, Any]) -> str:
        """Run the prompt through a chain with output parsing"""

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(variables)

    def generate_sql(self, question: Union[str, List[str]], use_columns: bool = True, use_descriptions: bool = True) -> List[Dict[str, str]]:
        if not ((hasattr(self, 'tables_data') and self.tables_data) or self.table_schemas):
            raise ValueError("No table schemas found. Load schemas first.")

        table_info = self.tables_data
        column_info = self.columns_descriptions if use_descriptions and hasattr(self, 'columns_descriptions') else self.columns_data
        if not use_columns:
            column_info = {}

        # Prepare variables
        if isinstance(question, list):
            questions_text = "\n ".join(question)
            llm_output = self._run_llm(self.multiple_queries_prompt, {
                "questions": questions_text,
                "table_info": table_info,
                "column_info": column_info
            })
        else:
            llm_output = self._run_llm(self.single_query_prompt, {
                "question": question,
                "table_info": table_info,
                "column_info": column_info
            })
        # Validate JSON using Pydantic
        raw_text = re.sub(r"```(?:json|sql)?\n?", "", llm_output).replace("```", "")

        # Replace escaped single quotes (\' -> ')
        cleaned = raw_text.replace("\\'", "'")

        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]

        cleaned = cleaned.strip()
        if not cleaned.endswith("]"):
            cleaned += "]"

        # Ensure it's wrapped in a JSON array
        if not cleaned.startswith("["):
            cleaned = f"[{cleaned}"
        breakpoint()  # Debugging point to inspect cleaned output
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                data = {"outputs": data}
            output_list = SQLOutputList(**data)
            return output_list.to_dict_list()
        except (json.JSONDecodeError, ValidationError) as e:
            # Fallback: return stub if validation fails
            if isinstance(question, list):
                return [{"question": q, "sql": "-- SQL generation failed"} for q in question]
            return [{"question": question, "sql": "-- SQL generation failed"}]

    def is_available(self) -> bool:
        return self.llm_agent.is_available()

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.llm_agent.__class__.__name__,
            "model_name": self.llm_agent.model_name,
            "available": self.is_available(),
        }


# Example usage and factory functions
def create_sql_agent(provider: LLMProvider, 
                    model_name: Optional[str] = None,
                    **kwargs) -> SQLGenerationAgent:
    """
    Factory function to create a SQL Generation Agent
    
    Args:
        provider: LLM provider (OpenAI, Groq, or Ollama)  
        model_name: Optional model name override
        **kwargs: Additional configuration
        
    Returns:
        Configured SQLGenerationAgent instance
    """
    return SQLGenerationAgent(
        provider=provider,
        model_name=model_name,
        **kwargs
    )

def main():
    """Main function with argparse for CLI usage"""
    parser = argparse.ArgumentParser(
        description="SQL Generation Agent - Convert natural language to SQL queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --provider groq --descriptive
  python agent.py --provider openai --no-descriptive
  python agent.py --provider ollama --model "qwen2:0.5b"
        """
    )
    
    # Provider selection
    parser.add_argument(
        '--provider', 
        choices=['groq', 'openai', 'ollama'],
        default='groq',
        help='LLM provider to use (default: groq)'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        help='Model name to use (optional, uses provider defaults if not specified)'
    )
    
    # Column descriptions flag
    parser.add_argument(
        '--use_column',
        action='store_true',
        default=False,
        help='Use column information in prompts (default: False)'
    )
    
    parser.add_argument(
        '--no-descriptive',
        dest='descriptive',
        action='store_false',
        help='Do not use column descriptions, use basic column info only'
    )
    
    # Single question input
    parser.add_argument(
        '--question',
        type=str,
        help='Single question to convert to SQL',
        default=None
    )
    
    args = parser.parse_args()
    
    # Map provider string to enum
    provider_map = {
        'groq': LLMProvider.GROQ,
        'openai': LLMProvider.OPENAI,
        'ollama': LLMProvider.OLLAMA
    }
    
        # Create agent with specified provider and model
    agent_kwargs = {}
    if args.model:
        agent_kwargs['model_name'] = args.model
        
    agent = create_sql_agent(
        provider=provider_map[args.provider],
        **agent_kwargs
    )
    
    print(f"✅ Agent created successfully!")
    print(f"📊 Model Info: {agent.get_model_info()}")
    print(f"🔧 Using descriptions: {args.descriptive}")
    
    # Determine questions to process
    
    if args.use_column is False:
        args.descriptive = False

    if args.question:
        # Single question
        print(f"\n🔍 Processing single question:")
        print(f"   Question: {args.question}")
        result = agent.generate_sql(args.question, use_columns=args.use_column, use_descriptions=args.descriptive)
        print(f"   Result: {json.dumps(result, indent=2)}")
        
    else:

        with open("datasets/total_interest.json") as f:
            data = json.load(f)
        questions = [item["question"] for item in data]
        if args.model is None:
            args.model = agent.get_model_info().get("model_name")
        
        result = agent.generate_sql(questions, use_columns=args.use_column, use_descriptions=args.descriptive)
        print(f"   Results: {json.dumps(result, indent=2)}")
        model_name = "-".join(args.model.split("/"))
        string_name = f"{args.provider}_{model_name}_{args.descriptive}_{args.use_column}.json"
        with open("datasets/" + string_name, "w") as f:
            json.dump(result, f, indent=2)

    return 0


if __name__ == "__main__":
    main()