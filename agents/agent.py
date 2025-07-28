"""
SQL Generation Agent using LangChain and LLM Generator
This agent takes natural language queries and converts them to SQL using table schema information.
"""

import json
import os
import re
import argparse
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

from llm_generator import LLMAgentFactory, LLMProvider, BaseLLMAgent

class SQLOutputParser:
    """Custom parser to extract SQL queries from LLM responses and format as question-SQL pairs"""
    
    def parse(self, text: str, questions: Union[str, List[str]] = None) -> List[Dict[str, str]]:
        """
        Parse SQL query/queries from text and format as question-SQL pairs
        
        Args:
            text: LLM response text containing SQL queries
            questions: Single question string or list of questions
            
        Returns:
            List of dictionaries with 'question' and 'sql' keys
        """
        # First, try to find JSON array in the text
        json_array_pattern = r'\[\s*\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"sql"\s*:\s*"[^"]*"\s*\}(?:\s*,\s*\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"sql"\s*:\s*"[^"]*"\s*\})*\s*\]'
        json_match = re.search(json_array_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            try:
                json_data = json.loads(json_match.group(0))
                if isinstance(json_data, list) and len(json_data) > 0 and all(
                    isinstance(item, dict) and 'question' in item and 'sql' in item 
                    for item in json_data
                ):
                    return json_data
            except json.JSONDecodeError:
                pass
        
        # If no JSON array found, try to find JSON code blocks
        json_code_blocks = re.findall(r'```json\n(.*?)\n```', text, re.DOTALL)
        for block in json_code_blocks:
            try:
                json_data = json.loads(block)
                if isinstance(json_data, list) and len(json_data) > 0 and all(
                    isinstance(item, dict) and 'question' in item and 'sql' in item 
                    for item in json_data
                ):
                    return json_data
            except json.JSONDecodeError:
                continue
        
        # Remove markdown code blocks and clean text
        cleaned_text = re.sub(r'```sql\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
        cleaned_text = re.sub(r'```\n(.*?)\n```', r'\1', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'```json\n(.*?)\n```', r'\1', cleaned_text, flags=re.DOTALL)
        
        # Try to parse the entire cleaned text as JSON
        try:
            json_data = json.loads(cleaned_text)
            if isinstance(json_data, list) and len(json_data) > 0 and all(
                isinstance(item, dict) and 'question' in item and 'sql' in item 
                for item in json_data
            ):
                return json_data
        except json.JSONDecodeError:
            pass
        
        # Extract SQL queries using regex patterns
        sql_pattern = r'((?:SELECT|INSERT|UPDATE|DELETE).*?(?:;|$|WHERE.*?;?))'
        matches = re.findall(sql_pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
        
        # Clean up the matches
        cleaned_queries = []
        for match in matches:
            query = match.strip().rstrip(';')
            # Remove extra whitespace and newlines
            query = re.sub(r'\s+', ' ', query)
            if query and len(query) > 10:  # Filter out very short matches
                cleaned_queries.append(query)
        
        # If still no queries found, try a more aggressive approach
        if not cleaned_queries:
            # Look for any line that starts with SELECT, INSERT, UPDATE, DELETE
            lines = cleaned_text.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^(SELECT|INSERT|UPDATE|DELETE)', line, re.IGNORECASE):
                    cleaned_queries.append(line.rstrip(';'))
        
        # If still no queries, try to extract from legacy format
        if not cleaned_queries:
            try:
                json_data = json.loads(cleaned_text)
                if isinstance(json_data, dict) and 'queries' in json_data:
                    cleaned_queries = json_data['queries']
                else:
                    cleaned_queries = [cleaned_text.strip()]
            except json.JSONDecodeError:
                cleaned_queries = [cleaned_text.strip()]
        
        # Format as question-SQL pairs
        result = []
        
        if isinstance(questions, list):
            # Multiple questions - pair each with corresponding SQL
            for i, question in enumerate(questions):
                sql_query = cleaned_queries[i] if i < len(cleaned_queries) else "-- SQL generation failed"
                result.append({
                    "question": question,
                    "sql": sql_query
                })
        else:
            # Single question
            if cleaned_queries:
                result.append({
                    "question": questions or "Generated SQL query",
                    "sql": cleaned_queries[0]
                })
            else:
                result.append({
                    "question": questions or "Generated SQL query", 
                    "sql": "-- SQL generation failed"
                })
        
        return result


class SQLGenerationAgent:
    """
    Agent that converts natural language queries to SQL using LangChain and LLM Generator
    """
    
    def __init__(self, 
                 provider: LLMProvider, 
                 model_name: Optional[str] = None,
                 env_path_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize the SQL Generation Agent
        
        Args:
            provider: LLM provider (OpenAI, Groq, or Ollama)
            model_name: Name of the model to use
            env_path_name: Path to environment file
            **kwargs: Additional configuration parameters
        """
        # Create LLM agent
        self.llm_agent = LLMAgentFactory.create_agent(
            provider=provider,
            model_name=model_name,
            env_path_name=env_path_name,
            **kwargs
        )
        
        # Get the LLM instance
        self.llm = self.llm_agent.create_llm()
        
        # Initialize table schemas dictionary (for backward compatibility)
        self.table_schemas = {}
        
        # Initialize prompts
        self._setup_prompts()
        
        # Load default schema data
        self._load_default_schemas()
    
    def _load_default_schemas(self):
        """Load default schemas from the project data files"""
        try:
            # Get the project root directory (go up from agents/ to root)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            # Load table structure from cordis/tables.json
            tables_file = os.path.join(project_root, "sciencebenchmark_dataset", "cordis", "tables.json")
            columns_file = os.path.join(project_root, "datasets", "columns.json")
            columns_desc_file = os.path.join(project_root, "datasets", "columns_description.json")
            
            if os.path.exists(tables_file) and os.path.exists(columns_file):
                self._load_schemas_from_files(tables_file, columns_file, columns_desc_file)
            else:
                print(f"⚠️  Warning: Schema files not found. Please ensure files exist:")
                print(f"   - {tables_file}")
                print(f"   - {columns_file}")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not load default schemas: {e}")
    
    def _load_schemas_from_files(self, tables_file: str, columns_file: str, columns_desc_file: str):
        """Load schema information from JSON files"""
        try:
            # Load tables structure
            with open(tables_file, 'r') as f:
                self.tables_data = json.load(f)
            
            # Load columns information
            with open(columns_file, 'r') as f:
                self.columns_data = json.load(f)
                
            # Load columns descriptions (if available)
            if os.path.exists(columns_desc_file):
                with open(columns_desc_file, 'r') as f:
                    self.columns_descriptions = json.load(f)
            else:
                self.columns_descriptions = {}
            
            print(f"✅ Loaded schema data from project files")
            
        except Exception as e:
            print(f"❌ Error loading schemas from files: {e}")
            raise
    
    def reload_schemas(self, dataset_path: Optional[str] = None):
        """
        Reload schemas from project files or custom path
        
        Args:
            dataset_path: Optional custom path to dataset directory
        """
        if dataset_path:
            tables_file = os.path.join(dataset_path, "tables.json")
            columns_file = os.path.join(dataset_path, "columns.json") 
            columns_desc_file = os.path.join(dataset_path, "columns_description.json")
        else:
            # Use default paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            tables_file = os.path.join(project_root, "sciencebenchmark_dataset", "cordis", "tables.json")
            columns_file = os.path.join(project_root, "datasets", "columns.json")
            columns_desc_file = os.path.join(project_root, "datasets", "columns_description.json")
        
        # Clear existing schemas and data
        self.table_schemas = {}
        if hasattr(self, 'tables_data'):
            delattr(self, 'tables_data')
        if hasattr(self, 'columns_data'):
            delattr(self, 'columns_data')
        if hasattr(self, 'columns_descriptions'):
            delattr(self, 'columns_descriptions')
        
        # Reload from files
        self._load_schemas_from_files(tables_file, columns_file, columns_desc_file)
    
    def _setup_prompts(self):
        """Setup LangChain prompts for SQL generation"""
        
        # Single query prompt
        self.single_query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query generator. Your task is to convert natural language questions into precise SQL queries.

IMPORTANT GUIDELINES:
1. Only return SQL queries, no explanations or markdown formatting
2. Use DISTINCT when the question asks for "all", "find all", or "list all" to avoid duplicates
3. Only select relevant columns that answer the specific question
4. Use appropriate WHERE clauses to filter for relevant rows
5. Make the query as human-readable and efficient as possible
6. Use proper SQL syntax and formatting
7. Make sure that the questions are not changed or altered in any way

Table Schema Information:
{table_info}

Column Information (if available):
{column_info}"""),
            ("human", "Convert this natural language question to SQL: {question}")
        ])
        
        # Multiple queries prompt
        self.multiple_queries_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query generator. Your task is to convert multiple natural language questions into precise SQL queries.

IMPORTANT GUIDELINES:
1. Return a JSON array where each element is an object with "question" and "sql" keys
2. Each SQL query should correspond to each input question in order
3. Use DISTINCT when the question asks for "all", "find all", or "list all" to avoid duplicates
4. Only select the required columns that answer the specific question
5. Use appropriate WHERE clauses to filter for relevant rows
6. Make queries as human-readable and efficient as possible
7. All the questions are independent and should be answered separately
8. Use proper SQL syntax and formatting
9. Make sure that the questions are not changed or altered in any way
10. Make the tables and the columns as descriptive as possible
11. Use the tables that are in the table information and the table names original are the columns present in the actual database

Table Information:
{table_info}

Column Information (if available):
{column_info}


Expected output format:
[{{"question":"First question","sql":"SELECT DISTINCT ..."}},{{"question":"Second question","sql":"SELECT ..."}}, ...]"""),
            ("human", "Convert these natural language questions to SQL queries: {questions}")
        ])

    def generate_sql(self, question: Union[str, List[str]], use_columns: bool = True, use_descriptions: bool = True) -> List[Dict[str, str]]:
        """
        Generate SQL query/queries from natural language question(s)
        
        Args:
            question: Single question string or list of questions
            use_columns: Whether to include column information in prompts
            use_descriptions: Whether to include column descriptions in prompts
            
        Returns:
            List of dictionaries with 'question' and 'sql' keys
            Format: [{"question": <human question>, "sql": <generated sql query>}]
        """
        # Check if we have schema data (either raw JSON data or parsed table schemas)
        has_schema_data = (hasattr(self, 'tables_data') and self.tables_data) or self.table_schemas
        if not has_schema_data:
            raise ValueError("No table schemas provided. Use add_table_schema() or ensure schema files are loaded.")
        
        table_info = self.tables_data
        # Use descriptions based on the flag
        column_info = self.columns_descriptions if use_descriptions and hasattr(self, 'columns_descriptions') else self.columns_data
        if not use_columns:
            # If not using columns, just pass empty info
            column_info = {}

        # Create parser instance
        parser = SQLOutputParser()
        
        # Handle list input - multiple questions
        if isinstance(question, list):
            questions_str = "\n".join([f"{q}" for i, q in enumerate(question)])

            chain = self.multiple_queries_prompt | self.llm
            
            llm_result = chain.invoke({
                "questions": questions_str,
                "table_info": table_info,
                "column_info": column_info
            })
            llm_result = llm_result.content if hasattr(llm_result, 'content') else llm_result
            # Parse the result with questions context
            result = parser.parse(llm_result, questions=question)
            return result
        
        # Handle single string input
        else:
            chain = self.single_query_prompt | self.llm
            
            llm_result = chain.invoke({
                "question": question,
                "table_info": table_info,
                "column_info": column_info
            })
            llm_result = llm_result.content if hasattr(llm_result, 'content') else llm_result
            # Parse the result with questions context
            result = parser.parse(llm_result, questions=question)
            return result
    
    def is_available(self) -> bool:
        """Check if the LLM agent is available"""
        return self.llm_agent.is_available()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
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
        string_name = f"{args.provider}_{args.model}_{args.descriptive}_{args.use_column}.json"
        with open("datasets/" + string_name, "w") as f:
            json.dump(result, f, indent=2)

    return 0


if __name__ == "__main__":
    main()