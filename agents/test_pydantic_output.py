#!/usr/bin/env python3
"""
Test script to verify the Pydantic output format works correctly
"""

import json
from agents.agent import SQLResult, SQLOutputList

def test_pydantic_models():
    """Test the Pydantic models"""
    
    # Test individual SQLResult
    sql_result = SQLResult(
        question="What are all the project titles?",
        sql="SELECT DISTINCT title FROM projects"
    )
    print("SQLResult:", sql_result.model_dump())
    
    # Test SQLOutputList
    output_list = SQLOutputList(outputs=[
        SQLResult(question="What are all the project titles?", sql="SELECT DISTINCT title FROM projects"),
        SQLResult(question="How many projects are there?", sql="SELECT COUNT(*) FROM projects")
    ])
    
    print("\nSQLOutputList:", output_list.model_dump())
    print("\nAs dict list:", output_list.to_dict_list())
    
    # Test JSON serialization
    json_output = output_list.model_dump_json()
    print("\nJSON output:", json_output)
    
    # Test JSON parsing back
    parsed = SQLOutputList.model_validate_json(json_output)
    print("\nParsed back:", parsed.to_dict_list())
    
    # Test the expected format
    expected_format = {
        "outputs": [
            {"question": "First question", "sql": "SELECT DISTINCT ..."},
            {"question": "Second question", "sql": "SELECT ..."}
        ]
    }
    
    expected_output = SQLOutputList(**expected_format)
    print("\nExpected format test:", expected_output.to_dict_list())

if __name__ == "__main__":
    test_pydantic_models()
