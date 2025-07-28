"""
Execution metrics for evaluating Text-to-SQL models.
This module provides functionality to compare the different execution results.
"""

import signal
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from tqdm import tqdm
from database_connector.connect_sql import Connector
from utils.metrics import execution_match

class ExecutionMetrics:
    def __init__(self, db_path: str = "database_connector/.env"):
        self.connector = Connector(path=db_path)

    def execute_with_timeout(self, prediction, ground_truth, timeout_seconds=4):
        """Execute both queries with timeout using ThreadPoolExecutor"""
        def execute_queries():
            prediction_result = self.get_execution_result(prediction)
            ground_truth_result = self.get_execution_result(ground_truth)
            prediction_result = [row._asdict() for row in prediction_result]
            ground_truth_result = [row._asdict() for row in ground_truth_result]
            return prediction_result, ground_truth_result
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(execute_queries)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                raise TimeoutError(f"Query execution timed out after {timeout_seconds} seconds")

    def get_execution_result(self, query: str):
        """Executes a SQL query and returns the result."""
        return self.connector.get_query(query)

    def compare_results(self, predictions: list, ground_truths: list, fully_comparable: list, return_executables: bool = False) -> bool:
        """Compares two execution results for equality."""
        if fully_comparable is None:
            fully_comparable = [True]* len(predictions)
        if len(predictions) != len(ground_truths) or len(predictions) != len(fully_comparable):
            raise ValueError("Predictions, ground truths, and fully_comparable must have the same length.")

        correct = []
        executables = []
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(
            zip(predictions, ground_truths), 
            total=len(predictions),
            desc="Comparing SQL results",
            unit="query"
        )
        
        for prediction, ground_truth in progress_bar:
            executable = True
            try:
                # Execute with timeout using ThreadPoolExecutor
                prediction_result, ground_truth_result = self.execute_with_timeout(
                    prediction, ground_truth, timeout_seconds=4
                )
            except TimeoutError as e:
                executable = False
                progress_bar.set_postfix({"Status": "Timeout"})
            except Exception as e:
                executable = False
                progress_bar.set_postfix({"Status": f"Error: {str(e)[:20]}..."})

            if not executable:
                correct.append(False)
                executables.append(False)
                continue
                
            executables.append(True)
            try:
                if execution_match(prediction_result, ground_truth_result):
                    correct.append(True)
                else:
                    correct.append(False)
            except Exception as e:
                print(f"Error comparing results: {e}")
                correct.append(False)
            
            # Update progress bar with current stats
            current_correct = sum(correct)
            current_executable = sum(executables)
            total_processed = len(correct)
            
            progress_bar.set_postfix({
                "Correct": f"{current_correct}/{total_processed}",
                "Executable": f"{current_executable}/{total_processed}",
                "Accuracy": f"{current_correct/total_processed*100:.1f}%" if total_processed > 0 else "0%"
            })
            
        # Final summary
        total_correct = sum(correct)
        total_executable = sum(executables)
        total_queries = len(predictions)
        
        print(f"\n📊 Final Results:")
        print(f"   Total queries: {total_queries}")
        print(f"   Executable: {total_executable} ({total_executable/total_queries*100:.1f}%)")
        print(f"   Correct: {total_correct} ({total_correct/total_queries*100:.1f}%)")
        print(f"   Accuracy (of executable): {total_correct/total_executable*100:.1f}%" if total_executable > 0 else "   No executable queries")
            
        if return_executables:
            return correct, executables
        
        return correct