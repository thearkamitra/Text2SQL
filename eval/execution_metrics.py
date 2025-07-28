"""
Execution metrics for evaluating Text-to-SQL models.
This module provides functionality to compare the different execution results.
"""

import signal
from contextlib import contextmanager
from tqdm import tqdm
from database_connector.connect_sql import Connector
from utils.metrics import execution_match

class ExecutionMetrics:
    def __init__(self, db_path: str = "database_connector/.env"):
        self.connector = Connector(path=db_path)

    @contextmanager
    def timeout(self, seconds):
        """Context manager for timing out operations"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Restore the old signal handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

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
                # Execute with 60 second timeout
                with self.timeout(60):
                    prediction_result = self.get_execution_result(prediction)
                    ground_truth_result = self.get_execution_result(ground_truth)
                    prediction_result = [row._asdict() for row in prediction_result]
                    ground_truth_result = [row._asdict() for row in ground_truth_result]
            except TimeoutError as e:
                executable = False
                progress_bar.set_postfix({"Error": f"Timeout: {str(e)[:30]}..."})
            except Exception as e:
                executable = False
                progress_bar.set_postfix({"Error": f"Execution failed: {str(e)[:30]}..."})

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