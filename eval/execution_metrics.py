"""
Execution metrics for evaluating Text-to-SQL models.
This module provides functionality to compare the different execution results.
"""

from database_connector.connect_sql import Connector
from utils.metrics import execution_match

class ExecutionMetrics:
    def __init__(self, db_path: str = "database_connector/.env"):
        self.connector = Connector(path=db_path)

    def get_execution_result(self, query: str):
        """Executes a SQL query and returns the result."""
        return self.connector.get_query(query)

    def compare_results(self, predictions: list, ground_truths: list, fully_comparable: list) -> bool:
        """Compares two execution results for equality."""
        if fully_comparable is None:
            fully_comparable = [True]* len(predictions)
        if len(predictions) != len(ground_truths) or len(predictions) != len(fully_comparable):
            raise ValueError("Predictions, ground truths, and fully_comparable must have the same length.")

        correct = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            executable = True
            try:
                prediction_result = self.get_execution_result(prediction)
                ground_truth_result = self.get_execution_result(ground_truth)
                prediction_result = [row._asdict() for row in prediction_result]
                ground_truth_result = [row._asdict() for row in ground_truth_result]
            except:
                executable = False
            if not executable:
                correct.append(False)
                continue
            try:
                if execution_match(prediction_result, ground_truth_result):
                    correct.append(True)
                else:
                    correct.append(False)
            except Exception as e:
                print(f"Error comparing results: {e}")
                correct.append(False)
            # breakpoint()
        return correct