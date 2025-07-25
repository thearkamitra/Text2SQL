import json
from eval.execution_metrics import ExecutionMetrics

def get_gt_data():
    """Function to get data from the database."""
    with open("datasets/total_interest.json", "r") as f:
        data = json.load(f)
    return data

def get_model_predictions(path = "datasets/all_queries_openai_with_updated_desc.json"):
    """Function to get model predictions."""
    with open(path, "r") as f:
        data = json.load(f)
    return data

def combine_all():
    gt_data = get_gt_data()
    model_predictions = get_model_predictions()
    for item in gt_data:
        question = item["question"].lower().rstrip()
        for model in model_predictions:
            if model["question"].lower().rstrip() == question:
                item["model_prediction"] = model["sql"]
                break
        if not item.get("model_prediction"):
            print(f"No prediction found for question: {question}")
    return gt_data

def get_comparison():
    """Function to get the comparison of ground truth and model predictions."""
    combined_data = combine_all()
    predictions = [item["model_prediction"] for item in combined_data]
    ground_truths = [item["query"] for item in combined_data]
    metric_calc = ExecutionMetrics()
    results = metric_calc.compare_results(predictions, ground_truths, None)
    print("Accuracy of model predictions:", sum(results) / len(results))
    
if __name__ == "__main__":
    get_comparison()