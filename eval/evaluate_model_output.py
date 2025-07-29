import json
import os
from eval.execution_metrics import ExecutionMetrics
import argparse
import pandas as pd

def get_gt_data():
    """Function to get data from the database."""
    with open("datasets/total_interest.json", "r") as f:
        data = json.load(f)
    return data

def get_model_predictions(path = "datasets/openai_4o_False_False.json"):
    """Function to get model predictions."""
    with open(path, "r") as f:
        data = json.load(f)
    return data

def combine_all(path="datasets/openai_4o_False_False.json"):
    gt_data = get_gt_data()
    model_predictions = get_model_predictions(path=path)
    for item in gt_data:
        question = item["question"].lower().rstrip()
        item["model_prediction"] = None
        for model in model_predictions:
            if model["question"].lower().rstrip().lstrip() == question:
                item["model_prediction"] = model["sql"]
                break
        if not item.get("model_prediction"):
            print(f"No prediction found for question: {question}")
    return gt_data

def get_comparison(path="datasets/openai_4o_False_False.json"):
    """Function to get the comparison of ground truth and model predictions."""
    combined_data = combine_all(path=path)
    predictions = [item["model_prediction"] for item in combined_data]
    ground_truths = [item["query"] for item in combined_data]
    metric_calc = ExecutionMetrics()
    results, executables = metric_calc.compare_results(predictions, ground_truths, None, return_executables=True)
    print("Accuracy of model predictions:", sum(results) / len(results))
    # breakpoint()  
    df = pd.DataFrame(combined_data)
    df["model_prediction"] = predictions
    df["executable"] = executables
    df["correct"] = results
    df.to_csv(f"{path.replace('.json', '_comparison.csv')}", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth.")
    parser.add_argument(
        '--path',
        type=str,
        default="datasets/openai_4o_False_False.json",
        help='Path to the model predictions file (default: datasets/openai_4o_False_False.json)'
    )
    args = parser.parse_args()
    if os.path.isdir(args.path):
        file_paths = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.json')]
        for file_path in file_paths:
            comparison_file = file_path.replace('.json', '_comparison.csv')
            if os.path.exists(comparison_file):
                print(f"Comparison file already exists for {file_path}, skipping...")
                continue
            print(f"Evaluating {file_path}...")
            get_comparison(path=file_path)
    else:
        print(f"Evaluating {args.path}...")
        get_comparison(path=args.path)