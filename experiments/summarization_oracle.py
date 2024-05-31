import mlflow

from tldr.summarize.data import load_cnn
from tldr.summarize.eval import evaluate_summarization
from tldr.tracking.main import setup_mlflow_experiment

setup_mlflow_experiment("summarize/no-op")

with mlflow.start_run(log_system_metrics=True):
    # summarization_datasets = ...
    dataset = load_cnn(["highlights"], True)

    evaluate_summarization(dataset["test"]["highlights"], dataset["test"]["highlights"])
