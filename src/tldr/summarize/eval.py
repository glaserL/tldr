import evaluate
import mlflow
from evaluate import EvaluationModule


def evaluate_summarization(predictions, references, log_to_table=False):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    mlflow.log_metrics(results)
    if log_to_table:
        mlflow.log_table(
            {"predictions" : predictions, "references": references},
                          "output.json")
    return results


def log_bleu(bleu: EvaluationModule):
    metrics = bleu.compute()
    precisions = metrics.pop("precisions")
    mlflow.log_metrics(metrics)
    for i, prec in enumerate(precisions):
        mlflow.log_metric(f"precision-{i}", prec)
    return metrics
