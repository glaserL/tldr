from pathlib import Path

GLOBAL_DATASET_PREFIX = Path.home() / "Developer" / "text_summarization" / "storage"

def make_path(dataset_name, args_run_name, pred_run_name):
    return GLOBAL_DATASET_PREFIX / dataset_name / f"args-{args_run_name}_pred-{pred_run_name}"

