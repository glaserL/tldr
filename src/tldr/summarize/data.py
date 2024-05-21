from typing import List, Optional

import mlflow
from datasets import DatasetDict, load_dataset
from mlflow.data.huggingface_dataset import from_huggingface


def load_cnn(data_columns: Optional[List[str]] = None, log_to_mlflow=False):
    if data_columns is None:
        data_columns = ["highlights", "article"]
    data_columns.append("id")
    path = "cnn_dailymail"
    revision = "3.0.0"
    dataset: DatasetDict = load_dataset(path, revision)

    dataset = DatasetDict(
        {split: ds.select_columns(data_columns) for split, ds in dataset.items()}
    )
    dataset.rename_column("article", "text")
    if not log_to_mlflow:
        return dataset

    for split_name in dataset.keys():
        hf_ds = from_huggingface(
            dataset[split_name],
            path,
            "highlights" if "highlights" in data_columns else None,
            revision=revision,
            name="cnn_dailymail",
        )
        mlflow.log_input(hf_ds, split_name, {"task": "summarization"})
    return dataset
