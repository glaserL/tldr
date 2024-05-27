import logging
from pathlib import Path

from mlflow.entities.model_registry import ModelVersion
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
)

logger = logging.getLogger(__name__)


def strip_mlflow_prefix(uri: str):
    return uri.split(":")[-1][1:]


def load_model(path: Path):
    model = AutoModelForTokenClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.tokenizer = tokenizer
    return model


def load_model_from_mlflow(
    model: ModelVersion,
    local_path_to_artifacts="/home/lglaser/serve/mlflow/mlartifacts",
):
    full_path = Path(local_path_to_artifacts) / strip_mlflow_prefix(model.source)
    full_path = [x for x in full_path.glob("artifacts/checkpoint-*")]
    full_path = full_path[0]
    return load_model(full_path)

def load_trainer_from_path(path: Path):

    model = AutoModelForTokenClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=collator)

    return trainer

def load_trainer_via_model_info(
        
    model: ModelVersion,
    local_path_to_artifacts="/home/lglaser/serve/mlflow/mlartifacts"
):
    full_path = Path(local_path_to_artifacts) / strip_mlflow_prefix(model.source)
    full_path = [x for x in full_path.glob("artifacts/checkpoint-*")]
    full_path = full_path[0]
    return load_trainer_from_path(full_path)


def load_trainer_from_run(
    run, local_path_to_artifacts="/home/lglaser/serve/mlflow/mlartifacts/"
):
    full_path = (
        Path(local_path_to_artifacts)
        / strip_mlflow_prefix(run.info.artifact_uri)
        / run.data.params["output_dir"]
    )
    full_path = [x for x in full_path.glob("checkpoint-*/artifacts/checkpoint-*")][0]
    trainer = load_trainer_from_path(full_path)
    logger.info(f"Loaded previous state from {run.info.run_name}")
    return trainer

def print_sample_as_table(sample):
    cols = {k: len(v) for k, v in sample.items() if isinstance(v, list)}

    print("\t".join(f"{k}:{v}" for k, v in sample.items() if not isinstance(v, list))) # noqa: T201
    longest = max(cols.values())
    col_names = sorted(cols.keys())
    print("\t".join(col_names)) # noqa: T201
    for i in range(longest):
        row = []
        for col in col_names:
            v = sample[col]
            if isinstance(v, list) and len(v) > i:
                val = v[i]
            else:
                val = "分"

            row.append(val)
        print("\t\t".join([str(r) for r in row])) # noqa: T201

