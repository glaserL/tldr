import mlflow

import src.tldr.tracking.optuna as optuna
from tldr.generate.serialization import (
    available_serializers,
    get_serializer,
)


def suggest_categorical(trial: optuna.Trial, name, values, log_to_mlflow=True):
    suggestions = trial.suggest_categorical(name, values)
    if log_to_mlflow:
        mlflow.log_param(name, suggestions)
    return suggestions


def suggest_serializer(trial: optuna.Trial, log_to_mlflow=True):
    values = available_serializers()
    serializer = suggest_categorical(trial, "serializer", values, log_to_mlflow)
    return get_serializer(serializer)
