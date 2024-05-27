MODELS = [
    "distilbert/distilbert-base-uncased",
    "distilbert/distilbert-base-cased",
    # "facebook/bart-large", its a bart not bert
    "albert/albert-base-v2",
    "bert-base-cased",
    # "martincc98/srl_bert_advanced", Probably risky
    "prajjwal1/bert-tiny",
    # "YituTech/conv-bert-base", # Broken Vocab
    "google/mobilebert-uncased",
]


from argparse import ArgumentParser

import mlflow
import optuna
import torch
from transformers import (
    TrainingArguments,
)

from tldr.srl.train import CustomArguments, run
from tldr.tracking.training import mark_as_champion


def suggest_args(trial: optuna.Trial, task: str):
    model = trial.suggest_categorical("model", MODELS)
    torch.cuda.empty_cache()
    debug = False
    inflate_training_data = False
    batch_size = (
        32
        if model
        in ["albert/albert-base-v2", "bert-base-cased", "google/mobilebert-uncased"]
        else 64
    )
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-05])
    weight_decay = trial.suggest_categorical("weight_decay", [0, 0.1])
    lr_scheduler_type = trial.suggest_categorical(
        "lr_scheduler_type", ["linear", "polynomial"]
    )
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000, step=100)
    early_stopping_patience = trial.suggest_int("early_stopping_patience", 3, 10)
    corpus = ["/home/lglaser/Developer/shallow-srl/data/export"]
    custom_args = CustomArguments(
        task=task,
        corpus=corpus,
        debug=debug,
        inflate_training_data=inflate_training_data,
        base_model=model,
        early_stopping_patience=early_stopping_patience,
        verbose=False,
    )
    training_args = TrainingArguments(
        "output",
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5000,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        optim="adamw_torch",
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        max_grad_norm=0,
        num_train_epochs=10,
        load_best_model_at_end=True,
    )
    return training_args, custom_args


def suggest_training_args(trial: optuna.Trial):

    return


def objective(trial, task) -> float:
    training_args, custom_args = suggest_args(trial, task)
    return run(training_args, custom_args)


def main(task: str):
    study = optuna.create_study(
        study_name="srl_training", directions=["minimize", "maximize"]
    )
    study.optimize(
        lambda trial: objective(trial, task), n_trials=5, show_progress_bar=True
    )

    for trial in study.best_trials:

        mark_as_champion(trial)

    img = optuna.visualization.plot_pareto_front(study)
    mlflow.log_figure("pareto.png", img)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    main(args.task)
