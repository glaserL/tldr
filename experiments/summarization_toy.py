import optuna

from tldr.data.alice_and_bob import get_toy_data
from tldr.data.graph import srl2document
from tldr.generate.main import Generator
from tldr.generate.serialization import prune_O
from tldr.optuna import suggest_categorical, suggest_serializer
from tldr.summarize.eval import evaluate_summarization
from tldr.summarize.main import filter_with_text_rank
from tldr.tracking.main import (
    setup_mlflow_experiment,
)
from tldr.tracking.training import champion_callback

text = """Last week, Alice bought a new car.
The car cost 30.000$, a fairly large sum.
Bob's car only cost 20.000$.
Alice bought the car at the Ford Dealership in Singapore. 
Bob bought his Toyota in Tokyo, before returning home.
The Toyota Corolla has a bright red color and 5 wheels.
After seeing Bob's new Corolla, Alice regretted buying such an expensive Ford."""
DEBUG = True
import mlflow


def suggest_hyperparameters(trial: optuna.Trial):
    pass


def objective(trial: optuna.Trial) -> float:
    with mlflow.start_run(log_system_metrics=True):
        document = get_toy_data()
        highlight = document["highlights"]
        document = document["article"]

        document = srl2document(document)
        do_text_rank = trial.suggest_categorical("do_text_rank", [True, False])
        if do_text_rank:
            summarizer_k = trial.suggest_int("summarizer_k", 1, 10)
            mlflow.log_param("summarizer_k", summarizer_k)
            document = filter_with_text_rank(document, summarizer_k)

        do_prune_O = suggest_categorical(trial, "do_prune_O", [True, False])
        # TODO: Log to MLFLOW
        document = apply_transformation(document, prune_O) if do_prune_O else document

        do_remove_stopwords = suggest_categorical(
            trial, "do_remove_stopwords", [True, False]
        )
        document = (
            apply_transformation(document, remove_stopwords)
            if do_remove_stopwords
            else document
        )

        do_remove_empty = suggest_categorical(trial, "do_remove_empty", [True, False])
        document = (
            apply_transformation(document, remove_empty)
            if do_remove_empty
            else document
        )

        generator = Generator.default()
        serializer = suggest_serializer(trial)
        result = generator.generate(document, serializer, None, False)
        result = " ".join(result)
        rouge = evaluate_summarization([result], [highlight], True)

        return rouge["rougeLsum"]


def main():
    from pathlib import Path

    experiment_name = Path(__file__).stem

    setup_mlflow_experiment(experiment_name)

    study = optuna.create_study(study_name=experiment_name, direction="maximize")
    study.optimize(
        objective, n_trials=10, callbacks=[champion_callback], show_progress_bar=True
    )


if __name__ == "__main__":

    main()
