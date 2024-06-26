import mlflow
import optuna
from datasets import Dataset, load_from_disk

from tldr.data import make_path
from tldr.data.alice_and_bob import get_chat_gpt_summa
from tldr.data.prep import prepare_cnn_corpus
from tldr.generate.main import Generator
from tldr.srl.inference import infer
from tldr.summarize.data import load_cnn
from tldr.summarize.eval import evaluate_summarization
from tldr.summarize.main import Acklington, get_summarizer_new_type_beat
from tldr.tracking.main import (
    newest_srl_parser,
    setup_mlflow_experiment,
)
from tldr.tracking.optuna import suggest_serializer
from tldr.tracking.query import get_run_by_run_name
from tldr.tracking.training import champion_callback

text = """Last week, Alice bought a new car.
The car cost 30.000$, a fairly large sum.
Bob's car only cost 20.000$.
Alice bought the car at the Ford Dealership in Singapore. 
Bob bought his Toyota in Tokyo, before returning home.
The Toyota Corolla has a bright red color and 5 wheels.
After seeing Bob's new Corolla, Alice regretted buying such an expensive Ford."""
DEBUG = True


def objective(trial: optuna.Trial) -> float:
    with mlflow.start_run(log_system_metrics=True):

        dataset = [
            {"article": sent, "id": 0, "sent_id": i}
            for i, sent in enumerate(text.split("\n"))
        ]
        highlight = get_chat_gpt_summa()  # FIXME: replace with 'real' data
        dataset = Dataset.from_list(dataset)

        mlflow.log_param("srl_parser", "champion")
        srl_parser = newest_srl_parser()

        dataset_name = "alice_and_bob"

        data_path = make_path(dataset_name, "champion", "coool")
        if data_path.exists():
            dataset = load_from_disk(str(data_path))
        else:
            data_path = prepare_cnn_corpus(log_to_mlflow=True)
            dataset = load_from_disk(str(data_path))
            dataset = dataset["test"]
            dataset = srl_parser.parse_dataset(dataset)

        dataset = dataset["test"]

        mlflow.log_param("summarizer", "acklington")
        k = trial.suggest_int("k", 3, 8)
        mlflow.log_param("k", k)

        summarizer = Acklington(k)
        dataset = summarizer.summarize(dataset)

        generator = Generator.default()
        serializer = suggest_serializer(trial)
        result = generator.generate(dataset, serializer, None, False)

        rouge = evaluate_summarization(result, highlight, True)

    return rouge["rougeLsum"]


def main():
    experiment_name = "Summarization_debugging"

    setup_mlflow_experiment(experiment_name)

    study = optuna.create_study(study_name=experiment_name, direction="maximize")
    study.optimize(
        objective, n_trials=3, callbacks=[champion_callback], show_progress_bar=True
    )


if __name__ == "__main__":

    main()
