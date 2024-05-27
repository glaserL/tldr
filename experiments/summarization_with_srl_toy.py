import optuna

from tldr.data import make_path
from tldr.data.alice_and_bob import get_chat_gpt_summa
from tldr.generate.main import Generator
from tldr.optuna import suggest_serializer
from tldr.srl.inference import infer
from tldr.summarize.eval import evaluate_summarization
from tldr.tracking.main import (
    setup_mlflow_experiment,
)
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


from datasets import Dataset, load_from_disk


def objective(trial: optuna.Trial) -> float:

    dataset = [
        {"article": sent, "id": 0, "sent_id": i}
        for i, sent in enumerate(text.split("\n"))
    ]
    highlight = get_chat_gpt_summa()
    dataset = Dataset.from_list(dataset)

    pred_run = get_run_by_run_name("traveling-koi-720")
    args_run = get_run_by_run_name("thoughtful-lynx-568")
    dataset_name = "alice_and_bob"

    data_path = make_path(dataset_name, args_run.info.run_name, pred_run.info.run_name)
    if data_path.exists():
        dataset = load_from_disk(str(data_path))
    else:
        dataset = infer(dataset, data_path, pred_run, args_run, False)

    dataset = dataset["test"]

    prompt = "Generate a sentence from the following semantic parse:"
    generator = Generator(
        "mistralai/Mistral-7B-Instruct-v0.2",
        0.6,
        False,
        True,
        False,
        56,
        prompt,
        44,
        0.6,
        0.02,
    )
    serializer = suggest_serializer(trial)
    result = generator.generate(
        dataset, lambda x: {"data": serializer(x)}, prompt, False
    )
    result = " ".join(result)
    rouge = evaluate_summarization(result, highlight, True)

    return rouge["rougeLsum"]


def main():
    from pathlib import Path

    experiment_name = Path(__file__).stem

    setup_mlflow_experiment(experiment_name)

    study = optuna.create_study(study_name=experiment_name, direction="maximize")
    study.optimize(
        objective, n_trials=3, callbacks=[champion_callback], show_progress_bar=True
    )


if __name__ == "__main__":

    main()
