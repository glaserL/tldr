# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

from collections import defaultdict
from logging import getLogger

import evaluate
import mlflow
import optuna
import timeout_decorator
from datasets import Dataset

from tldr.data.graph import (
    srl2document,
)
from tldr.generate.hyperparam_util import (
    GenerationParameters,
    PromptConfig,
    suggest_hyperparameters,
)
from tldr.generate.main import generate, template
from tldr.generate.pipelines import get_pipeline
from tldr.generate.serialization import (
    available_serializers,
    get_serializer,
)
from tldr.optuna import suggest_categorical
from tldr.srl.data import read_huggingface_dataset
from tldr.srl.train import extract_args_data
from tldr.summarize.eval import log_bleu
from tldr.summarize.putzfisch import meister_propper
from tldr.tracking.main import (
    setup_mlflow_experiment,
)
from tldr.tracking.training import champion_callback, mark_as_champion

logger = getLogger(__name__)


def run_experiment(
    pipe,
    data: Dataset,
    cool_func,
    generation_config: GenerationParameters,
    prompt_config: PromptConfig,
):
    mlflow.log_params(prompt_config)

    mlflow.log_param("model", pipe.model.name_or_path)
    mlflow.log_param("tokenizer", pipe.tokenizer.name_or_path)
    mlflow.log_params(generation_config)

    dataset = template(data, pipe.tokenizer, cool_func, **prompt_config)
    result = generate(pipe, dataset, generation_config, 16)
    return result


def suggest_serializer(trial: optuna.Trial):

    serializer = suggest_categorical(trial, "serializer", available_serializers())
    return get_serializer(serializer)


import optuna

implemented_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


@timeout_decorator.timeout(
    60 * 60, timeout_exception=optuna.TrialPruned, use_signals=True
)
def objective(trial) -> float:
    with mlflow.start_run(log_system_metrics=True) as run:

        # logging.basicConfig(level=logging.DEBUG) # TODO: Unify logging somewhere
        logger.info("Setting up..")
        trial.set_user_attr("run_id", run.info.run_id)
        generation_config, prompt_config = suggest_hyperparameters(trial)

        ## LOAD DATASET
        dataset = read_huggingface_dataset(
            lambda x: extract_args_data(x, add_sentence_index=True),
            ["../shallow-srl/data/export/ewt"],
            False,
        )

        dataset = dataset if isinstance(dataset, Dataset) else dataset["test"]
        ## TODO: Should you do this for the entire datset? Will probably take too long, no?

        dataset = dataset.rename_column("text", "tokens")
        dataset = dataset.rename_column("labels", "args")

        # dataset = dataset.map(shitty_util, batched=True, desc="Injecting fake sentence ids", batch_size=12)
        logger.info("Loaded dataset.")

        dataset = srl2document(dataset)
        dataset = meister_propper(dataset)

        original = dataset.sentences[:]

        serializer = suggest_serializer(trial)

        serialized = serializer.serialize(dataset)

        logger.info(f"Serialized {len(serialized)}")
        logger.debug(serialized[0])

        generation_config, prompt_config = suggest_hyperparameters(trial)
        model = trial.suggest_categorical("model", implemented_models)

        bleu = evaluate.load("bleu")

        pipe, template_function = get_pipeline(model)

        dataset = Dataset.from_list([{"data": serialized} for serialized in serialized])

        result = run_experiment(
            pipe, dataset, template_function, generation_config, prompt_config
        )

        log_table = defaultdict(list)
        for prediction, orig, data in zip(result, original, serialized):
            reference = " ".join(orig.full_text)

            bleu.add(
                prediction=prediction,
                reference=[reference],
            )
            log_table["data"].append(data)
            log_table["generated_text"].append(prediction)

        mlflow.log_dict(prompt_config, "prompt_config.json")
        mlflow.log_dict(generation_config, "generation_config.json")

        mlflow.log_table(log_table, "output.json")
        metrics = log_bleu(bleu)

        return metrics["bleu"]


def main():
    experiment = setup_mlflow_experiment(
        "generation/regenerate-gold-data",
        "Trying different generation configurations to recreate SRL parses.",
    )
    study = optuna.create_study(
        study_name=experiment.name, direction="maximize", storage="sqlite:///optuna.db"
    )
    study.optimize(
        objective,
        n_trials=50,
        callbacks=[champion_callback],
        show_progress_bar=True,
    )

    trial = study.best_trial
    mark_as_champion(trial)


if __name__ == "__main__":

    main()
