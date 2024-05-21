from typing import List, Optional

import logging

import mlflow
from mlflow.entities import Run

logger = logging.getLogger(__name__)


def get_generation_configs_from_run(run: Run):

    artifact_uri = run.info.artifact_uri

    generation_config = mlflow.artifacts.load_dict(
        artifact_uri + "/generation_config.json"
    )
    prompt_config = mlflow.artifacts.load_dict(artifact_uri + "/prompt_config.json")

    return run.data.params["model"], generation_config, prompt_config


def search_runs(filter_string, experiment_names, search_all_experiments):

    runs = mlflow.search_runs(  # TODO: DROPPY TABLE??
        filter_string=filter_string,
        experiment_names=experiment_names,
        search_all_experiments=search_all_experiments,
        max_results=1,
        output_format="list",
    )
    return runs


def get_runs_by_tag(
    tag_name: str, tag_value, experiment_names: Optional[List[str]] = None
):
    logger.debug(f"Searching tag {tag_name} = {tag_value}")

    search_all_experiments = experiment_names is None

    bad_runs = search_runs(
        f"tags.{tag_name} = '{tag_value}'", experiment_names, search_all_experiments
    )

    return bad_runs


def get_runs_by_task(task: str, experiment_names: Optional[List[str]] = None):
    return get_runs_by_tag("task", task, experiment_names)


def get_run_by_run_name(
    run_name: str, experiment_names: Optional[List[str]] = None
) -> Run:
    search_all_experiments = experiment_names is None

    logger.debug(f"Searching run.name={run_name}")
    bad_runs = search_runs(
        f'attributes.run_name = "{run_name}"', experiment_names, search_all_experiments
    )

    return bad_runs[0]


def get_model(task):
    client = mlflow.MlflowClient()
    models = client.search_model_versions(f"name='{task}'")
    if not len(models):
        raise TypeError(f"No registered model for task {task}.")
    return models[0]
