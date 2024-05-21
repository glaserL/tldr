from typing import Optional

import logging
import os

import mlflow
from mlflow.entities import Run
from transformers import set_seed

from tldr.srl.inference import SRLParser
from tldr.srl.util import load_trainer_from_run, load_trainer_via_model_info
from tldr.tracking.query import get_model, get_run_by_run_name

TRACKING_URI = "http://localhost:5000"
logger = logging.getLogger(__name__)

SEED = 322


def setup_mlflow_experiment(
    experiment_name: str, experiment_description: Optional[str] = None
):
    experiment = mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(TRACKING_URI)
    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
    if experiment_description is not None:
        mlflow.set_experiment_tag("mlflow.note.content", experiment_description)

    set_seed(SEED)
    logger.info(f"Fixed seed at {SEED}")
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    logger.info(f"Created experiment {experiment} to be logged to {TRACKING_URI}")
    return experiment


class SRLParserFromMLflow(SRLParser):
    @classmethod
    def from_runs(cls, pred_run: Run, args_run: Run):
        logger.info(
            f"Creating SRL Parser from runs PRED[{pred_run}] and ARGS[{args_run}]"
        )
        pred_trainer = load_trainer_from_run(pred_run)
        args_trainer = load_trainer_from_run(args_run)
        return cls(pred_trainer, args_trainer)

    @classmethod
    def from_run_names(cls, pred_run: str, args_run: str):
        return cls.from_runs(
            get_run_by_run_name(pred_run), get_run_by_run_name(args_run)
        )


def newest_srl_parser():
    pred = get_model("SRL-pred")
    args = get_model("SRL-args")
    return SRLParser(
        load_trainer_via_model_info(pred), load_trainer_via_model_info(args)
    )
