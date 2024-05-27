from typing import Tuple

import logging
from enum import Enum

from optuna import Trial

from tldr.generate.config import GenerationParameters, PromptConfig

logger = logging.getLogger(__name__)


class Strategy(Enum):
    CONTRASTIVE_SEARCH = 1
    GREEDY_DECODING = 2
    MULTINOMINAL_SAMPLING = 3
    BEAM_SEARCH = 4
    BEAM_SEARCH_MULTINOMINAL = 5


def suggest_hyperparameters(trial: Trial) -> Tuple[GenerationParameters, PromptConfig]:
    # Learning rate on a logarithmic scale

    technique = trial.suggest_categorical("technique", [s.name for s in Strategy])
    technique = Strategy[technique]

    generation_params: GenerationParameters = {
        "do_sample": technique not in [Strategy.GREEDY_DECODING, Strategy.BEAM_SEARCH],
        "max_new_tokens": trial.suggest_int("max_new_tokens", 8, 128, step=4),
    }
    prompt_params: PromptConfig = {
        "add_generation_prompt": trial.suggest_categorical(
            "add_generation_prompt", [True, False]
        ),
        "tokenize": False,
        "prompt": trial.suggest_categorical(
            "prompt",
            [
                "You are a linguistic robot that translates messages in the form of python dictionaries into text. You may only return a single sentence and you can’t use semicolons as part of your answer.",
                "You are a linguistic robot that generates natural language from semantic parses.",
                "Generate a sentence from the following semantic parse:",
                "Generate text from this semantic parse.",
                "Generate text from this semantic parse. It might contain more than one sentence, if so, put each in a new line.",
                "Generate text from this semantic parse. Only output text and do not add any further information or explanation.",
            ],
        ),
    }
    if technique == Strategy.BEAM_SEARCH_MULTINOMINAL:
        generation_params["num_beams"] = trial.suggest_int("num_beams", 2, 5)
    if technique == Strategy.MULTINOMINAL_SAMPLING:
        generation_params["num_beams"] = 1
        generation_params["early_stopping"] = False

    elif technique == Strategy.CONTRASTIVE_SEARCH:
        generation_params["penalty_alpha"] = trial.suggest_float("penality_alpha", 0, 1)

    if generation_params["do_sample"]:

        generation_params["top_k"] = trial.suggest_int("top_k", 2, 64)
        generation_params["temperature"] = trial.suggest_float("temperature", 0.1, 1.4)
        generation_params["top_p"] = trial.suggest_float("top_p", 0.5, 0.999)

    return generation_params, prompt_params
