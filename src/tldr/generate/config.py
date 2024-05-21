import json
from importlib import resources
from pathlib import Path

from typing_extensions import NotRequired, TypedDict

from . import data


class GenerationParameters(TypedDict):
    do_sample: bool
    max_new_tokens: int
    early_stopping: NotRequired[bool]
    num_beams: NotRequired[int]
    top_k: NotRequired[int]
    top_p: NotRequired[float]
    temperature: NotRequired[float]
    penalty_alpha: NotRequired[float]


class PromptConfig(TypedDict):
    prompt: str
    tokenize: bool
    add_generation_prompt: bool


class GenerationConfig(TypedDict):
    generation_parameters: GenerationParameters
    prompt_config: PromptConfig
    model_name: str


def get_default_params() -> GenerationConfig:
    path = resources.files(data) / "default_config.json"

    with open(path) as f:
        return json.load(f)


HERE = Path(__file__).parent


def get_access_token():
    with open(HERE.parent.parent.parent / ".hftoken") as f:
        return f.readlines()[0]
