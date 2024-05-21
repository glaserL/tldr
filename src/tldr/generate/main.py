from typing import Callable, Dict, List, Optional, Union

import logging

from datasets import Dataset
from mlflow.entities import Run
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from tldr.generate.config import get_default_params
from tldr.generate.hyperparam_util import GenerationParameters, PromptConfig
from tldr.generate.pipelines import get_pipeline
from tldr.generate.serialization import Serializer
from tldr.tracking.query import get_generation_configs_from_run
from tldr.types import Document

logger = logging.getLogger(__name__)


def generate(pipe, dataset, generation_config, batch_size=None):
    batch_size = batch_size or len(dataset)

    result = []
    for batch in tqdm(
        dataset.iter(batch_size=batch_size), total=len(dataset) / batch_size
    ):

        outputs = pipe(
            batch["formatted_prompt"], return_full_text=False, **generation_config
        )

        for out in outputs:
            out = out[0]

            result.append(out["generated_text"])

    return result


def _template(examples, tokenizer, template_function, prompt, *args, **kwargs):
    examples["formatted_prompt"] = tokenizer.apply_chat_template(
        template_function(prompt, examples["data"]), *args, **kwargs
    )
    return examples


def template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    template_function: Callable[[str, str], List[Dict]],
    prompt: str,
    *args,
    **kwargs,
):

    dataset = dataset.map(
        lambda x: _template(x, tokenizer, template_function, prompt, *args, **kwargs),
        desc="Templating..",
    )
    return dataset


def prepare_data(
    text: Union[Document, List[Document]],
    serializer: Serializer,
) -> Dataset:
    if isinstance(text, Document):
        dataset = serializer.serialize_to_list_of_dicts(text)
    elif isinstance(text, list):

        dataset = [serializer.serialize_to_list_of_dicts(t) for t in text]

        dataset = [item for sublist in dataset for item in sublist]

    dataset = Dataset.from_list(dataset)
    return dataset


class Generator:

    def __init__(
        self,
        model_name,
        top_p,
        add_generation_prompt,
        do_sample,
        tokenize,
        top_k,
        prompt,
        max_new_tokens,
        temperature,
        penalty_alpha,
    ):
        self.pipeline, self.template_function = get_pipeline(model_name)
        self.batch_size = 16
        self.gen_params = {
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "penalty_alpha": penalty_alpha,
        }

        self.tokenize = tokenize
        self.prompt = prompt

        self.add_generation_prompt = add_generation_prompt

    def generate(
        self,
        text: Union[Document, List[Document]],
        serializer: Serializer,
        prompt=None,
        add_generation_prompt=None,
    ):

        prompt = prompt or self.prompt
        add_generation_prompt = add_generation_prompt or self.add_generation_prompt
        dataset = prepare_data(text, serializer)

        dataset = template(
            dataset,
            self.pipeline.tokenizer,
            self.template_function,
            prompt,
            add_generation_prompt=add_generation_prompt,
            tokenize=self.tokenize,
        )

        generated = generate(self.pipeline, dataset, self.gen_params, self.batch_size)

        dataset = dataset.add_column("generated_text", generated)

        return dataset

    @classmethod
    def from_run(cls, run: Run):
        model_name, gen_params, template_params = get_generation_configs_from_run(run)
        return Generator.from_configs(model_name, gen_params, template_params)

    @classmethod
    def from_run_names(cls, run_name: str):
        raise NotImplementedError

    @classmethod
    def from_configs(
        cls,
        model_name: str,
        gen_params: GenerationParameters,
        prompt_config: Optional[PromptConfig],
    ):
        _prompt_config = prompt_config or {}
        return Generator(model_name, **gen_params, **_prompt_config)

    @classmethod
    def default(cls):
        defaults = get_default_params()
        return Generator(
            defaults["model_name"],
            **defaults["prompt_config"],
            **defaults["generation_parameters"],
        )
