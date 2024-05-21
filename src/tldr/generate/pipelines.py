from typing import Callable, Dict, List, Tuple

import torch
from transformers import TextGenerationPipeline, pipeline

from tldr.generate.config import get_access_token


def init_mistral():

    pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto",
        token=get_access_token(),
    )

    def template_function(prompt: str, data: str):
        return [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "Understood. Please input the python dictionary.",
            },
            {"role": "user", "content": data},
        ]

    return pipe, template_function


def init_tinyllama() -> Tuple[TextGenerationPipeline, Callable[[str, str], List[Dict]]]:

    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    def template_function(prompt: str, data: str):
        return [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": str(data)},
        ]

    return pipe, template_function


def get_pipeline(model_name: str):
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        return init_tinyllama()
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.2":
        return init_mistral()
    else:
        raise KeyError(f"Unsupported model: {model_name}")
