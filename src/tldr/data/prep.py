from typing import List

import logging
from collections import defaultdict

import mlflow
import pandas as pd
from datasets import Dataset, DatasetDict
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from tldr.data import GLOBAL_DATASET_PREFIX
from tldr.summarize.data import load_cnn

logger = logging.getLogger()


def split_sentences(dataset, cols_to_split: List[str], sentence_split_func):
    tmp = defaultdict(list)
    progress = tqdm(dataset)

    for split in progress:
        subset = dataset[split]
        data = []
        progress.set_description(f"Splitting {split} split..")
        cols_to_copy = [col for col in subset.column_names if col not in cols_to_split]

        for row in tqdm(subset, leave=False):
            new_row = {col: sentence_split_func(row[col]) for col in cols_to_split}

            update_row = {col: row[col] for col in cols_to_copy}
            new_row.update(update_row)

            data.append(new_row)
        tmp[split] = data
    result = DatasetDict()
    for split_name, split_data in tmp.items():
        result[split_name] = Dataset.from_pandas(
            pd.DataFrame(split_data), split=split_name.upper()
        )
    return result


def prepare_cnn_corpus(overwrite_if_exists=False, log_to_mlflow=False):
    target_folder = GLOBAL_DATASET_PREFIX / "cnn" / "split_only"
    if target_folder.exists() and not overwrite_if_exists:
        logger.info(
            f"{target_folder} found. Set overwrite_if_exists=True to overwrite."
        )
        return target_folder

    if log_to_mlflow:
        mlflow.log_param("Sentence Tokenizer", "nltk.tokenize.sent_tokenize")

    dataset = load_cnn(["highlights"], log_to_mlflow)
    target_folder.mkdir(parents=True, exist_ok=True)

    dataset = split_sentences(dataset, ["highlights"], sent_tokenize)

    logger.info(f"Storing dataset to {target_folder}")

    dataset.save_to_disk(target_folder)
    return target_folder
