from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypedDict,
)

import json
import logging
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Split, Value
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.dataset import Dataset as MLFlowDataset
from mlflow.data.dataset_source import DatasetSource
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)



def yield_conll(file):
    with open(file, "r") as input:
        buffer = ""
        for line in input:
            line = line.strip()
            if line == "":
                yield buffer
                buffer = ""
            else:
                buffer += line + "\n"
        yield buffer


def write_to_path(data: List[str], file: Path):
    with open(file, "w") as f:
        f.writelines(data)


def manual_train_test_split(
    file: Path,
) -> Tuple[Iterator[str], Iterator[str], Iterator[str]]:
    logger.info(f"Creating manual split for {file}..")
    conll = list(yield_conll(file))

    train, validation = train_test_split(conll, test_size=0.2, train_size=0.8)
    test, validation = train_test_split(validation, test_size=0.4, train_size=0.6)

    return train, validation, test



def get_git_revision_short_hash(path) -> str:

    result = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=path)

    return result.decode("utf8")


def create_feature_description(all_labels):
    labels = []
    labels = all_labels
    # for label in all_labels:
    #     labels.extend(generate_iobes(label))
    labels = ClassLabel(names=labels)

    labels = Sequence(feature=labels)

    features = Features({"text": Sequence(Value("string")), "labels": labels})

    return features
    # Features()
    # Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),




class LoggerDataset(MLFlowDataset):

    def __init__(self, source: DatasetSource, labels, path: Path, data=None):
        name = path.parent.name
        digest = get_git_revision_short_hash(path.parent)
        self._features = list(labels)
        self._profile = {}
        if data is not None:
            self._profile["num_sents"] = len(data)
            self._profile["num_tokens"] = sum(len(sent["text"]) for sent in data)

        super().__init__(source, name, digest)

    # def _compute_digest(self) -> str:
    #     hashable = (self._features, self.name, self._profile)
    #     hasher = hashlib.sha1(str(hashable).encode("utf-8"))
    #     return str(hasher.digest()[:8])

    def to_dict(self) -> Dict[str, str]:
        """Create config dictionary for the dataset.

        Returns a string dictionary containing the following fields: 
        name, digest, source, source, type, schema, and profile.
        """
        schema = json.dumps(self.schema.to_dict()) if self.schema else None
        config = super().to_dict()
        config.update(
            {
                "schema": schema,
                "profile": json.dumps(self.profile),
            }
        )
        return config

    @property
    def profile(self) -> Optional[Any]:
        """
        Modified from NumpyDataset
        """
        out = self._features[:]
        out.extend(f"{k}: {v}" for k, v in self._profile.items())
        return out


class SimpleDataset(TypedDict):
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    label_inventory: List[str]


def collect_paths(corpus):
    result = []

    while len(corpus) > 0:
        file = corpus.pop()
        if os.path.isdir(file):
            corpus.extend([os.path.join(file, subfile) for subfile in os.listdir(file)])
        else:
            result.append(file)

    return result
    # display.total = display.total + len(subfiles)


def generate_raw_with_progress(
    corpus,
) -> Iterator[Tuple[str, Path, Iterator[str]]]:  # -> Generator[Tuple[str, List[str]]]:
    display = tqdm(total=len(corpus))

    while len(corpus) > 0:
        file = corpus.pop()
        if os.path.isdir(file):
            subfiles = [os.path.join(file, subfile) for subfile in os.listdir(file)]
            display.refresh()
            corpus.extend(subfiles)
            display.update(1)
            continue

        file = Path(file)
        if "dev" in file.name or "validation" in file.name:

            yield "validation",file, yield_conll(file)
        elif "test" in file.name:
            yield "test", file, yield_conll(file)
        elif "train" in file.name:
            yield "train", file, yield_conll(file)

        elif "conll" in file.suffix:

            logger.info(
                f"No split name in {file} but it looks like conll. Trying manual split."
                )
            train, validation, test = manual_train_test_split(file)
            yield "train", file, train
            yield "test", file,  test
            yield "validation", file,  validation
        display.update(1)


def read_dataset(
    file2data: Callable[[Path], Tuple[Iterable[str], List[Dict[str, Any]]]],
    corpus,
    inflate_training_data=False,
    log_to_mlflow=True,
) -> SimpleDataset:
    tmp = {"train": [], "validation": [], "test": [], "label_inventory": []}
    corpus = collect_paths(corpus)
    labels = set()
    for splt, file, data in generate_raw_with_progress(corpus):

        added_labels, data = file2data(data)
        labels.update(added_labels)

        tmp[splt].extend(data)

        if log_to_mlflow:

            source = CodeDatasetSource({"task": file2data.__name__})

            dataset = LoggerDataset(source, added_labels, file, data)
            mlflow.log_input(dataset, context=splt)


    labels = sorted(set(labels))
    return {
        "validation": pd.DataFrame(tmp["validation"]),
        "test": pd.DataFrame(tmp["test"]),
        "train": pd.DataFrame(tmp["train"]),
        "label_inventory": labels,
    }


def read_huggingface_dataset(
    file2data: Callable[[Path], Tuple[Iterable[str], Iterable[Dict[str, Any]]]],
    corpus,
    to_nummern_labels=True,
    inflate_training_data=False,
    log_to_mlflow=True,
) -> DatasetDict:
    data = read_dataset(file2data, corpus, inflate_training_data, log_to_mlflow)
    other = DatasetDict()
    if to_nummern_labels:
        mapping = {
            label: i for i, label in enumerate(sorted(set(data["label_inventory"])))
        }
        features = create_feature_description(data["label_inventory"])
    else:
        features = None
    other["train"] = Dataset.from_pandas(
        data["train"], split=Split.TRAIN, features=features
    )

    other["test"] = Dataset.from_pandas(
        data["test"], split=Split.TEST, features=features
    )
    other["validation"] = Dataset.from_pandas(
        data["validation"], split=Split.VALIDATION, features=features
    )
    if to_nummern_labels:
        other.align_labels_with_mapping(mapping, "labels")
    return other
