from typing import List, Literal, Union

import logging
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from mlflow.entities import Run
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    Trainer,
)

from tldr.srl.util import load_trainer_from_path, load_trainer_from_run

BATCH_SIZE = 128
MAX_SENTENCE_LENGTH = 256

logger = logging.getLogger(__name__)


def flatten(examples, column_to_flatten="text", flatten_column_name=None):
    result = defaultdict(list)

    copy_cols = [col for col in examples.keys() if col != column_to_flatten]
    for i, row in enumerate(examples[column_to_flatten]):
        result[column_to_flatten].extend(row)
        # print(row)
        if (
            flatten_column_name is not None
            and flatten_column_name not in examples.keys()
        ):
            sent_ids = [x for x in range(0, len(row))]
            # print(sent_ids)
            result[flatten_column_name].extend(sent_ids)
        for cp in copy_cols:
            result[cp].extend([examples[cp][i]] * len(row))
    return result


def find_begin_and_end_id(tokenizer):
    begin = tokenizer(">")["input_ids"][1]
    end = tokenizer("<")["input_ids"][1]
    return begin, end


def run_predicate(loader: DataLoader, model, device: str):
    results = []

    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        for pred in predictions:
            result = [p.item() for p in pred]
            results.append(result)

    return results


def tokenize(examples, text_column, tokenizer, *args, **kwargs):

    tokenized_inputs = tokenizer(
        examples[text_column],
        *args,
        **kwargs,
    )
    tokenized_inputs["id"] = examples["id"]
    return tokenized_inputs


def juggle_dataset(dataset: Union[DatasetDict, Dataset, Path]) -> DatasetDict:
    if isinstance(dataset, Path):
        dataset = load_from_disk(str(dataset))

    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"test": dataset})
    elif isinstance(dataset, DatasetDict):
        dataset = dataset
    else:
        raise TypeError(f"Unsupported dataset type {type(dataset)}")

    if "highlights" in dataset.column_names:
        dataset = dataset.remove_columns(["highlights"])
    ##### PREPARE FOR PREDICATE TASK #####
    # print(dataset)
    logger.info(f"Loaded dataset {dataset}")
    return dataset


def recover_args_and_inject_pred(tokenizer, input_ids, arg_ids, id2label):
    tmp_toks, tmp_args = [], []
    foo = tokenizer.decode(input_ids, skip_special_tokens=True)
    bar = tokenizer(foo, return_offsets_mapping=True)
    baz = [(foo[i:j], arg_ids[a]) for a, (i, j) in enumerate(bar["offset_mapping"])]
    prev = ""
    for token, anno in baz:

        if prev == ">":
            might_append = "PRED"
        else:
            might_append = id2label[anno]
        prev = token
        if not len(token) or token in [">", "<"]:
            continue
        tmp_toks.append(token)
        tmp_args.append(might_append)
    return tmp_toks, tmp_args


def infer_args(dataset: DatasetDict, args_run: Run):
    trainer = load_trainer_from_run(args_run)
    return infer_args2(dataset, trainer, "cuda")


def resolve_args(annos, id2label, tokenizer, begin, end):
    resolved_tokens = []
    resolved_args = []
    for input_ids, arg_ids in zip(annos["input_ids"], annos["arg_ids"]):
        tmp_toks, tmp_args = recover_args_and_inject_pred(
            tokenizer, input_ids, arg_ids, id2label
        )

        resolved_tokens.append(tmp_toks)
        resolved_args.append(tmp_args)

    annos["args"] = resolved_args
    annos["tokens"] = resolved_tokens

    return annos


def infer_args2(dataset: DatasetDict, trainer: Trainer, device: str):

    # print_sample_as_table(dataset["test"][0])
    # print_sample_as_table(dataset["test"][1])
    # assert False
    ##### PREPARE FOR ARGUMENTS TASK #####

    config = trainer.model.config
    model = trainer.model.to(device)
    tokenizer = trainer.tokenizer
    logger.info("Loaded model for argument prediction.")

    ##### POST PROCESS ARGUMENTS TASK #####

    ##### RUNNING ARGUMENTS TASK #####

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    for splt in dataset.keys():
        logger.info(f"Predicting arguments on {splt} split..")
        tensors_only = dataset[splt].select_columns(["input_ids", "attention_mask"])
        data_loader = DataLoader(
            tensors_only, batch_size=BATCH_SIZE, collate_fn=data_collator
        )
        preds = run_predicate(data_loader, model, device)
        dataset[splt] = dataset[splt].add_column("arg_ids", preds)

    begin, end = find_begin_and_end_id(tokenizer)
    dataset = dataset.map(
        lambda ex: resolve_args(ex, config.id2label, tokenizer, begin, end),
        batched=True,
        num_proc=8,
        desc="Mapping annotations..",
        remove_columns=["arg_ids", "input_ids", "attention_mask"],
    )
    return dataset


def infer_pred(dataset: DatasetDict, pred_run: Run):
    logger.info(f"Infering for predicate task {pred_run}..")

    trainer = load_trainer_from_run(pred_run)
    return infer_pred2(dataset, trainer, "cuda")


def infer_pred2(dataset: DatasetDict, trainer: Trainer, device: str):
    tokenizer = trainer.tokenizer

    dataset = dataset.map(
        lambda ex: flatten(ex, "text", "sent_id"),
        remove_columns=dataset["test"].column_names,
        batched=True,
        num_proc=8,
        desc="Flattening..",
    )

    dataset = dataset.map(
        lambda ex: tokenize(
            ex,
            "text",
            tokenizer,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_SENTENCE_LENGTH,
            return_offsets_mapping=True,
            padding=True,
        ),
        batched=True,
        num_proc=12,
        desc="Tokenizing..",
    )

    model = trainer.model.to(device)

    ##### POST PROCESS PREDICATE TASK #####
    def duplicate_rows_per_annotation(examples, begin, end):
        result = defaultdict(list)
        for sent_idx, word_idxs in enumerate(examples["input_ids"]):
            preds = examples["pred"][sent_idx]
            attentions = examples["attention_mask"][sent_idx]

            indices = [
                j
                for j, p in enumerate(preds)
                if p == 1 and len(attentions) > j and attentions[j] == 1
            ]

            sents = []
            new_attentions = []
            new_word_ids = []
            for i in indices:
                if i < len(word_idxs):

                    prefix = word_idxs[:i]
                    marker = [begin, word_idxs[i], end]
                    suffix = word_idxs[i + 1 :]

                    new_indices = prefix + marker + suffix
                    sents.append(new_indices)

                    last_attended = -1
                    for a_i, a in enumerate(examples["attention_mask"][sent_idx]):
                        if a == 0:
                            last_attended = a_i - 1
                            break

                    last_attended += 2  # account for added symbols
                    last_attended += 1  # turn index into length for [1] * ..

                    atts = [1] * last_attended
                    atts = atts + [0] * (len(new_indices) - last_attended)

                    assert len(atts) == len(new_indices)
                    new_attentions.append(atts)
                    # logger.warning(warning_text)
            result["input_ids"].extend(sents)
            # print(examples["sent_id"][i])
            # print(sents)
            result["sent_id"].extend([examples["sent_id"][sent_idx]] * len(sents))
            result["id"].extend([examples["id"][sent_idx]] * len(sents))
            result["attention_mask"].extend(new_attentions)

        return result

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    for splt in dataset.keys():
        logger.info(f"Predicting predicates on {splt} split..")
        tensors_only = dataset[splt].select_columns(["input_ids", "attention_mask"])
        # print(tensors_only)
        data_loader = DataLoader(
            tensors_only, batch_size=BATCH_SIZE, collate_fn=data_collator
        )
        # print(data_loader)
        preds = run_predicate(data_loader, model, device)
        dataset[splt] = dataset[splt].add_column("pred", preds)

    column_names = dataset["test"].column_names

    begin, end = find_begin_and_end_id(tokenizer)
    # column_names = dataset["train"].column_names
    dataset = dataset.map(
        lambda ex: duplicate_rows_per_annotation(ex, begin, end),
        batched=True,
        remove_columns=column_names,
        desc="Duping rows per annotation",
    )
    return dataset


def infer(
    dataset: Union[Dataset, DatasetDict, Path],
    target_path: Path,
    pred_run: Run,
    args_run: Run,
    do_prep=True,
):

    ##### RUNNING PREDICATE TASK #####
    dataset = juggle_dataset(dataset)

    dataset = infer_pred(dataset, pred_run)
    dataset = infer_args(dataset, args_run)

    logger.info(f"Saving dataset to {target_path}")

    dataset.save_to_disk(str(target_path))
    return dataset


class SRLParser:
    def __init__(self, pred_trainer: Trainer, args_trainer: Trainer) -> None:
        self.pred_trainer = pred_trainer
        self.args_trainer = args_trainer
        if pred_trainer.tokenizer.vocab_size != args_trainer.tokenizer.vocab_size:
            raise TypeError(
                "Missmatched vocab size, are you sure the base models are equal?"
            )

    def parse_dataset(
        self,
        dataset,
        pred_device="cuda:0",
        args_device="cuda:1",
        tasks: List[Literal["pred", "args"]] = None,
    ):
        if tasks is None:
            tasks = ["pred", "args"]
        dataset = juggle_dataset(dataset)

        if "pred" in tasks:

            dataset = infer_pred2(dataset, self.pred_trainer, pred_device)
        if "args" in tasks:
            dataset = infer_args2(dataset, self.args_trainer, args_device)
        else:

            def do_thing(ex, tokenizer):
                ex["tokens"] = tokenizer.convert_ids_to_tokens(ex["input_ids"])
                return ex

            dataset = dataset.map(lambda ex: do_thing(ex, self.pred_trainer.tokenizer))

        return dataset

    def parse(
        self,
        text: str,
        do_split_into_sentences=False,
        pred_device="cuda:0",
        args_device="cuda:1",
        tasks: List[Literal["pred", "args"]] = None,
    ) -> Dataset:
        if do_split_into_sentences:
            dataset = Dataset.from_list([{"text": sent_tokenize(text), "id": 0}])

        else:
            dataset = Dataset.from_list([{"text": [text], "id": 0}])

        dataset = self.parse_dataset(dataset, pred_device, args_device, tasks)
        dataset = dataset[list(dataset.keys())[0]]
        return dataset

    @classmethod
    def from_paths(cls, pred_path: Path, args_path: Path):
        pred_trainer = load_trainer_from_path(pred_path)
        args_trainer = load_trainer_from_path(args_path)
        return cls(pred_trainer, args_trainer)

    @classmethod
    def from_models(cls, pred, args):
        return cls(pred, args)


def infer_debug(sentence: str, pred_run: Run, args_run: Run) -> Dataset:
    parser = SRLParser.from_runs(pred_run, args_run)
    dataset = parser.parse(sentence)
    return dataset
