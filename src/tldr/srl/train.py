from typing import Iterator, List, Optional

import logging
import os
import re
from dataclasses import dataclass, field
from functools import partial

import evaluate
import mlflow
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from tldr.srl.data import read_huggingface_dataset
from tldr.tracking.main import setup_mlflow_experiment

TRACKING_URI = "http://localhost:5000"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

seqeval = evaluate.load("seqeval")

f1 = evaluate.load("f1")
logger = logging.getLogger(__name__)


def store_best_model(trainer: Trainer):
    ckpt_dir = trainer.state.best_model_checkpoint

    logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")

    mlflow.pyfunc.log_model(
        ckpt_dir,
        artifacts={"model_path": ckpt_dir},
        python_model=mlflow.pyfunc.PythonModel(),
    )


@dataclass
class CustomArguments:
    corpus: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "path to training corpus in CoNLL format"
        },
    )
    labels: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "provide all values of output layers, required for -m option",
        },
    )
    base_model: str = field(
        default="distilbert/distilbert-base-uncased",
        metadata={
            "help": "Which base model to use to use.",
        },
    )
    verbose: bool = field(default=False)
    debug: bool = field(default=False)
    inflate_training_data: bool = field(default=False)
    num_proc: int = field(default=14)
    task: str = field(
        default="both",
        metadata={
            "help": "The task to train. Either pred(icates), arg(uments) or both.",
        },
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={
            "help": "Evaluations the EarlyStoppingCallback is able to wait.",
        },
    )
    tokenizer: str = field(
        default="bert-base-uncased",
        metadata={"help": "Which tokenizer to use. Also consider ''"},
    )



def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def metrics_computation(p, id2label, log_table=False):
    y_pred = [
        [id2label[p.item()] for (p, t) in zip(prediction, truth) if t != -100]
        for prediction, truth in zip(p.predictions, p.label_ids)
    ]
    y_true = [
        [id2label[t.item()] for (_, t) in zip(prediction, truth) if t != -100]
        for prediction, truth in zip(p.predictions, p.label_ids)
    ]

    # if task == "args":
    results = seqeval.compute(predictions=y_pred, references=y_true, scheme="IOB2")
    if log_table:  # {"V" : {accuracy: ..}}
        table = {}
        for label, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                table[f"{label}_{metric}"] = value
        mlflow.log_table(table, "metrics_flattened.json")
        table = {k: v for k, v in results.items() if isinstance(v, dict)}
        mlflow.log_table(table, "metrics.json")
    return {
        "f1": results["overall_f1"],
    }
    # elif task == "pred":
    #     results = f1.compute(predictions=y_pred_clean, references=y_true_clean)
    #     return {"f1": results["f1"]}
    # else:
    #     raise TypeError


def tokenize_and_align_labels(examples, tokenizer):

    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, is_split_into_words=True, max_length=256
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def remove_bracket_notation(buffer):
    """buffer is a list (~sentence) of lists (~word annos)"""
    open_bracket_cols = []
    for n in range(len(buffer)):
        for col in range(len(buffer[n])):
            buffer[n][col] = buffer[n][col].strip()
            if col in open_bracket_cols:
                if buffer[n][col] == "*":
                    buffer[n][col] = buffer[n - 1][col]
                elif buffer[n][col].endswith(")"):
                    buffer[n][col] = buffer[n - 1][col]
                    open_bracket_cols.remove(col)
            elif buffer[n][col].startswith("("):
                buffer[n][col] = buffer[n][col][1:].strip()
                if col not in open_bracket_cols:
                    open_bracket_cols.append(col)
                if buffer[n][col].endswith(")"):
                    buffer[n][col] = buffer[n][col][:-1].strip()
                    open_bracket_cols.remove(col)
            if buffer[n][col].endswith("*"):
                buffer[n][col] = buffer[n][col][:-1].strip()
            if col != 1 and buffer[n][col] in ["O", "", "", "-"]:
                buffer[n][col] = "O"
    return buffer


def count_lines(path):
    with open(path, "r") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def extract_pred_data(f: Iterator[str]):


    labels = set()
    data = []
    buffer = []
    for sentence in f:
        if len(buffer) > 0:
            buffer = [
                [w for w, _ in buffer],
                [p for _, p in buffer],
            ]
            data.append({"text": buffer[0], "labels": buffer[1]})
            buffer = []
        else:
            for line in sentence.split("\n"):
                line = line.split("#")[0].rstrip()
                fields = line.split("\t")
                if len(fields) > 11:
                    word = fields[3]
                    pred = fields[6]
                    pred = f"B-{re.match(r'.*[a-z0-9].*', pred.lower()) is not None}"
                    # Match => boolean => str

                    labels.add(pred)
                    buffer.append((word, pred))
    if len(buffer) > 0:
        buffer = [
                [w for w, _ in buffer],
                [p for _, p in buffer],
            ]
        data.append({"text": buffer[0], "labels": buffer[1]})
    
    return labels, data


def do_iob2(roles):
    current_role = None
    iob2_roles = []
    for role in roles:
        if role == "O":
            iob2_roles.append(role)
        elif role != current_role:
            current_role = role
            iob2_roles.append(f"B-{role}")
        else:
            iob2_roles.append(f"I-{role}")
    return iob2_roles


def extract_args_data(
    f: Iterator[str],
    text_column="text",
    labels_column="labels",
    add_sentence_index=False,
):
    labels = set()
    data = []

    buffer = []
    pred_nrs = []  # integer offset for each pred (first token only)
    c = 0
    for sentence in f:
        if len(buffer) > 0:
            buffer = remove_bracket_notation(buffer)
            for n, p in enumerate(pred_nrs):
                words = [row[0] for row in buffer]
                words = words[0:p] + [">", words[p], "<"] + words[p + 1 :]
                roles = [row[2 + n] for row in buffer]
                roles[p] = "PRED"
                roles = roles[0:p] + ["PRED", roles[p], "PRED"] + roles[p + 1 :]
                roles = do_iob2(roles)
                for role in roles:
                    labels.add(role)
                sth = {text_column: words, labels_column: roles}
                if add_sentence_index:
                    sth["sent_id"] = c
                data.append(sth)
            c += 1
            buffer = []
            pred_nrs = []
        else:
            for line in sentence.split("\n"):
                line = line.split("#")[0].rstrip()
                fields = line.split("\t")
                
                if len(fields) > 11:
                    word = fields[3]
                    pred = fields[6]
                    pred = (
                        re.match(r".*[a-z0-9].*", pred.lower()) is not None
                    )  # Match => boolean => str
                    if pred:
                        pred_nrs.append(len(buffer))
                    pred = str(pred)
                    
                    roles = fields[12:]
                    buffer.append([word, pred] + roles)
    if len(buffer) > 0:
        buffer = remove_bracket_notation(buffer)
        for n, p in enumerate(pred_nrs):
            words = [row[0] for row in buffer]
            words = words[0:p] + [">", words[p], "<"] + words[p + 1 :]
            roles = [row[2 + n] for row in buffer]
            roles[p] = "PRED"
            roles = roles[0:p] + ["PRED", roles[p], "PRED"] + roles[p + 1 :]
            roles = do_iob2(roles)
            for role in roles:
                labels.add(role)
            sth = {text_column: words, labels_column: roles}
            if add_sentence_index:
                sth["sent_id"] = c
            data.append(sth)

    return labels, data


def create_tokenizer(custom_args):
    model = custom_args.base_model
    if model == "facebook/bart-large":
        return AutoTokenizer.from_pretrained(model, add_prefix_space=True)
    return AutoTokenizer.from_pretrained(model)


def run_experiment(
    experiment_name, training_args: TrainingArguments, custom_args: CustomArguments
):
    logger.info(f"Running experiment {experiment_name} with task {custom_args.task}")
    logger.info("Reading data..")
    extract_func = (
        extract_args_data if custom_args.task == "args" else extract_pred_data
    )
    with mlflow.start_run(log_system_metrics=True):

        mlflow.set_tag("task", custom_args.task)
        dataset = read_huggingface_dataset(
            extract_func,
            custom_args.corpus,
            inflate_training_data=custom_args.inflate_training_data,
        )

        tokenizer = create_tokenizer(custom_args)
        mlflow.log_param("tokenizer", custom_args.base_model)

        why_so_boxy = dataset["train"].features["labels"].feature

        id2label = why_so_boxy._int2str
        label2id = why_so_boxy._str2int
        if not isinstance(id2label, dict):
            id2label = {i: v for i, v in enumerate(id2label)}

        sample_text = "\n".join(dataset["train"]["text"][0])
        sample_labels = "\n".join(
            [id2label[x] for x in dataset["train"]["labels"][0] if x != -100]
        )
        if custom_args.debug:
            dataset["test"] = dataset["test"].select(range(0, 100))
            dataset["train"] = dataset["train"].select(range(0, 100))
            dataset["validation"] = dataset["validation"].select(range(0, 100))

        mlflow.log_table(
            {"Text": sample_text, "Labels": sample_labels},
            artifact_file=f"data_example_{custom_args.task}.json",
        )

        dataset = dataset.map(
            lambda ex: tokenize_and_align_labels(ex, tokenizer),
            remove_columns=["text"],
            batched=True,
            num_proc=custom_args.num_proc,
            desc="Tokenizing..",
        )

        logger.info("Creating model..")

        model = AutoModelForTokenClassification.from_pretrained(
            custom_args.base_model,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )
        mlflow.log_param("base_model", custom_args.base_model)

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        metrics_func = partial(metrics_computation, id2label=id2label)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=metrics_func,
            callbacks=[EarlyStoppingCallback(custom_args.early_stopping_patience)],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        trainer.train()

        if training_args.load_best_model_at_end:
            store_best_model(trainer)
####
# Predict (from huggingface repo)
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
####

        logger.info("*** Predict ***")

        trainer.compute_metrics = lambda p: metrics_computation(
            p, id2label, log_table=True
        )
        _, _, metrics = trainer.predict(
            dataset["test"], metric_key_prefix="test"
        )

        mlflow.log_metrics(metrics)


        return metrics["test_loss"], metrics["test_f1"]



def parse_args():

    parser = HfArgumentParser((TrainingArguments, CustomArguments))
    training_args, custom_args = parser.parse_args_into_dataclasses()
    logger.debug(training_args)
    logger.debug(custom_args)
    return training_args, custom_args


def run(
    training_args: Optional[TrainingArguments] = None,
    custom_args: Optional[CustomArguments] = None,
    prefix="ssrl",
):

    if training_args is None or custom_args is None:
        training_args, custom_args = parse_args()

    logging.basicConfig(
        level="DEBUG" if custom_args.verbose else "INFO",
        format="%(levelname)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    )
    task = custom_args.task

    logger.info(f"Starting task {task}")

    use_mlflow = (
        "mlflow" in training_args.report_to
        or isinstance(training_args.report_to, str)
        and training_args.report_to == "all"
    )

    pred_description = (
        "Predict predicates in text data. Required for argument span prediction task."
    )

    args_description = (
        "Predicts arguments for predicates in text data."
        "Requires input that already annotates predicates."
    )

    if task == "pred":
        if use_mlflow:
            experiment = setup_mlflow_experiment(
                f"{prefix}/{task}", pred_description
            )
        return run_experiment(experiment.name, training_args, custom_args)
    elif task == "args":
        if use_mlflow:
            experiment = setup_mlflow_experiment(
                f"{prefix}/{task}", args_description
            )
        return run_experiment(experiment.name, training_args, custom_args)
    elif task == "both":

        if use_mlflow:
            experiment = setup_mlflow_experiment(
                f"{prefix}/{task}", pred_description
            )
        custom_args.task = "pred"
        corpus_files = custom_args.corpus[:]
        run_experiment(experiment, training_args, custom_args)

        if use_mlflow:
            experiment = setup_mlflow_experiment(
                f"{prefix}/{task}", args_description
            )
        custom_args.task = "args"
        custom_args.corpus = corpus_files
        return run_experiment(experiment, training_args, custom_args)


if __name__ == "__main__":
    run()
