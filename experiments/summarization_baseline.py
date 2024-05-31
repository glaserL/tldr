
import logging
from collections import defaultdict

import evaluate
import mlflow
import torch
from datasets import load_dataset
from mlflow.data.huggingface_dataset import from_huggingface
from tqdm import tqdm
from tracking import setup_mlflow_experiment
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)
rouge = evaluate.load('rouge')

def run_baseline_summarization(dataset_path="cnn_dailymail", model_name="sshleifer/distilbart-cnn-12-6", batch_size=32, device="cuda:0", max_new_tokens=100, do_sample=False):
    setup_mlflow_experiment("summarize/baseline", "Baseline summarization experiments with existing models and data.")

    table_to_log = defaultdict(list)


    with mlflow.start_run(log_system_metrics=True):
        revision = "3.0.0"
        dataset = load_dataset(dataset_path, revision,split="test")

        mlflow.log_param("model", model_name)
        mlflow.log_param("max_new_tokens", max_new_tokens)
        mlflow.log_param("do_sample", do_sample)

        hf_ds = from_huggingface(dataset, dataset_path, "highlights", revision=revision, name=dataset_path) 
        mlflow.log_input(hf_ds, "test", {"task" : "summarization"})
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to(device)

        for batch in tqdm(dataset.iter(batch_size=batch_size), total=(len(dataset)//batch_size)):

            with torch.no_grad():
                inputs = tokenizer(batch["article"], return_tensors="pt", max_length=512, padding=True, truncation=True)
                inputs = inputs.input_ids.to(device)
                y_pred = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
                y_pred = tokenizer.batch_decode(y_pred, skip_special_tokens=True)

            
            rouge.add_batch(predictions=y_pred, references=batch["highlights"])
            table_to_log["references"].extend(batch["highlights"])
            table_to_log["predictions"].extend(y_pred)

        results = rouge.compute()
        mlflow.log_metrics(results)
        mlflow.log_table(table_to_log, artifact_file = "summarizations.json")

run_baseline_summarization()
