import mlflow
from datasets import Dataset

from tldr.summarize.data import load_cnn
from tldr.summarize.eval import evaluate_summarization
from tldr.summarize.legacy import doc_to_raw_string, modified_TR
from tldr.tracking.main import setup_mlflow_experiment


# def test_tr():
#     path = ...
#     cnn =
#     dataset = Corpus(path)
#     summarizer =
#     summarized = [summarizer(doc ) for doc in dataset]
#     evaluate_summarization(dataset, summarized)
def batched_tr(examples, size):
    summarized = [
        doc_to_raw_string(modified_TR(doc, size)) for doc in examples["article"]
    ]
    examples["summarizations"] = summarized
    return examples


def main():
    setup_mlflow_experiment("summarization/modified-tr")
    with mlflow.start_run(log_system_metrics=True):
        dataset = load_cnn(log_to_mlflow=True)
        dataset = dataset if isinstance(dataset, Dataset) else dataset["test"]
        # dataset = dataset.select(range(0, 322))
        dataset = dataset.map(
            lambda batch: batched_tr(batch, 3), batched=True, batch_size=16, num_proc=8
        )

        evaluate_summarization(dataset["summarizations"], dataset["highlights"], True)


if __name__ == "__main__":
    main()
