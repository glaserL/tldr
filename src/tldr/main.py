from typing import Sequence, Union

from pathlib import Path

from datasets import concatenate_datasets

from tldr.data.graph import srl2document, srl2documents
from tldr.data.reader import read_files
from tldr.generate.main import Generator
from tldr.generate.serialization import Serializer, get_serializer
from tldr.srl.inference import SRLParser
from tldr.summarize.cluster_interface import get_cluster_mapping
from tldr.summarize.main import (
    Summarizer,
    get_summarizer_new_type_beat,
)
from tldr.tracking.main import newest_srl_parser


class TLDR:

    def __init__(
        self,
        srl_parser: SRLParser,
        summarizer: Summarizer,
        generator: Generator,
        serializer: Serializer,
    ) -> None:
        self.srl_parser = srl_parser
        self.summarizer = summarizer
        self.generator = generator
        self.serializer = serializer

    def run(self, data: str):
        dataset = self.srl_parser.parse(data, do_split_into_sentences=True)
        dataset = srl2document(dataset)
        dataset = self.summarizer.summarize(dataset)
        dataset = self.generator.generate(dataset, self.serializer)
        return dataset

    def run_batch(self, data: Sequence[Union[Path, str]]):
        data = [Path(d) if isinstance(d, str) else d for d in data]
        dataset = read_files(data)
        if len(dataset) > 1:
            clusters = get_cluster_mapping(dataset)
        else:
            clusters = {}

        dataset = [
            self.srl_parser.parse(text, do_split_into_sentences=True)
            for text in dataset
        ]
        dataset = concatenate_datasets(dataset)
        dataset = srl2documents(dataset)
        dataset = self.summarizer.summarize(dataset)
        dataset = self.generator.generate(dataset, self.serializer)
        if len(clusters):
            dataset = dataset.add_column(
                "cluster_id", [clusters[doc["id"]] for doc in dataset]
            )
        return dataset

    @classmethod
    def default(cls):
        parser = newest_srl_parser()
        return cls(
            parser,
            get_summarizer_new_type_beat("acklington")(),
            Generator.default(),
            get_serializer("hundertwasser"),
        )

    @classmethod
    def from_runs_names(cls, pred_run, args_run, gen_run):
        return cls(
            summarizer=get_summarizer_new_type_beat("acklington")(),
            srl_parser=SRLParser.from_run_names(pred_run, args_run),
            generator=(
                Generator.from_run(gen_run)
                if gen_run is not None
                else Generator.default()
            ),
            serializer=get_serializer("hundertwasser"),
        )
