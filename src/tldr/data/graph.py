from typing import List

import json
import logging
from collections import defaultdict
from pathlib import Path

from datasets import Dataset

from tldr.types import (
    Document,
    Parse,
    SentenceGraph,
    Token,
)

HERE = Path(__file__).parent
logger = logging.getLogger(__name__)


def get_data():
    with open(HERE / "jaguar.json") as f:
        return json.load(f)


def clean_markers(parse):
    pp = defaultdict(list)
    for arg, tok in zip(parse["args"], parse["tokens"]):
        if arg in ["PRED", "I-PRED", "B-PRED"] and tok in [">", "<"]:
            continue
        pp["args"].append(arg)
        pp["tokens"].append(tok)
    return pp


def parse_sentence_graph(
    parses: List[Parse], sent_id="322", remove_iobes=True, remove_markers=True
) -> SentenceGraph:
    graphs = []
    for parse in parses:

        subgraph = defaultdict(list)
        parse = clean_markers(parse) if remove_markers else parse
        for i, (arg, tok) in enumerate(zip(parse["args"], parse["tokens"])):
            if remove_iobes:
                arg = arg.split("-")[-1]

            subgraph[arg].append(Token(i, tok))

        graphs.append(subgraph)

    return SentenceGraph(sent_id, parse["tokens"], graphs)


def generate_rows_by_key(dataset: Dataset, key: str):
    current_sent_id = -1
    buffer = []
    for row in dataset:
        sent_id = row[key]
        if current_sent_id != sent_id and len(buffer):
            yield sent_id, buffer
            buffer = []
        current_sent_id = sent_id
        buffer.append(row)
    yield sent_id, buffer


def generate_rows_by_sent_id(dataset: Dataset):
    yield from generate_rows_by_key(dataset, "sent_id")


def srl2document(dataset: Dataset, sentence_id_column="sent_id") -> Document:
    doc = []
    for sent_id, sent in generate_rows_by_key(dataset, sentence_id_column):
        sent = parse_sentence_graph(sent, sent_id)
        doc.append(sent)

    return Document("322", doc)


def srl2documents(
    dataset: Dataset, document_id_column="id", sentence_id_column="sent_id"
) -> List[Document]:
    docs = []

    for document_id, sentences in generate_rows_by_key(dataset, document_id_column):
        doc = []
        for sent_id, sent in generate_rows_by_key(sentences, sentence_id_column):
            sent = parse_sentence_graph(sent, sent_id)

            doc.append(sent)
        docs.append(Document(document_id, doc))
    return docs
