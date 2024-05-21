from typing import List, Union

import logging
from abc import abstractmethod

from tldr.summarize.cluster_interface import k_means, merge_by_cluster
from tldr.summarize.legacy import tr_on_sent_graphs
from tldr.types import Document

logger = logging.getLogger(__name__)


def filter_with_text_rank(document: Document, summarizer_k: int) -> Document:
    document.sentences = tr_on_sent_graphs(document.sentences, summarizer_k)
    return document


class Summarizer:

    def summarize(self, data: Union[Document, List[Document]]):
        if isinstance(data, Document):
            return self._summarize(data)
        return [self._summarize(doc) for doc in data]

    @abstractmethod
    def _summarize(self, sentence: Document) -> Document:
        raise NotImplementedError


class Acklington(Summarizer):

    def __init__(self, sentences_per_document=5) -> None:
        self.k = sentences_per_document
        super().__init__()

    def _summarize(self, document: Document) -> Document:
        return filter_with_text_rank(document, self.k)


class NoOp(Summarizer):
    def _summarize(self, sentence: Document) -> Document:
        return sentence


class Albany(Acklington):
    def __init__(self, number_of_clusters=5, sentences_per_topic=10) -> None:
        super().__init__(sentences_per_topic)
        self.k = number_of_clusters

    def add_clusters(self, documents: List[Document]):
        clustered_documents = k_means(documents)
        return clustered_documents

    def summarize(self, data: Union[Document, List[Document]]):
        if isinstance(data, Document):
            logger.warning("Can't summarize a single document by cluster.")
            data = [data]
        data = self.add_clusters(data, self.k)
        data = merge_by_cluster(data)
        return data


new_type_beat_mapping = {
    "acklington": Acklington,
    "albany": Albany,
    "noop": NoOp,
}


def get_summarizer_new_type_beat(name: str) -> type[Summarizer]:
    if name in new_type_beat_mapping:
        return new_type_beat_mapping[name]
    raise TypeError
