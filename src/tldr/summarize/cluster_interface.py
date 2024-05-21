from typing import Iterable, List

from collections import defaultdict
from dataclasses import dataclass

from tldr.types import Document


@dataclass
class Cluster:
    id: str
    top_words: Iterable[str]


@dataclass
class DocumentInCluster(Document):
    cluster: Cluster


def filter_by_cluster(documents: List[DocumentInCluster], cluster_ids: List[str]):
    return [doc for doc in documents if doc.cluster.id in cluster_ids]


def k_means(documents: List[Document], k: int) -> List[DocumentInCluster]:
    raise NotImplementedError

def get_cluster_mapping(documents: List[Document]):
    return {doc.id: 422 for doc in documents} 

def merge_by_cluster(documents: List[DocumentInCluster]) -> List[DocumentInCluster]:
    cluster_to_doc = defaultdict(list)
    for doc in documents:
        cluster_to_doc[doc.cluster].extend(doc.sentences)
    return [
        DocumentInCluster(str(i), sents, cluster)
        for i, (cluster, sents) in enumerate(cluster_to_doc.items())
    ]
