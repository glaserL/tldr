from typing import Dict, List, Union

from abc import abstractmethod

from tldr.types import (
    Document,
    SentenceGraph,
    Token,
)


def prune_O(graph: Dict[str, List[Token]]):
    return {eat: ass for eat, ass in graph.items() if eat != "O"}


def stoooooopid(subgraph: Dict[str, List[Token]]) -> str:
    result = {key: [tok.value for tok in val] for key, val in subgraph.items()}
    result = str(result)
    return result


def hundertwasser(
    document: SentenceGraph,
    prune_Os=False,
) -> str:
    if prune_Os:
        subgraphs = [prune_O(s) for s in document.subgraphs]
    else:
        subgraphs = document.subgraphs
    result = [stoooooopid(s) for s in subgraphs]
    result = "\n".join(result)
    return result


def _lloyd_wright(subgraph):
    result = []
    for arg, vals in subgraph.items():
        rprr = f'{arg}: "' + " ".join(t.value for t in vals) + '"'
        result.append(rprr)
    return " ".join(result)


def lloyd_wright(sentence: SentenceGraph):
    result = "\n".join([_lloyd_wright(g) for g in sentence.subgraphs])
    return result


class Serializer:

    def __init__(self) -> None:
        pass

    def serialize(self, data: Union[Document, SentenceGraph]):
        if isinstance(data, Document):
            return [self._serialize(sentence) for sentence in data.sentences]
        return self._serialize(data)

    def serialize_to_list_of_dicts(
        self, data: Union[Document, SentenceGraph]
    ) -> List[Dict[str, str]]:
        if isinstance(data, Document):
            return [
                {
                    "id": data.id,
                    "sent_id": sentence.id,
                    "data": self._serialize(sentence),
                }
                for sentence in data.sentences
            ]
        return [{"sent_id": data.id, "data": self._serialize(data)}]

    @abstractmethod
    def _serialize(self, sentence: SentenceGraph) -> str:
        pass


class LloydWright(Serializer):

    def _serialize(self, sentence: SentenceGraph) -> str:
        return lloyd_wright(sentence)


class Hundertwasser(Serializer):
    def _serialize(self, sentence: SentenceGraph) -> str:
        return hundertwasser(sentence, True)


serializer_mapping = {
    "hundertwasser": Hundertwasser,
    # "gropius": gropius,
    "lloyd_wright": LloydWright,
}


def available_serializers():
    return list(serializer_mapping.keys())


def get_serializer(name) -> Serializer:
    if name in serializer_mapping:
        return serializer_mapping[name]()
    raise TypeError
