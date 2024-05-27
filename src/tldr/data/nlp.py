from typing import Callable, Dict, List

from nltk.stem import WordNetLemmatizer

from tldr.types import Document, SentenceGraph, Token

wnl = WordNetLemmatizer()


def lemmatize_verb(graph: Dict[str, List[Token]], verb_key="VERB"):
    for token in graph[verb_key]:
        token.value = wnl.lemmatize(token.value, pos="v")
    return graph


def prune_O(graph: Dict[str, List[Token]]):

    return {eat: ass for eat, ass in graph.items() if eat != "O"}


def prune_single_node_graphs(sentence: SentenceGraph) -> SentenceGraph:
    sentence.subgraphs = [s for s in sentence.subgraphs if len(s) > 2]
    return sentence


def apply_sentence_level(
    sentence: SentenceGraph, transformation: Callable[[SentenceGraph], SentenceGraph]
) -> SentenceGraph:
    return transformation(sentence)


def apply_transformation(document: List[SentenceGraph], transformation):
    for sentence in document:

        sentence.subgraphs = [transformation(g) for g in sentence.subgraphs]
    return document


def apply_transformation2(
    document: Document,
    transformation: Callable[[Dict[str, List[Token]]], Dict[str, List[Token]]],
):
    document.sentences = apply_transformation(document.sentences, transformation)
    return document


def apply_transformation3(
    document: Document, transformation: Callable[[SentenceGraph], SentenceGraph]
):
    document.sentences = [transformation(sent) for sent in document.sentences]
    return document


def prune_empty_sentence_graphs(document: Document) -> Document:
    document.sentences = [sent for sent in document.sentences if len(sent.subgraphs)]
    return document


def apply_transformation4(
    document: Document, transformation: Callable[[Document], Document]
):
    return transformation(document)


def remove_stopwords(graph: Dict[str, List[Token]]):
    from nltk.corpus import stopwords

    for arg in graph:
        graph[arg] = [
            t for t in graph[arg] if t.value not in stopwords.words("english")
        ]
    return graph


def remove_empty(graph: Dict[str, List[Token]]):
    return {key: val for key, val in graph.items() if len(val)}
