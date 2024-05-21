from typing import List

import logging
from math import log, sqrt

from networkx import Graph
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from tldr.types import SentenceGraph

_ZERO_DIVISION_PREVENTION = 1e-7
Sentence = List[str]
Document = List[Sentence]
Summary = List[Sentence]
logger = logging.getLogger()

def tokenize_text(text: str) -> Document:
    result = []
    sentences = sent_tokenize(text, language="english")
    for sent in sentences:
        tok = word_tokenize(sent)
        result.append(tok)
    return result


def filter_stopwords(sentence: Sentence):
    return [s for s in sentence if s not in stopwords.words("english")]


def to_lower(sentence: Sentence):
    return [s.lower() for s in sentence]


def inverse_sentence_frequency(word: str, document: Document):
    numerator = len(document)
    denominator = sum(1 for sent in document if word in sent)
    result = log(numerator / denominator)
    return result


def lower_thing(sent: Sentence, doc: Document):
    result = 0
    for word in sent:
        tf = sent.count(word)
        isf = inverse_sentence_frequency(word, doc)
        result += (tf * isf) ** 2
    return sqrt(result)


def isf_modified_cosine_similarity(sent_i: Sentence, 
                                   sent_j: Sentence, 
                                   doc: Document):
    
    words = set(sent_i).union(set(sent_j))
    numerator = 0
    for word in words:
        tf_i = sent_i.count(word)
        tf_j = sent_j.count(word)
        isf = inverse_sentence_frequency(word, doc)
        numerator += tf_i * tf_j * isf**2
    denominator = lower_thing(sent_i, doc) * lower_thing(sent_j, doc)
    return numerator / denominator


def build_graph(sentences: List[List[str]]):
    g = Graph()
    total_weight = 0
    count = 0
    for i, sent_i in enumerate(sentences):

        if len(sent_i) > 150:
            logger.warning(f"{sent_i} is longer than 150 words sure this is tokenized?")
        for j, sent_j in list(enumerate(sentences))[i:]:
            if j != i:
                w_i_j = isf_modified_cosine_similarity(sent_i, sent_j, sentences)
                g.add_edge(i, j, w=w_i_j)
                total_weight += w_i_j
                count += 1.0
    average_weight = total_weight / count
    return g, average_weight


def filter_graph(graph: Graph, cutoff: float):
    edges_to_remove = []
    for u, v, data in graph.edges.data(True):
        if data["w"] < cutoff:
            edges_to_remove.append((u, v))
    graph.remove_edges_from(edges_to_remove)


def avg(values: List[float]):
    return sum(values) / (len(values) + _ZERO_DIVISION_PREVENTION)


def initialize_node_weights(graph: Graph):
    for node, _ in graph.nodes.items():
        score = avg([data["w"] for _, _, data in graph.edges(node, True)])
        graph.nodes[node]["init_score"] = score


def text_rank(graph: Graph, dampening_factor: float):
    d = dampening_factor
    initialize_node_weights(graph)

    for node, data in graph.nodes.items():
        t = len(graph.nodes)
        rightthing = 0
        for neighbor in graph[node]:
            rightthing += graph.nodes[neighbor]["init_score"]
        rightthing = rightthing / (graph.degree[node] + _ZERO_DIVISION_PREVENTION)
        data["score"] = d / t + (1 - d) * rightthing


def get_top_sentences(graph: Graph, cutoff: int):
    nodes = graph.nodes.data(True)
    filtered = sorted(nodes, key=lambda n: n[1]["score"])[:cutoff]
    return [node for node, _ in filtered]

def get_indices_of_top_ranked(tokenized: Document, size: int):
    graph, average = build_graph(tokenized)
    filter_graph(graph, average)
    text_rank(graph, 0.15)
    highest_ranked = get_top_sentences(graph, size)
    return highest_ranked


def _modified_TR(tokenized: Document, size: int):
    highest_ranked = get_indices_of_top_ranked(tokenized, size)
    result = [sent for i, sent in enumerate(tokenized) if i in highest_ranked]
    return result

def transform(sentence, do_lower=True, do_filter_stopwords=True):
    data = sentence
    if do_lower:
        data = to_lower(data)
    if do_filter_stopwords:
        data = filter_stopwords(data)
    return data
    
    
def tr_on_sent_graphs(doc: List[SentenceGraph], 
                      size: int, 
                      do_lower=True, 
                      do_filter_stopwords=True):

    stopwords.words("english") # Fail early if it's not downloaded
    transforms = []
    if len(doc) == 1:
        logger.warning("Skipping single sentence document.")
        return doc
    if do_lower:
        transforms.append(to_lower)
    if do_filter_stopwords:
        transforms.append(do_filter_stopwords)
    
    text = [transform(d.full_text, do_lower, do_filter_stopwords) for d in doc]

    highest_ranked = get_indices_of_top_ranked(text, size)

    return [g for i, g in enumerate(doc) if i in highest_ranked]
