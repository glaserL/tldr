from typing import List

from tldr.data.nlp import (
    apply_transformation2,
    apply_transformation3,
    apply_transformation4,
    lemmatize_verb,
    prune_empty_sentence_graphs,
    prune_O,
    prune_single_node_graphs,
    remove_empty,
)
from tldr.types import Document


def meister_propper(doc: Document, pred_name = "VERB"):

    doc = apply_transformation2(doc, lambda g: lemmatize_verb(g,pred_name))

    doc = apply_transformation2(doc, prune_O)

    doc = apply_transformation2(doc, remove_empty)

    doc = apply_transformation3(doc, prune_single_node_graphs)

    doc = apply_transformation4(doc, prune_empty_sentence_graphs)

    return doc

def frau_holle(docs: List[Document], pred_name = "VERB"):
    docs = [meister_propper(doc, pred_name) for doc in docs ]
    return docs