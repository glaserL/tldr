import pytest

from tldr.data.alice_and_bob import toy_srl_parsed_data
from tldr.data.graph import srl2document
from tldr.summarize.main import get_summarizer_new_type_beat
from tldr.summarize.merge import do_the_thing
from tldr.types import Document, SentenceGraph, Token


@pytest.fixture()
def data():
    data = toy_srl_parsed_data()
    data = srl2document(data)
    data = [data, data]
    data = data[:]  # make sure its copied, dunno
    return data


tested_objects = ["acklington"]  # TODO: use again once its added
# expected = set(["bullshit", "their arguments"])

# QUERY = "What is bullshit?"


@pytest.mark.skip(reason="Not implemented yet.")
def test_clustering(data):

    data = cluster_documents(data)
    assert len(data)


@pytest.mark.parametrize("summarizer_name", tested_objects, ids=tested_objects)
def test_summarizers(data, summarizer_name):
    summarizer = get_summarizer_new_type_beat(summarizer_name)
    summarized = summarizer().summarize(data)
    assert len(summarized)


even_toyier_data = Document(
    "322",
    [
        SentenceGraph(
            "42",
            ["Bob", "sold", "a", "book", "to", "Alice", "."],
            [
                {
                    "PRED": [Token(1, "sell")],
                    "ARG0": [Token(0, "Bob")],
                    "ARG1": [Token(5, "Alice")],
                    "ARG2": [Token(3, "book")],
                }
            ],
        ),
        SentenceGraph(
            "666",
            ["The", "book", "was", "sold", "for", "$22", "."],
            [
                {
                    "PRED": [Token(3, "sell")],
                    "ARG3": [Token(5, "$22")],
                    "ARG2": [Token(2, "book")],
                }
            ],
        ),
    ],
)


# can_merge_subgraph() for g in
def test_merge_sentences():
    # previous =
    merged = do_the_thing(even_toyier_data)
    assert len(merged.sentences) == len(even_toyier_data.sentences) - 1
