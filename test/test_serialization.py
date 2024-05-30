import pytest

from tldr.data.alice_and_bob import toy_srl_parsed_data
from tldr.data.graph import srl2document
from tldr.generate.serialization import (
    get_serializer,
)
from tldr.types import SentenceGraph, Token

tested_objects = ["lloyd_wright", "hundertwasser", "calatrava", "hadid", "libeskind"]

# expected = set(["bullshit", "their arguments"])

# QUERY = "What is bullshit?"
sample = {
    "ARG0": [Token(0, "Alice")],
    "PRED": [Token(1, "like")],
    "ARG1": [Token(2, "ripe"), Token(3, "strawberries")],
}
sample = SentenceGraph("322", ["Alice", "likes", "ripe", "strawberries"], [sample])


@pytest.mark.parametrize("serializer_name", tested_objects, ids=tested_objects)
def test_serializer(serializer_name):

    data = toy_srl_parsed_data()
    data = srl2document(data)
    serializer = get_serializer(serializer_name)

    serialized = serializer.serialize(data)
    assert len(serialized) == 7, "It should contain 6 subgraphs"
    assert all(
        [isinstance(graph, str) for graph in serialized]
    ), "All graphs should be serialized to strings"


@pytest.fixture
def sample_sentence() -> SentenceGraph:
    data = toy_srl_parsed_data()
    return srl2document(data).sentences[0]


def test_hundertwasser():
    serialized = get_serializer("hundertwasser").serialize(sample)
    expected = "{'ARG0': ['Alice'], 'PRED': ['like'], 'ARG1': ['ripe', 'strawberries']}"

    assert serialized == expected


def test_hadid():
    serialized = get_serializer("hadid").serialize(sample)
    expected = '{"ARG0": "Alice", "PRED": "like", "ARG1": "ripe strawberries"}'

    assert serialized == expected


def test_lloyd_wright():
    serialized = get_serializer("lloyd_wright").serialize(sample)
    expected = 'ARG0: "Alice" PRED: "like" ARG1: "ripe strawberries"'

    assert serialized == expected


def test_calatrava():
    serialized = get_serializer("calatrava").serialize(sample)
    expected = "ARG0: Alice PRED: like ARG1: ripe strawberries"

    assert serialized == expected


def test_libeskind():
    serialized = get_serializer("libeskind").serialize(sample)
    expected = "Alice: ARG0 like: PRED ripe strawberries: ARG1"

    assert serialized == expected
