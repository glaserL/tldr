import pytest

from tldr.data.alice_and_bob import toy_srl_parsed_data
from tldr.data.graph import srl2document
from tldr.generate.serialization import (
    get_serializer,
)

tested_objects = ["lloyd_wright", "hundertwasser"]

# expected = set(["bullshit", "their arguments"])

# QUERY = "What is bullshit?"


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
