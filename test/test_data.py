from tldr.data.alice_and_bob import toy_srl_parsed_data
from tldr.data.graph import srl2document


def test_graph_creation():
    data = toy_srl_parsed_data()
    document_graph = srl2document(data)
    assert (
        len(document_graph.sentences) == 7
    ), "the document should contain 7 sentences."
