from typing import Dict, List

from tldr.types import Document, SentenceGraph, Token


def can_merge_graphs(a: SentenceGraph, b: SentenceGraph):
    return (
        len(a.subgraphs) == 1
        and len(b.subgraphs) == 1
        and can_merge_subgraph(a.subgraphs[0], b.subgraphs[0])
    )


def can_merge_subgraph(
    a: Dict[str, List[Token]], b: Dict[str, List[Token]], verb_key="PRED"
):

    if a[verb_key] != b[verb_key]:
        return False

    all_keys = set(a.keys())
    all_keys.update(b.keys())

    for key in all_keys:
        if key in a and key in b and a[key] != b[key]:
            return False
    return True


def merge_subgraph(a: Dict[str, List[Token]], b: Dict[str, List[Token]]):
    a.update(b)
    return a




def do_the_thing(document: Document):
    did_the_thing = []
    for i, (a, b) in enumerate(zip(document.sentences, document.sentences[1:])):
        if not can_merge_graphs(a, b):
            did_the_thing.append(a)
            if i == len(document.sentences):

                did_the_thing.append(b)
            continue

        print(f"MERGING !!!! \n{a} INTO \n{b}!!")
        replacement = SentenceGraph(
            a.id, ["HUH"], [merge_subgraph(a.subgraphs[0], b.subgraphs[0])]
        )
        did_the_thing.append(replacement)

    # assert False

    return Document(document.id, sentences=did_the_thing)


def do_the_thing_old(document: Document):
    did_the_thing = []
    for (
        i,
        j,
    ) in zip(document.sentences, document.sentences[1:]):
        # print("HUH")
        # print(j)
        # print(i)
        if len(i.subgraphs) > 1:
            did_the_thing.append(i)
            continue
        if len(j.subgraphs) > 1:
            # Will be appended in the next step
            continue

        # sent_id = i.id
        # print("OK HERE IT GETS INTERESTING??")
        # print(i)
        # print(j)
        a, b = i.subgraphs[0], j.subgraphs[0]

        if can_merge_subgraph(a, b):
            print(f"MERGING {a} with {b}!!")
            did_the_thing.append(SentenceGraph(i.id, ["HUH"], [merge_subgraph(a, b)]))
        else:
            did_the_thing.extend([i, j])

    return Document(document.id, sentences=did_the_thing)
