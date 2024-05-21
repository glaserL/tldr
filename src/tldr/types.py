from typing import Dict, List, TypedDict

from dataclasses import dataclass


class Parse(TypedDict):
    tokens: List[str]
    args: List[str]


@dataclass
class Token:
    index: int
    value: str

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.value == other.value


@dataclass
class SentenceGraph:
    id: str
    full_text: List[str]
    subgraphs: List[Dict[str, List[Token]]]


@dataclass
class Document:
    id: str
    sentences: List[SentenceGraph]
