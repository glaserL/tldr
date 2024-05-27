import pytest

from tldr.data.alice_and_bob import get_raw_text
from tldr.main import TLDR


@pytest.fixture(scope="module")
def youyaku():

    return TLDR.default()


@pytest.mark.needs_gpu
def test_summarize_sentence(youyaku: TLDR):
    text = get_raw_text()
    youyaku.run(text)
