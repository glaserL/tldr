import pytest

from tldr.generate.main import Generator

base_params = {}


def test_generator_creation_from_config():
    _ = Generator.default()


@pytest.mark.needs_gpu
def test_generation():
    pass
