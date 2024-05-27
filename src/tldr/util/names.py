import random
from pathlib import Path

THIS_MODULE = Path(__file__).parent


def read_words(path: Path):
    with open(path) as f:
        return [l.rstrip() for l in f if not l.startswith("#")]


ADJECTIVES = read_words(THIS_MODULE / "adjectives.txt")
DISHES = read_words(THIS_MODULE / "dishes.txt")


def generate_name():
    adjective = random.choice(ADJECTIVES)
    dish = random.choice(DISHES)
    number = random.randint(0, 999)
    return f"{adjective}-{dish}#{number}"
