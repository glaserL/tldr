from typing import Callable, List, Optional

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_raw_text(path: Path):
    with open(path) as f:
        return "\n".join(f.readlines())


def get_reader(path: Path) -> Optional[Callable[[Path], str]]:
    if path.suffix.lower() in [".text", ".txt"]:
        return read_raw_text


def read_files(paths: List[Path]):
    data = []
    for path in paths:
        reader = get_reader(path)
        if reader is None:
            logger.warning(f"No file reader found for {path.name}, skipping.")
            continue
        data.append(reader(path))
    return data
