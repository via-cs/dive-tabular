"""Build the categorical Anxiety dataset from ``raw/data.csv``.

This is the artifact-facing entry point for the implementation in the parent
directory's ``build_dataset.py``. All builder arguments remain available.

Examples:
    uv run python dataset/anxiety-categorical/raw/preprocess_anxiety.py \
        --overwrite
    uv run python dataset/anxiety-categorical/raw/preprocess_anxiety.py \
        --data-file /tmp/anxiety.csv \
        --output-dir /tmp/anxiety-categorical \
        --overwrite
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


BUILDER_PATH = Path(__file__).resolve().parent.parent / "build_dataset.py"


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "anxiety_categorical_builder",
        BUILDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Anxiety builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> None:
    load_builder().main(argv)


if __name__ == "__main__":
    main()
