"""Input resolution and output helpers for categorical discovery."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def resolve_input(path: Path, expected_name: str) -> Path:
    """Resolve an explicit file or a directory containing the expected file."""
    path = path.expanduser()
    resolved = path / expected_name if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def load_inputs(
    meta_path: Path, data_path: Path
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    """Load only columns marked ``cat`` in the sibling info.json."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset_description = meta.get("dataset_description")
    descriptions = meta.get("column_descriptions")
    if not isinstance(dataset_description, str) or not dataset_description.strip():
        raise ValueError("meta.json must contain a non-empty dataset_description")
    if not isinstance(descriptions, dict) or not descriptions:
        raise ValueError("meta.json must contain a column_descriptions object")

    info_path = meta_path.with_name("info.json")
    if not info_path.is_file():
        raise FileNotFoundError(
            "Categorical discovery requires info.json beside meta.json"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    column_types = info.get("col_types")
    if not isinstance(column_types, dict):
        raise ValueError("sibling info.json must contain a col_types object")
    categorical_columns = [
        name
        for name, specification in column_types.items()
        if isinstance(specification, dict) and specification.get("type") == "cat"
    ]
    if len(categorical_columns) < 2:
        raise ValueError("info.json must mark at least two columns with type 'cat'")

    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError("data.csv must contain at least one row")
    missing_columns = [name for name in categorical_columns if name not in data]
    if missing_columns:
        raise ValueError(
            "info.json categorical columns are missing from data.csv: "
            f"{missing_columns}"
        )
    missing_descriptions = [
        name for name in categorical_columns if name not in descriptions
    ]
    if missing_descriptions:
        raise ValueError(
            "meta.json is missing descriptions for categorical columns: "
            f"{missing_descriptions}"
        )
    categorical_data = data[categorical_columns]
    if categorical_data.isna().any().any():
        missing = categorical_data.columns[
            categorical_data.isna().any()
        ].tolist()
        raise ValueError(f"categorical columns contain missing values: {missing}")

    for column in categorical_columns:
        for value in categorical_data[column].drop_duplicates().tolist():
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"categorical column {column!r} contains a non-JSON scalar"
                )
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(
                    f"categorical column {column!r} contains a non-finite value"
                )

    return (
        meta,
        data,
        {name: descriptions[name] for name in categorical_columns},
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
