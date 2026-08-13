"""Input resolution and JSON output shared by equational pipeline stages."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def resolve_input(path: Path, expected_name: str) -> Path:
    """Resolve either an explicit file or a directory containing that file."""
    path = path.expanduser()
    resolved = path / expected_name if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def resolve_output(path: Path | None, input_path: Path, expected_name: str) -> Path:
    """Resolve an optional output file/directory, defaulting to the input file."""
    if path is None:
        return input_path
    path = path.expanduser()
    if path.exists() and path.is_dir():
        return path / expected_name
    if not path.suffix:
        return path / expected_name
    return path


def load_inputs(
    meta_path: Path, data_path: Path
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    """Load discovery inputs and select numerical columns from sibling info.json."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset_description = meta.get("dataset_description")
    descriptions = meta.get("column_descriptions")
    if not isinstance(dataset_description, str) or not dataset_description.strip():
        raise ValueError("meta.json must contain a non-empty dataset_description")
    if not isinstance(descriptions, dict) or not descriptions:
        raise ValueError("meta.json must contain a column_descriptions object")
    if not all(
        isinstance(name, str)
        and isinstance(description, str)
        and description.strip()
        for name, description in descriptions.items()
    ):
        raise ValueError("column_descriptions must map names to non-empty strings")

    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError("data.csv must contain at least one row")

    info_path = meta_path.with_name("info.json")
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        column_types = info.get("col_types")
        if not isinstance(column_types, dict):
            raise ValueError("sibling info.json must contain a col_types object")
        numerical_columns = [
            name
            for name, specification in column_types.items()
            if isinstance(specification, dict)
            and specification.get("type") == "num"
        ]
        missing_data_columns = [
            name for name in numerical_columns if name not in data.columns
        ]
        if missing_data_columns:
            raise ValueError(
                "info.json numerical columns are missing from data.csv: "
                f"{missing_data_columns}"
            )
        non_numeric = [
            name
            for name in numerical_columns
            if not pd.api.types.is_numeric_dtype(data[name])
        ]
        if non_numeric:
            raise ValueError(
                "info.json marks non-numeric data columns as num: "
                f"{non_numeric}"
            )
    else:
        numerical_columns = [
            name for name in data.columns if pd.api.types.is_numeric_dtype(data[name])
        ]

    if len(numerical_columns) < 2:
        raise ValueError("data.csv must contain at least two numerical columns")
    missing_descriptions = [
        name for name in numerical_columns if name not in descriptions
    ]
    if missing_descriptions:
        raise ValueError(
            "meta.json is missing descriptions for numerical columns: "
            f"{missing_descriptions}"
        )
    numeric_data = data[numerical_columns]
    if numeric_data.isna().any().any():
        missing = numeric_data.columns[numeric_data.isna().any()].tolist()
        raise ValueError(f"numerical columns contain missing values: {missing}")
    if not np.isfinite(numeric_data.to_numpy(dtype=float)).all():
        raise ValueError("numerical columns contain non-finite values")

    numerical_descriptions = {
        name: descriptions[name] for name in numerical_columns
    }
    return meta, data, numerical_descriptions


def load_optional_column_descriptions(
    data_path: Path, meta_path: Path | None = None
) -> dict[str, str]:
    """Load descriptions for standalone fix generation, if metadata is available."""
    candidate = meta_path or data_path.with_name("meta.json")
    if not candidate.is_file():
        return {}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    descriptions = payload.get("column_descriptions", {})
    if not isinstance(descriptions, dict):
        raise ValueError("meta.json column_descriptions must be an object")
    return {
        str(name): str(description)
        for name, description in descriptions.items()
        if isinstance(description, str) and description.strip()
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically replace a JSON artifact after the complete run succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
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
