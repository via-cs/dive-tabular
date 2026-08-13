"""Experiment artifact persistence and metadata validation."""

import json
from pathlib import Path
from typing import Any

from .categoricals import decode_categoricals
from .data import PreparedData


def load_label_maps(path: Path) -> dict[str, list[Any]]:
    """Load saved categorical label maps."""
    if not path.exists():
        raise FileNotFoundError(f'Label map file not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def write_training_artifacts(
    output_dir: Path,
    prepared: PreparedData,
    *,
    write_label_maps: bool = True,
) -> dict:
    """Write readable splits, optional label maps, and experiment metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    decode_categoricals(prepared.train, prepared.label_maps).to_csv(
        output_dir / 'train.csv',
        index=False,
    )
    decode_categoricals(prepared.test, prepared.label_maps).to_csv(
        output_dir / 'test.csv',
        index=False,
    )
    if write_label_maps:
        save_json(output_dir / 'label_maps.json', prepared.label_maps)
    metadata = {
        'target': prepared.target,
        'feature_columns': prepared.feature_columns,
        'categorical_columns': prepared.categorical_columns,
        'numerical_columns': prepared.numerical_columns,
        'column_order': list(prepared.train.columns),
        'train_size': int(len(prepared.train)),
        'test_size': int(len(prepared.test)),
    }
    save_json(output_dir / 'metadata.json', metadata)
    return metadata


def load_metadata(
    path: Path,
    available_columns: list[str] | None = None,
) -> dict:
    """Load and validate experiment metadata."""
    if not path.exists():
        raise FileNotFoundError(f'Metadata file not found: {path}')
    metadata = json.loads(path.read_text(encoding='utf-8'))
    required_keys = {
        'target',
        'feature_columns',
        'categorical_columns',
        'numerical_columns',
        'column_order',
    }
    missing_keys = sorted(required_keys - metadata.keys())
    if missing_keys:
        raise ValueError(f'{path} is missing required keys: {missing_keys}')
    if available_columns is not None and metadata['column_order'] != available_columns:
        raise ValueError(
            f'Columns in {path} do not match the training CSV. '
            f'Expected {metadata["column_order"]}, found {available_columns}.'
        )
    return metadata


def save_json(path: Path, value: Any) -> None:
    """Write a JSON artifact using stable, readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str),
        encoding='utf-8',
    )


def args_payload(args: Any, **resolved: Any) -> dict:
    """Create a JSON-serializable command configuration."""
    return {
        'command': args.command,
        'args': vars(args),
        'resolved': resolved,
    }
