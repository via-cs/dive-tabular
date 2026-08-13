"""Repeated sampling and synthetic CSV output handling."""

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .decimal_formatting import (
    format_synthetic_file,
    infer_decimal_places,
)

from .categoricals import (
    decode_categoricals,
    restore_numeric_categoricals,
)


def write_synthetic_samples(
    *,
    output_dir: Path,
    sample: Callable[[int, int, bool], Any],
    train: pd.DataFrame,
    categorical_columns: list[str],
    label_maps: dict[str, list[Any]],
    num_files: int,
    num_rows: int | None,
    seed: int,
    verbose: bool,
) -> list[Path]:
    """Replace an experiment's numbered synthetic CSV set."""
    if num_files < 1:
        raise ValueError('--num-files must be at least 1.')
    sample_size = len(train) if num_rows is None else num_rows
    if sample_size < 1:
        raise ValueError('--num-rows must be at least 1.')

    synthetic_dir = Path(output_dir) / 'synthetic'
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    for existing_path in synthetic_dir.glob('synthetic_*.csv'):
        existing_path.unlink()

    reference = decode_categoricals(train, label_maps)
    decimal_places = infer_decimal_places(reference)
    synthetic_paths = []
    for index in range(num_files):
        raw_synthetic = sample(sample_size, seed + index, verbose)
        synthetic = pd.DataFrame(raw_synthetic, columns=train.columns)
        synthetic = restore_numeric_categoricals(
            synthetic,
            train,
            categorical_columns,
        )
        synthetic = decode_categoricals(synthetic, label_maps)
        synthetic_path = synthetic_dir / f'synthetic_{index}.csv'
        synthetic.to_csv(synthetic_path, index=False)
        format_synthetic_file(
            reference,
            synthetic_path,
            decimal_places=decimal_places,
        )
        synthetic_paths.append(synthetic_path)
    return synthetic_paths
