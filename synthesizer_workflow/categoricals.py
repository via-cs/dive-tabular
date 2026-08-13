"""Categorical encoding and synthetic-value restoration."""

from typing import Any

import numpy as np
import pandas as pd


def encode_categoricals(
    data: pd.DataFrame,
    categorical_columns: list[str],
    label_maps: dict[str, list[Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[Any]]]:
    """Encode non-numeric categorical columns with stable integer labels."""
    data = data.copy()
    using_existing_maps = label_maps is not None
    resolved_maps = {} if label_maps is None else dict(label_maps)
    for column in categorical_columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            continue
        if using_existing_maps:
            if column not in resolved_maps:
                raise ValueError(
                    f'Column {column!r} requires encoding but is missing from '
                    'the label map.'
                )
            categories = resolved_maps[column]
        else:
            categories = sorted(data[column].dropna().unique().tolist())
        categorical_type = pd.CategoricalDtype(
            categories=categories,
            ordered=True,
        )
        original = data[column]
        codes = original.astype(categorical_type).cat.codes.astype(int)
        unknown = codes == -1
        if unknown.any():
            values = sorted(
                set(original.loc[unknown].dropna().astype(str).tolist())
            )
            raise ValueError(
                f'Column {column!r} contains values absent from the label map: '
                f'{values[:10]}'
            )
        resolved_maps[column] = categories
        data[column] = codes
    return data, resolved_maps


def decode_categoricals(
    data: pd.DataFrame,
    label_maps: dict[str, list[Any]],
) -> pd.DataFrame:
    """Decode integer labels back to their original categorical values."""
    data = data.copy()
    for column, categories in label_maps.items():
        if column not in data.columns:
            continue
        codes = data[column].round().astype(int).clip(0, len(categories) - 1)
        data[column] = codes.map(dict(enumerate(categories)))
    return data


def _snap_to_values(series: pd.Series, values: list[Any]) -> pd.Series:
    if not values:
        return series
    choices = np.asarray(sorted(values), dtype=float)
    numeric = (
        pd.to_numeric(series, errors='coerce')
        .fillna(choices[0])
        .to_numpy(dtype=float)
    )
    nearest = choices[np.abs(numeric[:, None] - choices[None, :]).argmin(axis=1)]
    return pd.Series(nearest, index=series.index)


def restore_numeric_categoricals(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Snap numerical categorical outputs to values observed during training."""
    synthetic = synthetic.copy()
    for column in categorical_columns:
        if not pd.api.types.is_numeric_dtype(train[column]):
            continue
        values = train[column].dropna().unique().tolist()
        synthetic[column] = _snap_to_values(synthetic[column], values)
        if pd.api.types.is_integer_dtype(train[column]):
            synthetic[column] = (
                synthetic[column].round().astype(train[column].dtype)
            )
        else:
            synthetic[column] = synthetic[column].astype(train[column].dtype)
    return synthetic
