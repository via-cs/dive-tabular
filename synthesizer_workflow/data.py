"""Dataset discovery, configuration resolution, encoding, and splitting."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .categoricals import encode_categoricals
from .runtime import parse_columns


@dataclass
class PreparedData:
    """Resolved dataset configuration and encoded train/test splits."""

    train: pd.DataFrame
    test: pd.DataFrame
    target: str
    feature_columns: list[str]
    categorical_columns: list[str]
    numerical_columns: list[str]
    label_maps: dict[str, list[Any]]
    data_path: Path
    info_path: Path | None
    utility_feature_path: Path | None
    train_path: Path | None
    test_path: Path | None


def _read_optional_json(
    default_path: Path,
    override_path: Path | None,
) -> tuple[dict | None, Path | None]:
    path = override_path or default_path
    if not path.exists():
        if override_path is not None:
            raise FileNotFoundError(f'Configuration file not found: {path}')
        return None, None
    return json.loads(path.read_text(encoding='utf-8')), path


def categorical_columns_from_info(info: dict | None) -> list[str] | None:
    """Read categorical columns from either supported info.json layout."""
    if info is None:
        return None
    col_types = info.get('col_types')
    if isinstance(col_types, dict):
        return [
            name
            for name, metadata in col_types.items()
            if isinstance(metadata, dict) and metadata.get('type') == 'cat'
        ]
    if isinstance(col_types, list):
        column_order = info.get('features_by_order') or []
        if len(col_types) != len(column_order):
            raise ValueError(
                'info.json has different lengths for col_types and '
                'features_by_order.'
            )
        return [
            name
            for name, column_type in zip(column_order, col_types)
            if column_type == 'cat'
        ]
    return None


def _split_data(
    data: pd.DataFrame,
    target: str,
    test_size: float,
    seed: int,
    no_stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    if not no_stratify:
        counts = data[target].value_counts(dropna=False)
        if len(counts) > 1 and counts.min() >= 2:
            stratify = data[target]
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare_data(
    *,
    data_dir: Path,
    data_file: Path | None,
    train_file: Path | None = None,
    test_file: Path | None = None,
    info_file: Path | None,
    utility_feature_file: Path | None,
    categorical_columns_override: str | None,
    test_size: float,
    seed: int,
    max_rows: int | None,
    no_stratify: bool,
) -> PreparedData:
    """Load, validate, encode, and split a dataset for training.

    When ``train_file`` and ``test_file`` are supplied together, those frozen
    splits are used verbatim and no random splitting is performed.
    """
    data_dir = Path(data_dir)
    if (train_file is None) != (test_file is None):
        raise ValueError('--train-file and --test-file must be supplied together.')
    frozen_splits = train_file is not None
    resolved_train_path = Path(train_file) if train_file is not None else None
    resolved_test_path = Path(test_file) if test_file is not None else None
    data_path = (
        Path(data_file)
        if data_file is not None
        else resolved_train_path or data_dir / 'data.csv'
    )
    if not frozen_splits and not data_path.exists():
        raise FileNotFoundError(f'Data file not found: {data_path}')

    info, resolved_info_path = _read_optional_json(
        data_dir / 'info.json',
        Path(info_file) if info_file is not None else None,
    )
    utility, resolved_utility_path = _read_optional_json(
        data_dir / 'utility_feature.json',
        Path(utility_feature_file) if utility_feature_file is not None else None,
    )

    frozen_train_rows = None
    if frozen_splits:
        assert resolved_train_path is not None
        assert resolved_test_path is not None
        for label, path in (
            ('Training split', resolved_train_path),
            ('Test split', resolved_test_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f'{label} not found: {path}')
        if max_rows is not None:
            raise ValueError('--max-rows cannot be used with frozen train/test files.')
        raw_train = pd.read_csv(resolved_train_path)
        raw_test = pd.read_csv(resolved_test_path)
        if raw_train.empty or raw_test.empty:
            raise ValueError('Frozen train and test files must both contain rows.')
        if list(raw_train.columns) != list(raw_test.columns):
            raise ValueError(
                'Frozen train and test files must have identical column order.'
            )
        missing_columns = sorted(
            set(raw_train.columns[raw_train.isna().any()])
            | set(raw_test.columns[raw_test.isna().any()])
        )
        if missing_columns:
            raise ValueError(
                'Frozen train/test files contain missing values in columns: '
                f'{missing_columns}'
            )
        frozen_train_rows = len(raw_train)
        data = pd.concat([raw_train, raw_test], ignore_index=True)
    else:
        data = pd.read_csv(data_path)
        rows_before = len(data)
        data = data.dropna().reset_index(drop=True)
        if rows_before != len(data):
            print(f'Dropped {rows_before - len(data)} rows containing missing values.')
        if max_rows is not None:
            if max_rows < 1:
                raise ValueError('--max-rows must be at least 1.')
            data = data.sample(
                n=min(max_rows, len(data)),
                random_state=seed,
            ).reset_index(drop=True)

    if utility is None:
        raise FileNotFoundError(
            f'Required utility-feature configuration not found: '
            f'{data_dir / "utility_feature.json"}. Provide that file or '
            '--utility-feature-file.'
        )
    target = utility.get('target_column') or utility.get('target')
    if not isinstance(target, str) or not target:
        raise ValueError(
            f'{resolved_utility_path} must contain a non-empty '
            '"target_column" (or "target") string.'
        )
    if target not in data.columns:
        raise ValueError(f'Target column {target!r} is not present in {data_path}.')

    feature_columns = utility.get('feature_columns')
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(
            f'{resolved_utility_path} must contain a non-empty '
            '"feature_columns" list.'
        )
    if not all(isinstance(column, str) and column for column in feature_columns):
        raise ValueError('feature_columns entries must be non-empty strings.')
    if target in feature_columns:
        raise ValueError(
            f'Target column {target!r} must not appear in feature_columns.'
        )
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError('feature_columns contains duplicate names.')
    missing_features = [
        column for column in feature_columns if column not in data.columns
    ]
    if missing_features:
        raise ValueError(
            f'Feature columns are missing from the data: {missing_features}'
        )

    categorical_columns = parse_columns(categorical_columns_override)
    if categorical_columns is None:
        categorical_columns = categorical_columns_from_info(info)
    if categorical_columns is None:
        categorical_columns = [
            column
            for column in data.columns
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
    missing_categorical = [
        column for column in categorical_columns if column not in data.columns
    ]
    if missing_categorical:
        raise ValueError(
            f'Categorical columns are missing from the data: {missing_categorical}'
        )
    numerical_columns = [
        column for column in data.columns if column not in categorical_columns
    ]

    encoded, label_maps = encode_categoricals(data, categorical_columns)
    if frozen_splits:
        assert frozen_train_rows is not None
        train = encoded.iloc[:frozen_train_rows].reset_index(drop=True)
        test = encoded.iloc[frozen_train_rows:].reset_index(drop=True)
    else:
        train, test = _split_data(encoded, target, test_size, seed, no_stratify)
    return PreparedData(
        train=train,
        test=test,
        target=target,
        feature_columns=list(feature_columns),
        categorical_columns=list(categorical_columns),
        numerical_columns=numerical_columns,
        label_maps=label_maps,
        data_path=data_path,
        info_path=resolved_info_path,
        utility_feature_path=resolved_utility_path,
        train_path=resolved_train_path,
        test_path=resolved_test_path,
    )
