"""GReaT-specific serialization and generated-schema validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd


_RESERVED_FRAGMENTS = (',', '\n', '\r', ' is ')


@dataclass(frozen=True)
class GreatValueCodec:
    """Reversibly protect categorical values from GReaT's text delimiters."""

    encoded_columns: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        categorical_columns: list[str],
    ) -> 'GreatValueCodec':
        encoded = []
        for column in categorical_columns:
            if column not in data.columns:
                continue
            values = data[column].dropna().astype(str)
            if values.map(
                lambda value: any(fragment in value for fragment in _RESERVED_FRAGMENTS)
            ).any():
                encoded.append(column)
        return cls(tuple(encoded))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode dangerous categorical values for model-facing data."""
        transformed = data.copy()
        for column in self.encoded_columns:
            transformed[column] = transformed[column].map(
                lambda value: value
                if pd.isna(value)
                else quote(str(value), safe='-_.~')
            )
        return transformed

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Restore encoded categorical values in generated output."""
        restored = data.copy()
        for column in self.encoded_columns:
            restored[column] = restored[column].map(
                lambda value: value
                if pd.isna(value)
                else unquote(str(value))
            )
        return restored

    def as_dict(self) -> dict:
        return {
            'method': 'url_quote',
            'encoded_columns': list(self.encoded_columns),
            'reserved_fragments': list(_RESERVED_FRAGMENTS),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> 'GreatValueCodec':
        if payload.get('method') != 'url_quote':
            raise ValueError(
                f"Unsupported GReaT value codec: {payload.get('method')!r}"
            )
        columns = payload.get('encoded_columns', [])
        if not isinstance(columns, list) or not all(
            isinstance(column, str) for column in columns
        ):
            raise ValueError('encoded_columns must be a list of strings.')
        return cls(tuple(columns))


def format_great_row(row: pd.Series, float_precision: int | None) -> str:
    """Format a row exactly like GReaTDataset, without column shuffling."""
    parts = []
    for column, value in row.items():
        if isinstance(value, (float, np.floating)) and float_precision is not None:
            rendered = f'{value:.{float_precision}f}'
            if '.' in rendered:
                rendered = rendered.rstrip('0').rstrip('.')
        else:
            rendered = str(value).strip()
        parts.append(f'{column} is {rendered}')
    return ', '.join(parts)


def _nearest_observed(
    values: pd.Series,
    observed: pd.Series,
) -> tuple[pd.Series, int]:
    choices = np.asarray(sorted(observed.dropna().unique()), dtype=float)
    numeric = pd.to_numeric(values, errors='coerce')
    finite = numeric.notna() & np.isfinite(numeric)
    snapped = numeric.copy()
    if choices.size and finite.any():
        source = numeric.loc[finite].to_numpy(dtype=float)
        nearest = choices[
            np.abs(source[:, None] - choices[None, :]).argmin(axis=1)
        ]
        snapped.loc[finite] = nearest
    changed = int((snapped.loc[finite] != numeric.loc[finite]).sum())
    return snapped, changed


def validate_great_sample(
    sample: pd.DataFrame,
    *,
    model_train: pd.DataFrame,
    output_train: pd.DataFrame,
    categorical_columns: list[str],
    codec: GreatValueCodec,
) -> tuple[pd.DataFrame, dict]:
    """Reject structurally invalid rows and restore the public data schema."""
    expected = list(model_train.columns)
    missing = [column for column in expected if column not in sample.columns]
    if missing:
        return pd.DataFrame(columns=expected), {
            'candidate_rows': int(len(sample)),
            'accepted_rows': 0,
            'missing_columns': missing,
            'invalid_categorical_cells': 0,
            'invalid_numeric_cells': 0,
            'out_of_range_cells': 0,
            'snapped_categorical_cells': 0,
        }

    candidate = sample.loc[:, expected].copy()
    valid_rows = pd.Series(True, index=candidate.index)
    diagnostics = {
        'candidate_rows': int(len(candidate)),
        'accepted_rows': 0,
        'missing_columns': [],
        'invalid_categorical_cells': 0,
        'invalid_numeric_cells': 0,
        'out_of_range_cells': 0,
        'snapped_categorical_cells': 0,
    }

    categorical = set(categorical_columns)
    for column in expected:
        reference = model_train[column]
        if column in categorical and not pd.api.types.is_numeric_dtype(reference):
            rendered = candidate[column].astype(str).str.strip()
            allowed = set(reference.dropna().astype(str))
            cell_valid = rendered.isin(allowed)
            diagnostics['invalid_categorical_cells'] += int((~cell_valid).sum())
            valid_rows &= cell_valid
            candidate[column] = rendered
            continue

        numeric = pd.to_numeric(candidate[column], errors='coerce')
        finite = numeric.notna() & np.isfinite(numeric)
        diagnostics['invalid_numeric_cells'] += int((~finite).sum())
        valid_rows &= finite

        if column in categorical:
            numeric, changed = _nearest_observed(numeric, reference)
            diagnostics['snapped_categorical_cells'] += changed
        else:
            minimum = float(reference.min())
            maximum = float(reference.max())
            in_range = numeric.between(minimum, maximum, inclusive='both')
            diagnostics['out_of_range_cells'] += int((finite & ~in_range).sum())
            valid_rows &= in_range
        candidate[column] = numeric

    accepted = candidate.loc[valid_rows].reset_index(drop=True)
    accepted = codec.inverse_transform(accepted)

    for column in categorical_columns:
        reference = output_train[column]
        if pd.api.types.is_numeric_dtype(reference):
            accepted[column], _ = _nearest_observed(accepted[column], reference)
            if pd.api.types.is_integer_dtype(reference):
                accepted[column] = accepted[column].round().astype(reference.dtype)

    diagnostics['accepted_rows'] = int(len(accepted))
    return accepted, diagnostics
