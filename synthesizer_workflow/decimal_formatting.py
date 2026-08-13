"""Reusable decimal-precision formatting for synthetic tabular data."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from pathlib import Path

import pandas as pd


def _as_finite_decimal(value):
    """Return value as a finite Decimal, or None when it is not numeric."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return None
    return decimal if decimal.is_finite() else None


def _decimal_places(value):
    decimal = _as_finite_decimal(value)
    if decimal is None:
        return None
    return max(0, -decimal.as_tuple().exponent)


def infer_decimal_places(reference):
    """Infer maximum decimal places for fully numeric reference columns."""
    decimal_places = {}
    for column in reference.columns:
        values = reference[column].dropna()
        if values.empty:
            continue
        places = values.map(_decimal_places)
        if places.notna().all():
            decimal_places[column] = int(places.max())
    return decimal_places


def _format_decimal(value, places):
    decimal = _as_finite_decimal(value)
    if decimal is None:
        return value
    quantum = Decimal(1).scaleb(-places)
    digits = len(decimal.as_tuple().digits)
    with localcontext() as context:
        context.prec = max(28, digits + abs(decimal.adjusted()) + places + 2)
        rounded = decimal.quantize(quantum, rounding=ROUND_HALF_UP)
    return format(rounded, f'.{places}f')


def format_synthetic_dataframe(reference, synthetic, decimal_places=None):
    """Return a copy of synthetic formatted using reference precision."""
    missing = [
        column for column in reference.columns if column not in synthetic.columns
    ]
    if missing:
        raise ValueError(f'Synthetic data is missing reference columns: {missing}')
    formatted = synthetic.copy()
    if decimal_places is None:
        decimal_places = infer_decimal_places(reference)
    for column, places in decimal_places.items():
        formatted[column] = formatted[column].map(
            lambda value, digits=places: _format_decimal(value, digits)
        )
    return formatted


def format_synthetic_file(
    reference,
    synthetic_path,
    output_path=None,
    decimal_places=None,
):
    """Format one synthetic CSV and return the path written."""
    synthetic_path = Path(synthetic_path)
    output_path = synthetic_path if output_path is None else Path(output_path)
    synthetic = pd.read_csv(synthetic_path, dtype=str, keep_default_na=False)
    formatted = format_synthetic_dataframe(reference, synthetic, decimal_places)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(output_path, index=False)
    return output_path
