"""SDMetrics quality metrics for synthetic tabular data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sdmetrics_metadata(columns, numerical_columns, categorical_columns=()):
    """Build the single-table metadata shape expected by SDMetrics."""
    numerical_columns = set(numerical_columns)
    categorical_columns = set(categorical_columns)
    return {
        "columns": {
            column: {
                "sdtype": (
                    "categorical"
                    if column in categorical_columns
                    else "numerical"
                    if column in numerical_columns
                    else "categorical"
                )
            }
            for column in columns
        }
    }


def _json_value(value: Any) -> Any:
    """Convert pandas and NumPy values into strict JSON values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_value(frame.to_dict(orient="records"))


def evaluate_quality(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return the overall SDMetrics quality score and property scores."""
    from sdmetrics.reports.single_table import QualityReport

    report = QualityReport()
    report.generate(
        real_data=real_train,
        synthetic_data=synthetic,
        metadata=metadata,
        verbose=False,
    )
    properties = report.get_properties()
    return {
        "score": _json_value(float(report.get_score())),
        "properties": _records(properties),
    }


def calculate_quality_breakdown(
    train_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return the SDMetrics report with each underlying detail table."""
    from sdmetrics.reports.single_table import QualityReport

    report = QualityReport()
    report.generate(
        real_data=train_data.copy(),
        synthetic_data=synthetic_data.copy(),
        metadata=metadata,
        verbose=False,
    )
    property_scores = report.get_properties()
    details = {
        property_name: _records(report.get_details(property_name))
        for property_name in property_scores["Property"]
    }
    return {
        "score": _json_value(float(report.get_score())),
        "properties": _records(property_scores),
        "details": details,
    }
