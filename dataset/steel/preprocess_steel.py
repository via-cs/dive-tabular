"""Preprocess the Steel Industry Energy Consumption dataset.

Examples:
    uv run python data/steel/preprocess_steel.py --force
    uv run python data/steel/preprocess_steel.py data/steel/raw --force
    uv run python data/steel/preprocess_steel.py /tmp/steel.csv \
        --output /tmp/steel-clean.csv \
        --metadata-output /tmp/preprocessing_info.json \
        --info-output /tmp/info.json \
        --force

The input may be a directory containing ``Steel_industry_data.csv`` or an
explicit CSV path. Output and metadata arguments likewise accept either a
directory or an explicit filename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sdv.metadata import Metadata


DATASET_HANDLE = "csafrit2/steel-industry-energy-consumption"
DATASET_URL = (
    "https://www.kaggle.com/datasets/"
    "csafrit2/steel-industry-energy-consumption"
)
INPUT_FILENAME = "Steel_industry_data.csv"
OUTPUT_FILENAME = "data.csv"
PREPROCESSING_INFO_FILENAME = "preprocessing_info.json"
INFO_FILENAME = "info.json"

DEFAULT_INPUT = Path("data/steel/raw")
DEFAULT_OUTPUT = Path("data/steel/data.csv")
CO2_FACTOR_TONNES_PER_KWH = 0.0004585
CO2_DECIMALS = 2
TARGET = "Usage_kWh"

RAW_COLUMNS = (
    "date",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
)
CATEGORICAL_COLUMNS = (
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
)
NUMERICAL_COLUMNS = (
    "Lagging_Current_Reactive_Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "Usage_kWh",
)
OUTPUT_COLUMNS = (
    "Lagging_Current_Reactive_Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    *CATEGORICAL_COLUMNS,
    TARGET,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input CSV or directory containing Steel_industry_data.csv. "
            "Defaults to data/steel/raw."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Cleaned CSV filename or directory. "
            "Defaults to data/steel/data.csv."
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=(
            "Preprocessing metadata JSON filename or directory. Defaults to "
            "preprocessing_info.json beside the raw input CSV."
        ),
    )
    parser.add_argument(
        "--info-output",
        type=Path,
        default=None,
        help=(
            "Model metadata JSON filename or directory. Defaults to info.json "
            "beside the cleaned output CSV."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing CSV and metadata outputs.",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        return path / INPUT_FILENAME
    if path.is_file():
        return path

    if path == DEFAULT_INPUT:
        for fallback in (
            Path("data/steel") / INPUT_FILENAME,
            Path("data/steel/raw_data.csv"),
        ):
            if fallback.is_file():
                return fallback
    return path


def resolve_named_output(path: Path, filename: str) -> Path:
    path = path.expanduser()
    return path if path.suffix.lower() in {".csv", ".json"} else path / filename


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(data: pd.DataFrame) -> None:
    actual = tuple(data.columns)
    if actual != RAW_COLUMNS:
        raise ValueError(
            "Unexpected raw columns. "
            f"Expected {list(RAW_COLUMNS)}, got {list(actual)}."
        )
    missing = int(data.isna().sum().sum())
    if missing:
        raise ValueError(
            f"Raw Steel CSV contains {missing} missing values; expected none."
        )


def rounded_power_factor(
    usage: pd.Series, reactive: pd.Series
) -> pd.Series:
    usage_values = usage.to_numpy(dtype=float)
    reactive_values = reactive.to_numpy(dtype=float)
    denominator = np.hypot(usage_values, reactive_values)
    values = np.divide(
        100.0 * usage_values,
        denominator,
        out=np.zeros(len(usage), dtype=float),
        where=denominator > 0,
    )
    return pd.Series(np.round(values, 2), index=usage.index)


def strict_co2_expected(usage: pd.Series) -> pd.Series:
    return (
        usage.mul(CO2_FACTOR_TONNES_PER_KWH)
        .round(CO2_DECIMALS)
    )


def preprocessing_metadata(
    source: Path,
    destination: Path,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    co2_violation: pd.Series,
    expected_co2: pd.Series,
    lagging_violations: int,
    leading_violations: int,
) -> dict[str, Any]:
    removed = raw.loc[co2_violation].copy()
    removed_dates = pd.to_datetime(
        removed["date"],
        dayfirst=True,
        errors="raise",
    )
    expected_counts = (
        expected_co2.loc[co2_violation]
        .value_counts()
        .sort_index()
    )

    return {
        "download": {
            "provider": "Kaggle",
            "dataset_handle": DATASET_HANDLE,
            "dataset_url": DATASET_URL,
            "file": INPUT_FILENAME,
            "command": "uv run python data/steel/raw/download_steel.py",
        },
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(destination.resolve()),
        "preprocessing_command": (
            "uv run python data/steel/preprocess_steel.py --force"
        ),
        "preprocessing_steps": [
            {
                "order": 1,
                "operation": "validate_schema_and_completeness",
                "details": (
                    "Require the 11 expected source columns in their original "
                    "order and require zero missing values."
                ),
            },
            {
                "order": 2,
                "operation": "validate_power_factor_equations",
                "details": (
                    "Check both stored power-factor columns against "
                    "round(100 * Usage_kWh / hypot(Usage_kWh, reactive_kVarh), 2). "
                    "Rows are reported but not removed by this validation."
                ),
            },
            {
                "order": 3,
                "operation": "remove_strict_co2_equation_violations",
                "details": (
                    "Remove rows where CO2(tCO2) differs from "
                    "round(Usage_kWh * 0.0004585, 2)."
                ),
            },
            {
                "order": 4,
                "operation": "rename_lagging_reactive_column",
                "details": (
                    "Rename Lagging_Current_Reactive.Power_kVarh to "
                    "Lagging_Current_Reactive_Power_kVarh in processed data."
                ),
            },
            {
                "order": 5,
                "operation": "drop_unique_timestamp",
                "details": (
                    "Drop date because every timestamp is unique. Retain NSM, "
                    "Day_of_week, and WeekStatus as modelable temporal fields."
                ),
            },
            {
                "order": 6,
                "operation": "reorder_columns",
                "details": (
                    "Place numerical features first, categorical features next, "
                    "and the Usage_kWh regression target last."
                ),
            },
            {
                "order": 7,
                "operation": "reset_row_index",
                "details": "Reset the retained rows to a zero-based contiguous index.",
            },
        ],
        "strict_co2_filter": {
            "equation": "CO2(tCO2) == round(Usage_kWh * 0.0004585, 2)",
            "coefficient": CO2_FACTOR_TONNES_PER_KWH,
            "coefficient_unit": "tCO2/kWh",
            "rounding_decimals": CO2_DECIMALS,
            "rows_removed": int(co2_violation.sum()),
            "violation_rate": float(co2_violation.mean()),
            "all_removed_rows_have_recorded_co2_zero": bool(
                removed["CO2(tCO2)"].eq(0.0).all()
            ),
            "removed_timestamp_start": (
                removed_dates.min().isoformat() if len(removed_dates) else None
            ),
            "removed_timestamp_end": (
                removed_dates.max().isoformat() if len(removed_dates) else None
            ),
            "removed_usage_kwh_min": (
                float(removed["Usage_kWh"].min()) if len(removed) else None
            ),
            "removed_usage_kwh_max": (
                float(removed["Usage_kWh"].max()) if len(removed) else None
            ),
            "expected_co2_counts": {
                f"{value:.2f}": int(count)
                for value, count in expected_counts.items()
            },
        },
        "power_factor_validation": {
            "lagging_equation_violations": lagging_violations,
            "leading_equation_violations": leading_violations,
            "rows_removed": 0,
        },
        "columns": {
            "source": list(RAW_COLUMNS),
            "output": list(OUTPUT_COLUMNS),
            "categorical": list(CATEGORICAL_COLUMNS),
            "numerical": list(NUMERICAL_COLUMNS),
            "target": TARGET,
            "removed": ["date"],
            "renamed": {
                "Lagging_Current_Reactive.Power_kVarh":
                    "Lagging_Current_Reactive_Power_kVarh",
            },
        },
        "row_counts": {
            "input_rows": int(len(raw)),
            "removed_strict_co2_violations": int(co2_violation.sum()),
            "output_rows": int(len(processed)),
        },
    }


def model_info(processed: pd.DataFrame) -> dict[str, Any]:
    metadata = Metadata.detect_from_dataframe(
        data=processed,
        table_name="table",
        infer_keys=None,
    ).to_dict()
    col_types = {}
    for column in processed.columns:
        col_types[column] = {
            "type": "cat" if column in CATEGORICAL_COLUMNS else "num",
            "unique_values": int(processed[column].nunique(dropna=True)),
            "missing_values": int(processed[column].isna().sum()),
        }

    return {
        "name": "steel",
        "source": DATASET_URL,
        "task": "regression",
        "target": TARGET,
        "features_by_order": list(processed.columns),
        "shape": list(processed.shape),
        "col_types": col_types,
        "sdv_metadata": metadata,
        "preprocessing_note": (
            "Removed rows violating CO2(tCO2) == "
            "round(Usage_kWh * 0.0004585, 2), then dropped the unique date "
            "column. NSM, Day_of_week, and WeekStatus retain modelable temporal "
            "information. See raw/preprocessing_info.json for the full audit."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_writable(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {rendered}; use --force")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    source = resolve_input(args.input)
    destination = resolve_named_output(args.output, OUTPUT_FILENAME)
    metadata_path = resolve_named_output(
        args.metadata_output or source.parent,
        PREPROCESSING_INFO_FILENAME,
    )
    info_path = resolve_named_output(
        args.info_output or destination.parent,
        INFO_FILENAME,
    )

    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    ensure_writable([destination, metadata_path, info_path], args.force)

    raw = pd.read_csv(source, encoding="utf-8-sig")
    validate_source(raw)

    expected_co2 = strict_co2_expected(raw["Usage_kWh"])
    co2_violation = raw["CO2(tCO2)"].sub(expected_co2).abs().gt(1e-9)

    expected_lagging = rounded_power_factor(
        raw["Usage_kWh"],
        raw["Lagging_Current_Reactive.Power_kVarh"],
    )
    expected_leading = rounded_power_factor(
        raw["Usage_kWh"],
        raw["Leading_Current_Reactive_Power_kVarh"],
    )
    lagging_violations = int(
        raw["Lagging_Current_Power_Factor"]
        .sub(expected_lagging)
        .abs()
        .gt(1e-9)
        .sum()
    )
    leading_violations = int(
        raw["Leading_Current_Power_Factor"]
        .sub(expected_leading)
        .abs()
        .gt(1e-9)
        .sum()
    )

    processed = (
        raw.loc[~co2_violation]
        .rename(
            columns={
                "Lagging_Current_Reactive.Power_kVarh":
                    "Lagging_Current_Reactive_Power_kVarh",
            }
        )
        .loc[:, OUTPUT_COLUMNS]
        .copy()
        .reset_index(drop=True)
    )
    for column in CATEGORICAL_COLUMNS:
        processed[column] = processed[column].astype("object")
    processed["NSM"] = processed["NSM"].astype("int64")

    temporary_csv = destination.with_name(f".{destination.name}.partial")
    processed.to_csv(temporary_csv, index=False)
    temporary_csv.replace(destination)

    write_json(
        metadata_path,
        preprocessing_metadata(
            source,
            destination,
            raw,
            processed,
            co2_violation,
            expected_co2,
            lagging_violations,
            leading_violations,
        ),
    )
    write_json(info_path, model_info(processed))

    print(f"Read {len(raw):,} rows from {source}")
    print(f"Removed {int(co2_violation.sum()):,} strict CO2 violations")
    print(f"Wrote {len(processed):,} rows to {destination}")
    print(f"Wrote preprocessing audit to {metadata_path}")
    print(f"Wrote model metadata to {info_path}")


if __name__ == "__main__":
    main()
