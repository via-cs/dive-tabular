"""Preprocess the 2015 NYC Green Taxi dataset.

Examples:
    uv run python data/taxi/preprocess_taxi.py --force
    uv run python data/taxi/preprocess_taxi.py data/taxi/raw --force
    uv run python data/taxi/preprocess_taxi.py /tmp/taxi.csv \
        --output /tmp/taxi-clean.csv \
        --metadata-output /tmp/preprocessing_info.json \
        --info-output /tmp/info.json \
        --force

The input may be a directory containing ``2015_green_60000.csv`` or an
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


DATASET_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
INPUT_FILENAME = "2015_green_60000.csv"
OUTPUT_FILENAME = "data.csv"
PREPROCESSING_INFO_FILENAME = "preprocessing_info.json"
INFO_FILENAME = "info.json"

DEFAULT_INPUT = Path("data/taxi/raw")
DEFAULT_OUTPUT = Path("data/taxi/data.csv")
TARGET = "total_amount"
MINUTES_PER_DAY = 1440.0
EQUATION_TOLERANCE = 1e-9

RAW_COLUMNS = (
    "Unnamed: 0",
    "vendorid",
    "pickup_datetime",
    "dropoff_datetime",
    "Store_and_fwd_flag",
    "rate_code",
    "Pickup_longitude",
    "Pickup_latitude",
    "Dropoff_longitude",
    "Dropoff_latitude",
    "Passenger_count",
    "Trip_distance",
    "Fare_amount",
    "Extra",
    "MTA_tax",
    "Tip_amount",
    "Tolls_amount",
    "Ehail_fee",
    "Improvement_surcharge",
    "Total_amount",
    "Payment_type",
    "Trip_type",
)

COLUMN_RENAMES = {
    "Store_and_fwd_flag": "store_and_fwd_flag",
    "Pickup_longitude": "pickup_longitude",
    "Pickup_latitude": "pickup_latitude",
    "Dropoff_longitude": "dropoff_longitude",
    "Dropoff_latitude": "dropoff_latitude",
    "Passenger_count": "passenger_count",
    "Trip_distance": "trip_distance",
    "Fare_amount": "fare_amount",
    "Extra": "extra",
    "MTA_tax": "mta_tax",
    "Tip_amount": "tip_amount",
    "Tolls_amount": "tolls_amount",
    "Ehail_fee": "ehail_fee",
    "Improvement_surcharge": "improvement_surcharge",
    "Total_amount": "total_amount",
    "Payment_type": "payment_type",
    "Trip_type": "trip_type",
}

PAYMENT_COMPONENTS = (
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
)

CATEGORICAL_COLUMNS = (
    "store_and_fwd_flag",
    "rate_code",
    "passenger_count",
    "payment_type",
    "trip_type",
    "pickup_day_of_week",
    "is_weekend",
)

NUMERICAL_COLUMNS = (
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_distance",
    *PAYMENT_COMPONENTS,
    "pickup_minute_of_day",
    "dropoff_minute_of_day",
    "duration",
    TARGET,
)

OUTPUT_COLUMNS = (
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_distance",
    *PAYMENT_COMPONENTS,
    "pickup_minute_of_day",
    "dropoff_minute_of_day",
    "duration",
    *CATEGORICAL_COLUMNS,
    TARGET,
)

REQUIRED_AFTER_DROPS = tuple(
    column for column in OUTPUT_COLUMNS if column != TARGET
) + (TARGET,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input CSV or directory containing 2015_green_60000.csv. "
            "Defaults to data/taxi/raw."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Cleaned CSV filename or directory. Defaults to data/taxi/data.csv.",
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
    return path / INPUT_FILENAME if path.is_dir() else path


def resolve_named_output(path: Path, filename: str) -> Path:
    path = path.expanduser()
    return path if path.suffix.lower() in {".csv", ".json"} else path / filename


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_writable(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {rendered}; use --force")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_source(raw: pd.DataFrame) -> None:
    actual = tuple(raw.columns)
    if actual != RAW_COLUMNS:
        raise ValueError(
            "Unexpected raw columns. "
            f"Expected {list(RAW_COLUMNS)}, got {list(actual)}."
        )


def minute_of_day(timestamp: pd.Series) -> pd.Series:
    """Return fractional minute-of-day values, preserving source seconds."""
    return (
        timestamp.dt.hour.astype(float) * 60.0
        + timestamp.dt.minute.astype(float)
        + timestamp.dt.second.astype(float) / 60.0
    )


def modular_error(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    """Return the nearest signed cyclic clock error in minutes."""
    return ((lhs - rhs + MINUTES_PER_DAY / 2) % MINUTES_PER_DAY) - (
        MINUTES_PER_DAY / 2
    )


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = raw.rename(columns=COLUMN_RENAMES).copy()
    pickup = pd.to_datetime(
        data["pickup_datetime"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    dropoff = pd.to_datetime(
        data["dropoff_datetime"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )

    numeric_source_columns = (
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_distance",
        *PAYMENT_COMPONENTS,
        "ehail_fee",
        TARGET,
        "payment_type",
        "trip_type",
        "rate_code",
    )
    for column in numeric_source_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["pickup_day_of_week"] = pickup.dt.dayofweek
    data["pickup_minute_of_day"] = minute_of_day(pickup)
    data["duration"] = (dropoff - pickup).dt.total_seconds().div(60.0)
    data["dropoff_minute_of_day"] = (
        data["pickup_minute_of_day"] + data["duration"]
    ) % MINUTES_PER_DAY
    data["is_weekend"] = data["pickup_day_of_week"].ge(5).astype("Int64")

    raw_dropoff_minute = minute_of_day(dropoff)
    raw_clock_error = modular_error(
        raw_dropoff_minute,
        data["pickup_minute_of_day"] + data["duration"],
    )

    adjusted_source_total = data[TARGET] - data["ehail_fee"].fillna(0.0)
    payment_total = data.loc[:, PAYMENT_COMPONENTS].sum(axis=1, min_count=len(PAYMENT_COMPONENTS))
    payment_error = adjusted_source_total - payment_total
    payment_comparable = adjusted_source_total.notna() & payment_total.notna()
    payment_violation = payment_comparable & payment_error.abs().gt(EQUATION_TOLERANCE)

    required_frame = data.loc[:, REQUIRED_AFTER_DROPS]
    missing_required = required_frame.isna().any(axis=1)
    valid_duration = data["duration"].gt(0.0) & data["duration"].lt(MINUTES_PER_DAY)
    invalid_duration = ~missing_required & ~valid_duration
    keep = ~missing_required & valid_duration

    # Rebuild the target after removing Ehail_fee. This both applies the requested
    # fee subtraction and repairs source rows that do not balance exactly.
    data[TARGET] = payment_total
    processed = data.loc[keep, OUTPUT_COLUMNS].copy().reset_index(drop=True)

    for column in CATEGORICAL_COLUMNS:
        processed[column] = processed[column].astype("object")

    final_payment_error = processed[TARGET] - processed.loc[:, PAYMENT_COMPONENTS].sum(axis=1)
    final_clock_error = modular_error(
        processed["dropoff_minute_of_day"],
        processed["pickup_minute_of_day"] + processed["duration"],
    )
    if processed.isna().any().any():
        raise ValueError("Processed taxi data unexpectedly contains missing values.")
    if final_payment_error.abs().gt(EQUATION_TOLERANCE).any():
        raise ValueError("Processed taxi data violates the payment equation.")
    if final_clock_error.abs().gt(EQUATION_TOLERANCE).any():
        raise ValueError("Processed taxi data violates the cyclic time equation.")

    retained_payment_violations = int((payment_violation & keep).sum())
    audit = {
        "missing_required_rows": int(missing_required.sum()),
        "invalid_duration_rows": int(invalid_duration.sum()),
        "nonpositive_duration_rows": int((~missing_required & data["duration"].le(0.0)).sum()),
        "duration_at_least_one_day_rows": int(
            (~missing_required & data["duration"].ge(MINUTES_PER_DAY)).sum()
        ),
        "cross_calendar_date_rows": int((pickup.dt.date != dropoff.dt.date).sum()),
        "raw_clock_equation_violations": int(
            (raw_clock_error.abs().gt(EQUATION_TOLERANCE) & pickup.notna() & dropoff.notna()).sum()
        ),
        "ehail_nonmissing_rows": int(data["ehail_fee"].notna().sum()),
        "ehail_nonmissing_total": float(data["ehail_fee"].dropna().sum()),
        "source_payment_equation_comparable_rows": int(payment_comparable.sum()),
        "source_payment_equation_violations": int(payment_violation.sum()),
        "retained_payment_rows_repaired": retained_payment_violations,
        "source_payment_max_absolute_error": (
            float(payment_error.loc[payment_comparable].abs().max())
            if payment_comparable.any()
            else None
        ),
        "final_payment_equation_violations": int(
            final_payment_error.abs().gt(EQUATION_TOLERANCE).sum()
        ),
        "final_clock_equation_violations": int(
            final_clock_error.abs().gt(EQUATION_TOLERANCE).sum()
        ),
        "duration_min_minutes": float(processed["duration"].min()),
        "duration_max_minutes": float(processed["duration"].max()),
    }
    return processed, audit


def preprocessing_metadata(
    source: Path,
    destination: Path,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "download": {
            "provider": "NYC Taxi and Limousine Commission",
            "dataset": "2015 Green Taxi Trip Records",
            "dataset_url": DATASET_URL,
            "file": INPUT_FILENAME,
        },
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(destination.resolve()),
        "preprocessing_command": "uv run python data/taxi/preprocess_taxi.py --force",
        "preprocessing_steps": [
            {
                "order": 1,
                "operation": "validate_source_schema",
                "details": "Require the expected 22 source columns in their original order.",
            },
            {
                "order": 2,
                "operation": "drop_identifier_and_vendor",
                "details": "Drop the unnamed row identifier and vendorid.",
            },
            {
                "order": 3,
                "operation": "parse_timestamps_and_augment_time",
                "details": (
                    "Parse pickup/drop-off timestamps; derive pickup_day_of_week, "
                    "fractional pickup/dropoff minute-of-day, duration in minutes, "
                    "and Saturday/Sunday is_weekend."
                ),
            },
            {
                "order": 4,
                "operation": "remove_ehail_fee_and_repair_payment_total",
                "details": (
                    "Subtract nonmissing Ehail_fee from the recorded Total_amount, "
                    "drop Ehail_fee, and recompute total_amount from the six retained "
                    "payment components to guarantee the accounting identity."
                ),
            },
            {
                "order": 5,
                "operation": "drop_missing_and_invalid_duration_rows",
                "details": (
                    "Drop rows missing any retained value and trips whose duration "
                    "is not strictly between 0 and 1440 minutes. Ehail_fee is excluded "
                    "from missingness because it is intentionally removed."
                ),
            },
            {
                "order": 6,
                "operation": "drop_source_timestamps",
                "details": (
                    "Drop pickup_datetime and dropoff_datetime after deriving the "
                    "retained temporal features."
                ),
            },
            {
                "order": 7,
                "operation": "reorder_and_reset_index",
                "details": (
                    "Order model features consistently, put total_amount last as "
                    "the regression target, and reset to a zero-based row index."
                ),
            },
        ],
        "equations": {
            "cyclic_time": (
                "dropoff_minute_of_day == "
                "(pickup_minute_of_day + duration) % 1440"
            ),
            "payment_total": (
                "total_amount == fare_amount + extra + mta_tax + tip_amount + "
                "tolls_amount + improvement_surcharge"
            ),
            "tolerance": EQUATION_TOLERANCE,
        },
        "audit": audit,
        "columns": {
            "source": list(RAW_COLUMNS),
            "output": list(OUTPUT_COLUMNS),
            "categorical": list(CATEGORICAL_COLUMNS),
            "numerical": list(NUMERICAL_COLUMNS),
            "target": TARGET,
            "removed": [
                "Unnamed: 0",
                "vendorid",
                "pickup_datetime",
                "dropoff_datetime",
                "Ehail_fee",
            ],
            "renamed": COLUMN_RENAMES,
            "derived": [
                "pickup_day_of_week",
                "pickup_minute_of_day",
                "dropoff_minute_of_day",
                "duration",
                "is_weekend",
            ],
        },
        "row_counts": {
            "input_rows": int(len(raw)),
            "removed_missing_required": audit["missing_required_rows"],
            "removed_invalid_duration": audit["invalid_duration_rows"],
            "output_rows": int(len(processed)),
        },
    }


def model_info(processed: pd.DataFrame) -> dict[str, Any]:
    metadata = Metadata.detect_from_dataframe(
        data=processed,
        table_name="table",
        infer_keys=None,
    ).to_dict()
    metadata_columns = metadata["tables"]["table"]["columns"]
    for column in NUMERICAL_COLUMNS:
        metadata_columns[column] = {"sdtype": "numerical"}
    for column in CATEGORICAL_COLUMNS:
        metadata_columns[column] = {"sdtype": "categorical"}

    col_types = {
        column: {
            "type": "cat" if column in CATEGORICAL_COLUMNS else "num",
            "unique_values": int(processed[column].nunique(dropna=True)),
            "missing_values": int(processed[column].isna().sum()),
        }
        for column in processed.columns
    }
    return {
        "name": "taxi",
        "source": DATASET_URL,
        "task": "regression",
        "target": TARGET,
        "features_by_order": list(processed.columns),
        "shape": list(processed.shape),
        "col_types": col_types,
        "sdv_metadata": metadata,
        "preprocessing_note": (
            "Dropped the unnamed identifier, vendorid, Ehail_fee, and source "
            "timestamps; added pickup weekday, fractional minute-of-day fields, "
            "duration in minutes, and Saturday/Sunday is_weekend. Removed missing "
            "and invalid-duration rows and rebuilt total_amount from retained "
            "payment components. See raw/preprocessing_info.json for the full audit."
        ),
    }


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
    processed, audit = preprocess(raw)

    temporary_csv = destination.with_name(f".{destination.name}.partial")
    processed.to_csv(temporary_csv, index=False)
    temporary_csv.replace(destination)
    write_json(
        metadata_path,
        preprocessing_metadata(source, destination, raw, processed, audit),
    )
    write_json(info_path, model_info(processed))

    print(f"Read {len(raw):,} rows from {source}")
    print(f"Removed {audit['missing_required_rows']:,} rows with missing retained data")
    print(f"Removed {audit['invalid_duration_rows']:,} invalid-duration rows")
    print(f"Repaired {audit['retained_payment_rows_repaired']:,} payment totals")
    print(f"Wrote {len(processed):,} rows to {destination}")
    print(f"Wrote preprocessing audit to {metadata_path}")
    print(f"Wrote model metadata to {info_path}")


if __name__ == "__main__":
    main()
