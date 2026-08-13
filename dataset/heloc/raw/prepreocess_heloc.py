"""Preprocess the FICO HELOC dataset into the repository schema.

The cleaning removes:

1. Rows whose 23 predictors are all the ``-9`` sentinel.
2. Rows violating any selected total-trade constraint from
   ``clean_total_trade_constraints.py``.

Examples:
    uv run python dataset/heloc/raw/prepreocess_heloc.py --force
    uv run python dataset/heloc/raw/prepreocess_heloc.py dataset/heloc/raw --force
    uv run python dataset/heloc/raw/prepreocess_heloc.py /tmp/heloc.csv \
        --output /tmp/data.csv --info-output /tmp/info.json --force

The input may be a directory containing a HELOC CSV or an explicit CSV path.
Output arguments may likewise be directories or explicit filenames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RAW_DIR = Path(__file__).resolve().parent
DATASET_DIR = RAW_DIR.parent
INPUT_FILENAME = "heloc_dataset_v1.csv"
OUTPUT_FILENAME = "data.csv"
INFO_FILENAME = "info.json"
DEFAULT_INPUT = RAW_DIR
DEFAULT_OUTPUT = DATASET_DIR / OUTPUT_FILENAME
DEFAULT_INFO_OUTPUT = DATASET_DIR / INFO_FILENAME

DATASET_HANDLE = "mstz/heloc"
DATASET_URL = "https://huggingface.co/datasets/mstz/heloc"
TARGET = "is_at_risk"
ALL_RECORDS_MISSING_SENTINEL = -9
SENTINEL_VALUES = (-9, -8, -7)
TOTAL_TRADE_COLUMN = "nr_total_trades"

# Keep this tuple aligned with dataset/heloc/clean_total_trade_constraints.py.
CONSTRAINED_COLUMNS = (
    "number_of_satisfactory_trades",
    "nr_trades_initiated_in_last_year",
    "nr_revolving_trades_with_balance",
    "nr_installment_trades_with_balance",
)

RAW_COLUMNS = (
    "RiskPerformance",
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
)

COLUMN_RENAMES = {
    "ExternalRiskEstimate": "estimate_of_risk",
    "MSinceOldestTradeOpen": "months_since_first_trade",
    "MSinceMostRecentTradeOpen": "months_since_last_trade",
    "AverageMInFile": "average_duration_of_resolution",
    "NumSatisfactoryTrades": "number_of_satisfactory_trades",
    "NumTrades60Ever2DerogPubRec": "nr_trades_insolvent_for_over_60_days",
    "NumTrades90Ever2DerogPubRec": "nr_trades_insolvent_for_over_90_days",
    "PercentTradesNeverDelq": "percentage_of_legal_trades",
    "MSinceMostRecentDelq": "months_since_last_illegal_trade",
    "MaxDelq2PublicRecLast12M": "maximum_illegal_trades_over_last_year",
    "MaxDelqEver": "maximum_illegal_trades",
    "NumTotalTrades": "nr_total_trades",
    "NumTradesOpeninLast12M": "nr_trades_initiated_in_last_year",
    "PercentInstallTrades": "percentage_of_installment_trades",
    "MSinceMostRecentInqexcl7days": "months_since_last_inquiry_not_recent",
    "NumInqLast6M": "nr_inquiries_in_last_6_months",
    "NumInqLast6Mexcl7days": "nr_inquiries_in_last_6_months_not_recent",
    "NetFractionRevolvingBurden": "net_fraction_of_revolving_burden",
    "NetFractionInstallBurden": "net_fraction_of_installment_burden",
    "NumRevolvingTradesWBalance": "nr_revolving_trades_with_balance",
    "NumInstallTradesWBalance": "nr_installment_trades_with_balance",
    "NumBank2NatlTradesWHighUtilization": "nr_banks_with_high_ratio",
    "PercentTradesWBalance": "percentage_trades_with_balance",
}

PREDICTOR_COLUMNS = tuple(COLUMN_RENAMES.values())
OUTPUT_COLUMNS = (*PREDICTOR_COLUMNS, TARGET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input CSV or directory containing heloc_dataset_v1.csv. "
            "Defaults to dataset/heloc/raw."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Cleaned CSV filename or directory. "
            "Defaults to dataset/heloc/data.csv."
        ),
    )
    parser.add_argument(
        "--info-output",
        type=Path,
        default=DEFAULT_INFO_OUTPUT,
        help=(
            "Model metadata JSON filename or directory. "
            "Defaults to dataset/heloc/info.json."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing data.csv and info.json outputs.",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    """Resolve an explicit CSV or find the sole/preferred CSV in a directory."""
    path = path.expanduser()
    if not path.is_dir():
        return path

    preferred = path / INPUT_FILENAME
    if preferred.is_file():
        return preferred

    candidates = sorted(path.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return preferred
    raise ValueError(
        f"Multiple CSV files found under {path}; pass one explicitly: "
        f"{[str(candidate) for candidate in candidates]}"
    )


def resolve_named_output(path: Path, filename: str) -> Path:
    path = path.expanduser()
    return path if path.suffix.lower() in {".csv", ".json"} else path / filename


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


def normalize_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Return integer-valued data in the repository's canonical column order."""
    actual = tuple(raw.columns)
    if actual == RAW_COLUMNS:
        target = raw["RiskPerformance"].astype("string").str.strip().str.lower()
        unknown = sorted(target.dropna().loc[~target.isin({"good", "bad"})].unique())
        if unknown:
            raise ValueError(f"Unexpected RiskPerformance values: {unknown}")
        if target.isna().any():
            raise ValueError("RiskPerformance contains missing values.")

        normalized = raw.drop(columns="RiskPerformance").rename(
            columns=COLUMN_RENAMES
        )
        normalized[TARGET] = target.map({"good": 0, "bad": 1})
    elif actual == OUTPUT_COLUMNS:
        normalized = raw.copy()
    else:
        raise ValueError(
            "Unexpected HELOC columns. Expected either the original FICO schema "
            f"{list(RAW_COLUMNS)} or repository schema {list(OUTPUT_COLUMNS)}; "
            f"got {list(actual)}."
        )

    normalized = normalized.loc[:, OUTPUT_COLUMNS].copy()
    for column in OUTPUT_COLUMNS:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        invalid = numeric.isna() & normalized[column].notna()
        if invalid.any():
            examples = normalized.loc[invalid, column].astype(str).unique()[:5]
            raise ValueError(
                f"Column {column!r} contains non-numeric values: {examples.tolist()}"
            )
        if numeric.isna().any():
            raise ValueError(f"Column {column!r} contains missing values.")
        if not numeric.mod(1).eq(0).all():
            raise ValueError(f"Column {column!r} contains non-integral values.")
        normalized[column] = numeric.astype("int64")

    target_values = set(normalized[TARGET].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{TARGET} must contain only 0 and 1; got {sorted(target_values)}."
        )
    return normalized


def all_sentinel_mask(data: pd.DataFrame) -> pd.Series:
    """Identify records with no bureau information in any predictor."""
    return data.loc[:, PREDICTOR_COLUMNS].eq(
        ALL_RECORDS_MISSING_SENTINEL
    ).all(axis=1)


def total_trade_violation_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the four masks from clean_total_trade_constraints.py."""
    return {
        column: data[TOTAL_TRADE_COLUMN].lt(data[column])
        for column in CONSTRAINED_COLUMNS
    }


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = normalize_schema(raw)
    sentinel_mask = all_sentinel_mask(normalized)
    after_sentinel = normalized.loc[~sentinel_mask].copy()

    constraint_masks = total_trade_violation_masks(after_sentinel)
    any_constraint_violation = pd.concat(
        constraint_masks.values(), axis=1
    ).any(axis=1)
    processed = (
        after_sentinel.loc[~any_constraint_violation, OUTPUT_COLUMNS]
        .copy()
        .reset_index(drop=True)
    )

    remaining_sentinel_mask = processed.loc[:, PREDICTOR_COLUMNS].isin(
        SENTINEL_VALUES
    )
    audit = {
        "input_rows": int(len(normalized)),
        "removed_all_predictors_minus_9": int(sentinel_mask.sum()),
        "constraint_violations": {
            column: int(mask.sum())
            for column, mask in constraint_masks.items()
        },
        "removed_total_trade_constraint_union": int(
            any_constraint_violation.sum()
        ),
        "output_rows": int(len(processed)),
        "remaining_rows_with_any_sentinel": int(
            remaining_sentinel_mask.any(axis=1).sum()
        ),
        "remaining_sentinel_cells": int(remaining_sentinel_mask.sum().sum()),
    }
    return processed, audit


def model_info(processed: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    columns_metadata = {
        column: {"sdtype": "categorical" if column == TARGET else "numerical"}
        for column in OUTPUT_COLUMNS
    }
    col_types = {
        column: {
            "type": "cat" if column == TARGET else "num",
            "unique_values": int(processed[column].nunique(dropna=True)),
            "missing_values": int(processed[column].isna().sum()),
        }
        for column in OUTPUT_COLUMNS
    }
    return {
        "name": "heloc",
        "source": DATASET_URL,
        "task": "binary_classification",
        "target": TARGET,
        "features_by_order": list(OUTPUT_COLUMNS),
        "shape": list(processed.shape),
        "col_types": col_types,
        "sdv_metadata": {
            "tables": {"table": {"columns": columns_metadata}},
            "relationships": [],
            "METADATA_SPEC_VERSION": "V1",
        },
        "preprocessing": {
            "download": {
                "provider": "Hugging Face",
                "dataset_handle": DATASET_HANDLE,
                "file": INPUT_FILENAME,
            },
            "all_sentinel_rule": (
                "Remove a row when every one of its 23 predictors equals -9."
            ),
            "total_trade_rules": [
                f"{TOTAL_TRADE_COLUMN} >= {column}"
                for column in CONSTRAINED_COLUMNS
            ],
            "row_counts": audit,
        },
        "preprocessing_note": (
            "Removed rows whose 23 predictors all equal -9, then removed rows "
            "violating any selected total-trade constraint. Other -9, -8, and "
            "-7 sentinel values are preserved."
        ),
    }


def main() -> None:
    args = parse_args()
    source = resolve_input(args.input)
    destination = resolve_named_output(args.output, OUTPUT_FILENAME)
    info_path = resolve_named_output(args.info_output, INFO_FILENAME)

    if not source.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {source}. Run download_heloc.py first."
        )
    ensure_writable([destination, info_path], args.force)

    raw = pd.read_csv(source, encoding="utf-8-sig")
    processed, audit = preprocess(raw)

    temporary_csv = destination.with_name(f".{destination.name}.partial")
    processed.to_csv(temporary_csv, index=False)
    temporary_csv.replace(destination)
    write_json(info_path, model_info(processed, audit))

    print(f"Read {audit['input_rows']:,} rows from {source}")
    print(
        "Removed "
        f"{audit['removed_all_predictors_minus_9']:,} all-(-9) sentinel rows"
    )
    print(
        "Removed "
        f"{audit['removed_total_trade_constraint_union']:,} rows violating "
        "selected total-trade constraints"
    )
    print(f"Wrote {audit['output_rows']:,} rows to {destination}")
    print(f"Wrote model metadata to {info_path}")


if __name__ == "__main__":
    main()
