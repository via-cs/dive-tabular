"""Preprocess UCI Online News Popularity for tabular synthesis.

Examples:
    uv run python dataset/news/raw/preprocess_news.py --force
    uv run python dataset/news/raw/preprocess_news.py dataset/news/raw --force
    uv run python dataset/news/raw/preprocess_news.py /tmp/news.csv \
        --output /tmp/data.csv \
        --metadata-output /tmp/preprocessing_info.json \
        --info-output /tmp/info.json \
        --force
    uv run python dataset/news/raw/preprocess_news.py dataset/news/raw \
        --input-file /tmp/news.csv --output /tmp/news-output --force

The input may be a directory containing ``OnlineNewsPopularity.csv`` or an
explicit CSV path. ``--input-file`` explicitly overrides positional directory
discovery. Output arguments accept either directories or exact filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_URL = (
    "https://archive.ics.uci.edu/dataset/332/online+news+popularity"
)
DATASET_DOI = "https://doi.org/10.24432/C5NS3V"
INPUT_FILENAME = "OnlineNewsPopularity.csv"
OUTPUT_FILENAME = "data.csv"
PREPROCESSING_INFO_FILENAME = "preprocessing_info.json"
INFO_FILENAME = "info.json"
RAW_TARGET = "shares"
TARGET = "is_popular"
POPULARITY_THRESHOLD = 1400
EQUATION_TOLERANCE = 1e-9

RAW_DIRECTORY = Path(__file__).resolve().parent
DATASET_DIRECTORY = RAW_DIRECTORY.parent
DEFAULT_INPUT = RAW_DIRECTORY
DEFAULT_OUTPUT = DATASET_DIRECTORY / OUTPUT_FILENAME

RAW_COLUMNS = (
    "url",
    "timedelta",
    "n_tokens_title",
    "n_tokens_content",
    "n_unique_tokens",
    "n_non_stop_words",
    "n_non_stop_unique_tokens",
    "num_hrefs",
    "num_self_hrefs",
    "num_imgs",
    "num_videos",
    "average_token_length",
    "num_keywords",
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
    "kw_min_min",
    "kw_max_min",
    "kw_avg_min",
    "kw_min_max",
    "kw_max_max",
    "kw_avg_max",
    "kw_min_avg",
    "kw_max_avg",
    "kw_avg_avg",
    "self_reference_min_shares",
    "self_reference_max_shares",
    "self_reference_avg_sharess",
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "weekday_is_saturday",
    "weekday_is_sunday",
    "is_weekend",
    "LDA_00",
    "LDA_01",
    "LDA_02",
    "LDA_03",
    "LDA_04",
    "global_subjectivity",
    "global_sentiment_polarity",
    "global_rate_positive_words",
    "global_rate_negative_words",
    "rate_positive_words",
    "rate_negative_words",
    "avg_positive_polarity",
    "min_positive_polarity",
    "max_positive_polarity",
    "avg_negative_polarity",
    "min_negative_polarity",
    "max_negative_polarity",
    "title_subjectivity",
    "title_sentiment_polarity",
    "abs_title_subjectivity",
    "abs_title_sentiment_polarity",
    RAW_TARGET,
)

WEEKDAY_COLUMNS = {
    "weekday_is_monday": "monday",
    "weekday_is_tuesday": "tuesday",
    "weekday_is_wednesday": "wednesday",
    "weekday_is_thursday": "thursday",
    "weekday_is_friday": "friday",
    "weekday_is_saturday": "saturday",
    "weekday_is_sunday": "sunday",
}
WEEKDAY_VALUES = tuple(WEEKDAY_COLUMNS.values())
WEEKEND_VALUES = {"saturday", "sunday"}

DATA_CHANNEL_COLUMNS = (
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world",
)

CATEGORICAL_COLUMNS = (
    *DATA_CHANNEL_COLUMNS,
    "week_day",
    "is_weekend",
    TARGET,
)

_output_columns = [
    column
    for column in RAW_COLUMNS
    if (
        column not in {"url", RAW_TARGET}
        and column not in WEEKDAY_COLUMNS
    )
]
_output_columns.insert(_output_columns.index("is_weekend"), "week_day")
_output_columns.append(TARGET)
OUTPUT_COLUMNS = tuple(_output_columns)
NUMERICAL_COLUMNS = tuple(
    column for column in OUTPUT_COLUMNS if column not in CATEGORICAL_COLUMNS
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input CSV or directory containing OnlineNewsPopularity.csv. "
            "Defaults to dataset/news/raw."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Exact source CSV override for positional directory discovery.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Processed CSV filename or directory. Defaults to "
            "dataset/news/data.csv."
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help=(
            "Preprocessing audit filename or directory. Defaults to "
            "preprocessing_info.json beside the raw CSV."
        ),
    )
    parser.add_argument(
        "--info-output",
        type=Path,
        default=None,
        help=(
            "Model metadata filename or directory. Defaults to info.json "
            "beside the processed CSV."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing processed CSV and metadata outputs.",
    )
    return parser.parse_args(argv)


def resolve_input(path: Path, input_file: Path | None = None) -> Path:
    source = (input_file or path).expanduser()
    if source.is_file():
        return source
    candidates = (
        source / INPUT_FILENAME,
        source / "OnlineNewsPopularity" / INPUT_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Input must be {INPUT_FILENAME} or a directory containing it: {source}"
    )


def resolve_named_output(path: Path, filename: str, suffix: str) -> Path:
    path = path.expanduser()
    return path if path.suffix.lower() == suffix else path / filename


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
    staging = path.with_name(f".{path.name}.partial")
    staging.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    staging.replace(path)


def validate_source(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data.columns = data.columns.str.strip()
    actual = tuple(data.columns)
    if actual != RAW_COLUMNS:
        raise ValueError(
            "Unexpected raw columns. "
            f"Expected {list(RAW_COLUMNS)}, got {list(actual)}."
        )
    if data.isna().any().any():
        raise ValueError("The official source unexpectedly contains missing values.")

    for column in RAW_COLUMNS:
        if column != "url":
            data[column] = pd.to_numeric(data[column], errors="raise")

    weekday_columns = list(WEEKDAY_COLUMNS)
    if not data[weekday_columns].isin([0, 1]).all().all():
        raise ValueError("Weekday indicator columns must be binary.")
    if not data[weekday_columns].sum(axis=1).eq(1).all():
        raise ValueError("Every source row must have exactly one active weekday.")

    if not data[list(DATA_CHANNEL_COLUMNS)].isin([0, 1]).all().all():
        raise ValueError("Data-channel indicator columns must be binary.")
    if data[list(DATA_CHANNEL_COLUMNS)].sum(axis=1).gt(1).any():
        raise ValueError("A source row has multiple active data channels.")

    if not data["is_weekend"].isin([0, 1]).all():
        raise ValueError("is_weekend must be binary.")
    expected_weekend = (
        data["weekday_is_saturday"] + data["weekday_is_sunday"]
    )
    if not data["is_weekend"].eq(expected_weekend).all():
        raise ValueError("is_weekend disagrees with the source weekday flags.")
    return data


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = validate_source(raw)
    positive_zero = data["rate_positive_words"].eq(0)
    negative_zero = data["rate_negative_words"].eq(0)
    both_rates_zero = positive_zero & negative_zero

    weekday_flags = data[list(WEEKDAY_COLUMNS)]
    data["week_day"] = weekday_flags.idxmax(axis=1).map(WEEKDAY_COLUMNS)
    data[TARGET] = data[RAW_TARGET].ge(POPULARITY_THRESHOLD).astype(int)
    data = data.loc[~both_rates_zero].copy()
    processed = data.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)

    processed["is_weekend"] = processed["is_weekend"].astype(int)
    for column in DATA_CHANNEL_COLUMNS:
        processed[column] = processed[column].astype(int)
    processed[TARGET] = processed[TARGET].astype(int)

    rate_sum_error = (
        processed["rate_positive_words"]
        + processed["rate_negative_words"]
        - 1.0
    ).abs()
    lda_columns = [f"LDA_0{index}" for index in range(5)]
    lda_sum_error = (processed[lda_columns].sum(axis=1) - 1.0).abs()
    title_polarity_error = (
        processed["abs_title_sentiment_polarity"]
        - processed["title_sentiment_polarity"].abs()
    ).abs()
    title_subjectivity_error = (
        processed["abs_title_subjectivity"]
        - (processed["title_subjectivity"] - 0.5).abs()
    ).abs()
    expected_weekend = processed["week_day"].isin(WEEKEND_VALUES).astype(int)

    if processed.isna().any().any():
        raise ValueError("Processed news data unexpectedly contains missing values.")
    if not processed["week_day"].isin(WEEKDAY_VALUES).all():
        raise ValueError("Processed news data contains an unknown week_day.")
    if not processed["is_weekend"].eq(expected_weekend).all():
        raise ValueError("week_day does not determine is_weekend.")
    if not processed[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must be binary.")
    if processed[TARGET].nunique() != 2:
        raise ValueError(f"{TARGET} must contain both classes.")
    if rate_sum_error.gt(EQUATION_TOLERANCE).any():
        raise ValueError("Processed data violates the sentiment-rate equation.")
    if lda_sum_error.gt(EQUATION_TOLERANCE).any():
        raise ValueError("Processed data violates the LDA topic-sum equation.")
    if title_polarity_error.gt(EQUATION_TOLERANCE).any():
        raise ValueError("Processed data violates absolute title polarity.")
    if title_subjectivity_error.gt(EQUATION_TOLERANCE).any():
        raise ValueError("Processed data violates absolute title subjectivity.")

    audit = {
        "positive_rate_zero_rows": int(positive_zero.sum()),
        "negative_rate_zero_rows": int(negative_zero.sum()),
        "both_rates_zero_rows_removed": int(both_rates_zero.sum()),
        "positive_rate_zero_only_rows_retained": int(
            (positive_zero & ~negative_zero).sum()
        ),
        "negative_rate_zero_only_rows_retained": int(
            (~positive_zero & negative_zero).sum()
        ),
        "unassigned_data_channel_rows": int(
            data[list(DATA_CHANNEL_COLUMNS)].sum(axis=1).eq(0).sum()
        ),
        "popularity_threshold_shares": POPULARITY_THRESHOLD,
        "not_popular_rows": int(processed[TARGET].eq(0).sum()),
        "popular_rows": int(processed[TARGET].eq(1).sum()),
        "popular_fraction": float(processed[TARGET].mean()),
        "final_rate_sum_max_absolute_error": float(rate_sum_error.max()),
        "final_lda_sum_max_absolute_error": float(lda_sum_error.max()),
        "final_title_polarity_max_absolute_error": float(
            title_polarity_error.max()
        ),
        "final_title_subjectivity_max_absolute_error": float(
            title_subjectivity_error.max()
        ),
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
            "provider": "UCI Machine Learning Repository",
            "dataset": "Online News Popularity",
            "dataset_url": DATASET_URL,
            "doi": DATASET_DOI,
            "license": "CC BY 4.0",
            "file": INPUT_FILENAME,
        },
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(destination.resolve()),
        "preprocessing_command": (
            "uv run python dataset/news/raw/preprocess_news.py --force"
        ),
        "preprocessing_steps": [
            {
                "order": 1,
                "operation": "normalize_and_validate_source_schema",
                "details": (
                    "Strip whitespace from UCI column names, require the official "
                    "61-column schema, coerce non-URL fields to numeric values, "
                    "and reject missing data."
                ),
            },
            {
                "order": 2,
                "operation": "validate_source_categorical_indicators",
                "details": (
                    "Require exactly one active weekday, at most one active data "
                    "channel, binary indicator values, and agreement between the "
                    "weekday flags and is_weekend."
                ),
            },
            {
                "order": 3,
                "operation": "collapse_weekday_indicators",
                "details": (
                    "Replace the seven weekday_is_* columns with one categorical "
                    "week_day column containing monday through sunday."
                ),
            },
            {
                "order": 4,
                "operation": "drop_rows_without_polarized_words",
                "details": (
                    "Remove rows where both rate_positive_words and "
                    "rate_negative_words are zero. Retain rows where exactly one "
                    "rate is zero because their valid rate pair is (0, 1) or (1, 0)."
                ),
            },
            {
                "order": 5,
                "operation": "derive_binary_popularity_target",
                "details": (
                    f"Create {TARGET}=1 when shares is at least "
                    f"{POPULARITY_THRESHOLD}, otherwise 0, then drop raw shares "
                    "to prevent target leakage."
                ),
            },
            {
                "order": 6,
                "operation": "drop_url",
                "details": "Drop the article URL identifier.",
            },
            {
                "order": 7,
                "operation": "validate_semantic_equations",
                "details": (
                    "Require week_day to determine is_weekend and validate the "
                    "sentiment-rate sum, LDA topic sum, and absolute-title equations."
                ),
            },
            {
                "order": 8,
                "operation": "reorder_and_reset_index",
                "details": (
                    "Preserve source feature order, place week_day immediately "
                    f"before is_weekend, place {TARGET} as the final binary "
                    "classification target, and reset the row index."
                ),
            },
        ],
        "equations": {
            "sentiment_rate_sum": (
                "rate_positive_words + rate_negative_words == 1"
            ),
            "lda_topic_sum": "LDA_00 + LDA_01 + LDA_02 + LDA_03 + LDA_04 == 1",
            "absolute_title_polarity": (
                "abs_title_sentiment_polarity == abs(title_sentiment_polarity)"
            ),
            "absolute_title_subjectivity": (
                "abs_title_subjectivity == abs(title_subjectivity - 0.5)"
            ),
            "tolerance": EQUATION_TOLERANCE,
        },
        "categorical_selection": {
            "columns": list(CATEGORICAL_COLUMNS),
            "rationale": {
                "week_day": (
                    "Nominal seven-level publication weekday derived from the "
                    "source one-hot indicators."
                ),
                "is_weekend": (
                    "Binary categorical status deterministically dependent on "
                    "week_day."
                ),
                "data_channel_is_*": (
                    "Six source binary category-membership indicators; all-zero "
                    "means the article is outside the named channels."
                ),
                TARGET: (
                    f"Binary popularity target derived as shares >= "
                    f"{POPULARITY_THRESHOLD}; treated categorically for "
                    "classification and stratified train/test splitting."
                ),
                "num_keywords": (
                    "Kept numerical rather than categorical because it is a count, "
                    "despite having only ten observed values."
                ),
            },
        },
        "audit": audit,
        "columns": {
            "source": list(RAW_COLUMNS),
            "output": list(OUTPUT_COLUMNS),
            "categorical": list(CATEGORICAL_COLUMNS),
            "numerical": list(NUMERICAL_COLUMNS),
            "target": TARGET,
            "removed": ["url", RAW_TARGET, *WEEKDAY_COLUMNS],
            "derived": ["week_day", TARGET],
        },
        "row_counts": {
            "input_rows": int(len(raw)),
            "removed_both_sentiment_rates_zero": audit[
                "both_rates_zero_rows_removed"
            ],
            "output_rows": int(len(processed)),
        },
    }


def model_info(processed: pd.DataFrame) -> dict[str, Any]:
    col_types = {
        column: {
            "type": "cat" if column in CATEGORICAL_COLUMNS else "num",
            "unique_values": int(processed[column].nunique(dropna=True)),
            "missing_values": int(processed[column].isna().sum()),
        }
        for column in processed.columns
    }
    metadata_columns = {
        column: {
            "sdtype": (
                "categorical"
                if column in CATEGORICAL_COLUMNS
                else "numerical"
            )
        }
        for column in processed.columns
    }
    return {
        "name": "news",
        "source": DATASET_URL,
        "doi": DATASET_DOI,
        "license": "CC BY 4.0",
        "task": "binary_classification",
        "target": TARGET,
        "features_by_order": list(processed.columns),
        "shape": list(processed.shape),
        "col_types": col_types,
        "sdv_metadata": {
            "tables": {"table": {"columns": metadata_columns}},
            "relationships": [],
            "METADATA_SPEC_VERSION": "V1",
        },
        "preprocessing_note": (
            "Downloaded from UCI; dropped the URL identifier; collapsed the seven "
            "weekday indicators into categorical week_day; removed rows where both "
            "positive and negative non-neutral-token rates were zero; retained "
            "is_weekend and the six data-channel indicators as categorical columns. "
            f"derived categorical {TARGET} from shares >= {POPULARITY_THRESHOLD} "
            "and removed raw shares to prevent leakage. num_keywords remains "
            "numerical because it is a count. See "
            "raw/preprocessing_info.json for the full audit."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = resolve_input(args.input, args.input_file)
    destination = resolve_named_output(args.output, OUTPUT_FILENAME, ".csv")
    metadata_path = resolve_named_output(
        args.metadata_output or source.parent,
        PREPROCESSING_INFO_FILENAME,
        ".json",
    )
    info_path = resolve_named_output(
        args.info_output or destination.parent,
        INFO_FILENAME,
        ".json",
    )

    ensure_writable([destination, metadata_path, info_path], args.force)
    raw = pd.read_csv(source, encoding="utf-8-sig", skipinitialspace=True)
    processed, audit = preprocess(raw)

    staging = destination.with_name(f".{destination.name}.partial")
    processed.to_csv(staging, index=False)
    staging.replace(destination)
    write_json(
        metadata_path,
        preprocessing_metadata(source, destination, raw, processed, audit),
    )
    write_json(info_path, model_info(processed))

    print(f"Read {len(raw):,} rows from {source}")
    print(
        "Removed "
        f"{audit['both_rates_zero_rows_removed']:,} rows where both sentiment "
        "rates were zero"
    )
    print(
        f"Derived {TARGET} using shares >= {POPULARITY_THRESHOLD}: "
        f"{audit['popular_rows']:,} popular / "
        f"{audit['not_popular_rows']:,} not popular"
    )
    print(f"Wrote {len(processed):,} rows and {len(processed.columns)} columns")
    print(f"Data: {destination}")
    print(f"Preprocessing audit: {metadata_path}")
    print(f"Model metadata: {info_path}")


if __name__ == "__main__":
    main()
