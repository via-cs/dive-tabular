"""Preprocess the Web Page Phishing Detection dataset for synthesis.

Examples:
    uv run python dataset/url/raw/preprocess_url.py --force
    uv run python dataset/url/raw/preprocess_url.py dataset/url/raw --force
    uv run python dataset/url/raw/preprocess_url.py /tmp/dataset_phishing.csv \
        --output /tmp/data.csv \
        --metadata-output /tmp/preprocessing_info.json \
        --info-output /tmp/info.json \
        --force
    uv run python dataset/url/raw/preprocess_url.py dataset/url/raw \
        --input-file /tmp/url-source.csv --output /tmp/url-output --force

The input may be a directory containing ``dataset_phishing.csv`` or an exact
CSV path. ``--input-file`` overrides positional directory discovery. Output
arguments accept either directories or exact filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DATASET_HANDLE = "shashwatwork/web-page-phishing-detection-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_HANDLE}"
MENDELEY_URL = "https://data.mendeley.com/datasets/c2gw7fy2j4/3"
DATASET_DOI = "https://doi.org/10.17632/c2gw7fy2j4.3"
INPUT_FILENAME = "dataset_phishing.csv"
OUTPUT_FILENAME = "data.csv"
PREPROCESSING_INFO_FILENAME = "preprocessing_info.json"
INFO_FILENAME = "info.json"
EXPECTED_ROWS = 11_430
TARGET = "status"
DROPPED_COLUMNS = ("url", "submit_email")

RAW_DIRECTORY = Path(__file__).resolve().parent
DATASET_DIRECTORY = RAW_DIRECTORY.parent
DEFAULT_INPUT = RAW_DIRECTORY
DEFAULT_OUTPUT = DATASET_DIRECTORY / OUTPUT_FILENAME

OUTPUT_COLUMNS = tuple(
    """
    length_url length_hostname ip nb_dots nb_hyphens nb_at nb_qm nb_and nb_or
    nb_eq nb_underscore nb_tilde nb_percent nb_slash nb_star nb_colon nb_comma
    nb_semicolumn nb_dollar nb_space nb_www nb_com nb_dslash http_in_path
    https_token ratio_digits_url ratio_digits_host punycode port tld_in_path
    tld_in_subdomain abnormal_subdomain nb_subdomains prefix_suffix random_domain
    shortening_service path_extension nb_redirection nb_external_redirection
    length_words_raw char_repeat shortest_words_raw shortest_word_host
    shortest_word_path longest_words_raw longest_word_host longest_word_path
    avg_words_raw avg_word_host avg_word_path phish_hints domain_in_brand
    brand_in_subdomain brand_in_path suspecious_tld statistical_report
    nb_hyperlinks ratio_intHyperlinks ratio_extHyperlinks ratio_nullHyperlinks
    nb_extCSS ratio_intRedirection ratio_extRedirection ratio_intErrors
    ratio_extErrors login_form external_favicon links_in_tags ratio_intMedia
    ratio_extMedia sfh iframe popup_window safe_anchor onmouseover right_clic
    empty_title domain_in_title domain_with_copyright whois_registered_domain
    domain_registration_length domain_age web_traffic dns_record google_index
    page_rank status
    """.split()
)

_raw_columns = list(OUTPUT_COLUMNS)
_raw_columns.insert(_raw_columns.index("ratio_intMedia"), "submit_email")
_raw_columns.insert(0, "url")
RAW_COLUMNS = tuple(_raw_columns)

CATEGORICAL_COLUMNS = {
    "ip",
    "nb_at",
    "nb_qm",
    "nb_or",
    "nb_tilde",
    "nb_star",
    "nb_colon",
    "nb_comma",
    "nb_dollar",
    "nb_space",
    "nb_www",
    "nb_com",
    "nb_dslash",
    "http_in_path",
    "https_token",
    "punycode",
    "port",
    "tld_in_path",
    "tld_in_subdomain",
    "abnormal_subdomain",
    "nb_subdomains",
    "prefix_suffix",
    "random_domain",
    "shortening_service",
    "path_extension",
    "nb_redirection",
    "nb_external_redirection",
    "phish_hints",
    "domain_in_brand",
    "brand_in_subdomain",
    "brand_in_path",
    "suspecious_tld",
    "statistical_report",
    "ratio_nullHyperlinks",
    "ratio_intRedirection",
    "ratio_intErrors",
    "login_form",
    "external_favicon",
    "sfh",
    "iframe",
    "popup_window",
    "onmouseover",
    "right_clic",
    "empty_title",
    "domain_in_title",
    "domain_with_copyright",
    "whois_registered_domain",
    "dns_record",
    "google_index",
    TARGET,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input CSV or directory containing dataset_phishing.csv. Defaults "
            "to dataset/url/raw."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Exact source CSV overriding positional directory discovery.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Processed CSV filename or directory. Defaults to dataset/url/data.csv.",
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
            "Model metadata filename or directory. Defaults to info.json beside "
            "the processed CSV."
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
    candidate = source / INPUT_FILENAME
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


def ensure_writable(paths: Sequence[Path], force: bool) -> None:
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
    actual = tuple(data.columns)
    if actual != RAW_COLUMNS:
        raise ValueError(
            "Unexpected raw columns. "
            f"Expected {list(RAW_COLUMNS)}, got {list(actual)}."
        )
    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Raw URL data has {len(data):,} rows; expected {EXPECTED_ROWS:,}."
        )
    if data.isna().any().any():
        raise ValueError("The official source unexpectedly contains missing values.")
    if data["url"].astype(str).str.len().eq(0).any():
        raise ValueError("Raw URL identifiers must not be empty.")
    if not data["submit_email"].eq(0).all():
        raise ValueError("submit_email is no longer constant zero in the source.")
    counts = data[TARGET].value_counts().to_dict()
    if counts != {"legitimate": 5_715, "phishing": 5_715}:
        raise ValueError(f"Unexpected status distribution: {counts}")
    return data


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    data = validate_source(raw)
    processed = data.drop(columns=list(DROPPED_COLUMNS)).copy()
    if tuple(processed.columns) != OUTPUT_COLUMNS:
        raise AssertionError("URL preprocessing produced an unexpected column order.")

    for column in OUTPUT_COLUMNS:
        if column != TARGET:
            processed[column] = pd.to_numeric(processed[column], errors="raise")
    processed[TARGET] = processed[TARGET].astype(str)

    if processed.isna().any().any():
        raise ValueError("Processed URL data unexpectedly contains missing values.")
    return processed.reset_index(drop=True)


def preprocessing_metadata(
    source: Path,
    destination: Path,
    raw: pd.DataFrame,
    processed: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "download": {
            "provider": "Kaggle",
            "dataset": "Web Page Phishing Detection Dataset",
            "dataset_handle": DATASET_HANDLE,
            "dataset_url": DATASET_URL,
            "upstream_repository": MENDELEY_URL,
            "doi": DATASET_DOI,
            "version": 2,
            "license": "CC BY 4.0",
            "file": INPUT_FILENAME,
            "command": "uv run python dataset/url/raw/download_url.py",
        },
        "source_file": source.name,
        "source_sha256": sha256(source),
        "output_file": destination.name,
        "preprocessing_command": (
            "uv run python dataset/url/raw/preprocess_url.py --force"
        ),
        "preprocessing_steps": [
            {
                "order": 1,
                "operation": "validate_source",
                "details": (
                    "Require the official 89-column schema, 11,430 rows, no "
                    "missing values, a constant-zero submit_email field, and "
                    "5,715 rows in each target class."
                ),
            },
            {
                "order": 2,
                "operation": "drop_url_identifier",
                "details": (
                    "Drop the near-unique raw URL string so generators do not "
                    "memorize or reproduce website identifiers."
                ),
            },
            {
                "order": 3,
                "operation": "drop_submit_email",
                "details": (
                    "Drop submit_email because it is constant zero in all source "
                    "rows and its name is interpreted as an email semantic type."
                ),
            },
            {
                "order": 4,
                "operation": "validate_and_preserve_order",
                "details": (
                    "Coerce the 86 retained predictors to numeric values, retain "
                    "the string status target, and preserve source feature order."
                ),
            },
        ],
        "columns": {
            "source": list(RAW_COLUMNS),
            "output": list(OUTPUT_COLUMNS),
            "categorical": [
                column for column in OUTPUT_COLUMNS if column in CATEGORICAL_COLUMNS
            ],
            "numerical": [
                column for column in OUTPUT_COLUMNS if column not in CATEGORICAL_COLUMNS
            ],
            "target": TARGET,
            "removed": list(DROPPED_COLUMNS),
        },
        "row_counts": {
            "input_rows": int(len(raw)),
            "output_rows": int(len(processed)),
            "legitimate": int(processed[TARGET].eq("legitimate").sum()),
            "phishing": int(processed[TARGET].eq("phishing").sum()),
        },
    }


def model_info(processed: pd.DataFrame) -> dict[str, Any]:
    metadata_columns = {
        column: {
            "sdtype": (
                "categorical" if column in CATEGORICAL_COLUMNS else "numerical"
            )
        }
        for column in processed.columns
    }
    return {
        "name": "url",
        "source": MENDELEY_URL,
        "doi": DATASET_DOI,
        "license": "CC BY 4.0",
        "task": "binary_classification",
        "target": TARGET,
        "features_by_order": list(processed.columns),
        "shape": list(processed.shape),
        "col_types": {
            column: {
                "type": "cat" if column in CATEGORICAL_COLUMNS else "num",
                "unique_values": int(processed[column].nunique(dropna=True)),
                "missing_values": int(processed[column].isna().sum()),
            }
            for column in processed.columns
        },
        "sdv_metadata": {
            "tables": {"table": {"columns": metadata_columns}},
            "relationships": [],
            "METADATA_SPEC_VERSION": "V1",
        },
        "preprocessing_note": (
            "Dropped raw url because it is a near-unique website identifier. "
            "Dropped submit_email because it is constant zero and its name is "
            "interpreted as an email semantic type. See "
            "raw/preprocessing_info.json for the full audit."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
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
    raw = pd.read_csv(source, encoding="utf-8-sig")
    processed = preprocess(raw)

    staging = destination.with_name(f".{destination.name}.partial")
    processed.to_csv(staging, index=False)
    staging.replace(destination)
    write_json(
        metadata_path,
        preprocessing_metadata(source, destination, raw, processed),
    )
    write_json(info_path, model_info(processed))

    print(f"Read {len(raw):,} rows and {len(raw.columns)} columns from {source}")
    print(f"Dropped columns: {', '.join(DROPPED_COLUMNS)}")
    print(f"Wrote {len(processed):,} rows and {len(processed.columns)} columns")
    print(f"Data: {destination}")
    print(f"Preprocessing audit: {metadata_path}")
    print(f"Model metadata: {info_path}")


if __name__ == "__main__":
    main()
