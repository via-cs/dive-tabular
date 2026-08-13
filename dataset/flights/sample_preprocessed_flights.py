"""Deterministically sample the processed Flights table used by experiments.

Examples:
    uv run python dataset/flights/sample_preprocessed_flights.py
    uv run python dataset/flights/sample_preprocessed_flights.py dataset/flights/raw
    uv run python dataset/flights/sample_preprocessed_flights.py /tmp/flights.csv --output /tmp/sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("dataset/flights/raw/flights_preprocessed.csv")
DEFAULT_OUTPUT = Path("dataset/flights/data.csv")
INPUT_FILENAME = "flights_preprocessed.csv"
OUTPUT_FILENAME = "data.csv"

OUTPUT_COLUMNS = tuple(
    """
    YEAR MONTH DAY AIRLINE AIRLINE_CODE ORIGIN ORIGIN_CITY DEST DEST_CITY
    CRS_DEP_TIME_UTC_MIN DEP_TIME_UTC_MIN DEP_DELAY TAXI_OUT
    WHEELS_OFF_UTC_MIN WHEELS_ON_UTC_MIN TAXI_IN CRS_ARR_TIME_UTC_MIN
    ARR_TIME_UTC_MIN CRS_ELAPSED_TIME ELAPSED_TIME AIR_TIME DISTANCE ARR_DELAY
    """.split()
)
STRING_COLUMNS = {
    "AIRLINE", "AIRLINE_CODE", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY"
}
INTEGER_COLUMNS = tuple(
    column for column in OUTPUT_COLUMNS if column not in STRING_COLUMNS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV or directory containing flights_preprocessed.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV filename or directory. Defaults to dataset/flights/data.csv.",
    )
    parser.add_argument("--rows", type=int, default=60_000, help="Rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Input rows processed at a time.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output CSV.",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    path = path.expanduser()
    return path / INPUT_FILENAME if path.is_dir() else path


def resolve_output(path: Path) -> Path:
    path = path.expanduser()
    return path if path.suffix.lower() == ".csv" else path / OUTPUT_FILENAME


def sample_rows(source: Path, rows: int, seed: int, chunksize: int) -> tuple[pd.DataFrame, int]:
    """Use random priorities to sample uniformly without loading all rows."""
    generator = np.random.default_rng(seed)
    sample: pd.DataFrame | None = None
    total_rows = 0

    for chunk in pd.read_csv(source, chunksize=chunksize):
        total_rows += len(chunk)
        chunk["_sample_priority"] = generator.random(len(chunk))
        if sample is None:
            sample = chunk
        else:
            sample = pd.concat([sample, chunk], ignore_index=True)
        if len(sample) > rows:
            sample = sample.nsmallest(rows, "_sample_priority")

    if sample is None or total_rows < rows:
        raise ValueError(f"Requested {rows:,} rows, but input contains {total_rows:,}")

    sample = sample.drop(columns="_sample_priority")
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sample, total_rows


def normalize_output(sample: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen experiment schema and stable integer serialization."""
    actual = set(sample.columns)
    expected = set(OUTPUT_COLUMNS)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Unexpected processed Flights schema; missing={missing}, "
            f"unexpected={unexpected}"
        )

    output = sample.loc[:, OUTPUT_COLUMNS].copy()
    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(output[column], errors="raise")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Expected integer-valued Flights column: {column}")
        output[column] = values.astype("int64")
    return output


def main() -> None:
    args = parse_args()
    source = resolve_input(args.input)
    destination = resolve_output(args.output)

    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    if args.rows < 1:
        raise ValueError("--rows must be positive")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {destination}; use --force")

    sample, total_rows = sample_rows(source, args.rows, args.seed, args.chunksize)
    output = normalize_output(sample)
    staging = destination.with_name(f".{destination.name}.partial")
    output.to_csv(staging, index=False)
    staging.replace(destination)
    print(f"Sampled {len(output):,} of {total_rows:,} rows to {destination}")


if __name__ == "__main__":
    main()
