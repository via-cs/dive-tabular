"""Preprocess the Kaggle flight sample for unconstrained CTGAN/TVAE training.

Examples:
    uv run python dataset/flights/preprocess_flights.py
    uv run python dataset/flights/preprocess_flights.py dataset/flights/raw
    uv run python dataset/flights/preprocess_flights.py /tmp/flights.csv --output /tmp/processed

The input may be either a CSV file or a directory containing
``flights_sample_3m.csv``. The output may be a directory or an explicit CSV filename. This stage writes
the complete validated intermediate; run sample_preprocessed_flights.py afterward
to create the frozen 60,000-row experiment table.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd


DEFAULT_INPUT = Path("dataset/flights/raw/flights_sample_3m.csv")
DEFAULT_AIRPORTS = Path("dataset/flights/raw/airports_labeled.dat")
DEFAULT_OUTPUT = Path("dataset/flights/raw")
INPUT_FILENAME = "flights_sample_3m.csv"
OUTPUT_FILENAME = "flights_preprocessed.csv"

DEPARTURE_CLOCK_COLUMNS = ("CRS_DEP_TIME", "DEP_TIME", "WHEELS_OFF")
ARRIVAL_CLOCK_COLUMNS = ("WHEELS_ON", "CRS_ARR_TIME", "ARR_TIME")
CLOCK_COLUMNS = DEPARTURE_CLOCK_COLUMNS + ARRIVAL_CLOCK_COLUMNS
LOCATION_COLUMNS = ("ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY")
TOP_LOCATION_VALUES = 100

CATEGORICAL_COLUMNS = (
    "YEAR",
    "MONTH",
    "DAY",
    "AIRLINE",
    "AIRLINE_CODE",
    "ORIGIN",
    "ORIGIN_CITY",
    "DEST",
    "DEST_CITY",
)
NUMERICAL_COLUMNS = (
    "CRS_DEP_TIME_UTC_MIN",
    "DEP_TIME_UTC_MIN",
    "DEP_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF_UTC_MIN",
    "WHEELS_ON_UTC_MIN",
    "TAXI_IN",
    "CRS_ARR_TIME_UTC_MIN",
    "ARR_TIME_UTC_MIN",
    "ARR_DELAY",
    "CRS_ELAPSED_TIME",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
)

SOURCE_COLUMNS = (
    "FL_DATE",
    "AIRLINE",
    "AIRLINE_CODE",
    "ORIGIN",
    "ORIGIN_CITY",
    "DEST",
    "DEST_CITY",
    *CLOCK_COLUMNS,
    "DEP_DELAY",
    "TAXI_OUT",
    "TAXI_IN",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
)
OUTPUT_COLUMNS = (*CATEGORICAL_COLUMNS, *NUMERICAL_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV or directory containing flights_sample_3m.csv.",
    )
    parser.add_argument(
        "--airports",
        type=Path,
        default=DEFAULT_AIRPORTS,
        help="OpenFlights airport metadata CSV containing iata and tz_db_timezone.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory or explicit preprocessed CSV filename.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Number of input rows processed at a time.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output CSV.",
    )
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        return path / INPUT_FILENAME
    if path.is_file():
        return path

    # The dataset may be kept alongside the processed outputs under raw.
    fallback = Path("dataset/flights/raw") / INPUT_FILENAME
    if path == DEFAULT_INPUT and fallback.is_file():
        return fallback
    return path


def resolve_output(path: Path) -> Path:
    path = path.expanduser()
    return path if path.suffix.lower() == ".csv" else path / OUTPUT_FILENAME


def load_timezones(path: Path) -> dict[str, str]:
    airports = pd.read_csv(path, usecols=["iata", "tz_db_timezone"])
    airports = airports.dropna(subset=["iata", "tz_db_timezone"])
    timezones: dict[str, str] = {}
    for iata, timezone in airports.itertuples(index=False):
        if timezone == "\\N":
            continue
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            continue
        timezones.setdefault(iata, timezone)
    return timezones


def find_top_location_values(source: Path, chunksize: int) -> dict[str, set[str]]:
    """Find the 100 most common values for each retained location column."""
    counts = {column: Counter() for column in LOCATION_COLUMNS}
    usecols = ["CANCELLED", "DIVERTED", *LOCATION_COLUMNS]
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=chunksize):
        completed = chunk[(chunk["CANCELLED"] == 0) & (chunk["DIVERTED"] == 0)]
        for column in LOCATION_COLUMNS:
            counts[column].update(completed[column].dropna().astype(str))

    return {
        column: {value for value, _ in count.most_common(TOP_LOCATION_VALUES)}
        for column, count in counts.items()
    }


def valid_hhmm(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    hours = values // 100
    minutes = values % 100
    return (
        values.notna()
        & (values >= 0)
        & (hours <= 24)
        & (minutes < 60)
        & ~((hours == 24) & (minutes != 0))
    )


def utc_minutes(
    dates: pd.Series, hhmm: pd.Series, timezones: pd.Series
) -> pd.Series:
    """Convert local HHMM values to UTC minute-of-day values.

    ``FL_DATE`` is the origin-local flight date. Arrival fields use this same
    calendar date to determine the timezone offset. The output intentionally
    stores only minute-of-day, so a next-day offset is not represented.
    """
    hhmm = pd.to_numeric(hhmm, errors="raise").astype("int64")
    minutes = (hhmm // 100) * 60 + (hhmm % 100)
    local_naive = pd.to_datetime(dates, errors="raise").dt.normalize()
    local_naive = local_naive + pd.to_timedelta(minutes, unit="m")
    result = pd.Series(index=hhmm.index, dtype="int16")

    for timezone, index in timezones.groupby(timezones, sort=False).groups.items():
        local = local_naive.loc[index].dt.tz_localize(
            timezone,
            ambiguous=False,
            nonexistent="shift_forward",
        )
        utc = local.dt.tz_convert("UTC")
        result.loc[index] = (utc.dt.hour * 60 + utc.dt.minute).astype("int16")

    return result


TIME_EQUATION_DESCRIPTIONS = {
    "wheels_off_from_departure": "WHEELS_OFF = DEP_TIME + TAXI_OUT (mod 1440)",
    "arrival_from_wheels_on": "ARR_TIME = WHEELS_ON + TAXI_IN (mod 1440)",
    "wheels_on_from_wheels_off": "WHEELS_ON = WHEELS_OFF + AIR_TIME (mod 1440)",
    "scheduled_arrival_from_departure": "CRS_ARR = CRS_DEP + CRS_ELAPSED (mod 1440)",
    "departure_from_schedule": "DEP_TIME = CRS_DEP + DEP_DELAY (mod 1440)",
    "arrival_from_schedule": "ARR_TIME = CRS_ARR + ARR_DELAY (mod 1440)",
    "elapsed_from_components": "ELAPSED = TAXI_OUT + AIR_TIME + TAXI_IN",
    "arrival_delay_from_duration": "ARR_DELAY = DEP_DELAY + ELAPSED - CRS_ELAPSED",
    "arrival_from_departure_and_elapsed": "ARR_TIME = DEP_TIME + ELAPSED (mod 1440)",
}


def modular_error(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    """Return the nearest signed clock difference in minutes."""
    return ((lhs - rhs + 720) % 1440) - 720


def valid_time_equations(chunk: pd.DataFrame, counters: Counter) -> pd.Series:
    """Return rows satisfying every requested exact time equality."""
    checks = {
        "wheels_off_from_departure": modular_error(
            chunk["WHEELS_OFF_UTC_MIN"],
            chunk["DEP_TIME_UTC_MIN"] + chunk["TAXI_OUT"],
        ).eq(0),
        "arrival_from_wheels_on": modular_error(
            chunk["ARR_TIME_UTC_MIN"],
            chunk["WHEELS_ON_UTC_MIN"] + chunk["TAXI_IN"],
        ).eq(0),
        "wheels_on_from_wheels_off": modular_error(
            chunk["WHEELS_ON_UTC_MIN"],
            chunk["WHEELS_OFF_UTC_MIN"] + chunk["AIR_TIME"],
        ).eq(0),
        "scheduled_arrival_from_departure": modular_error(
            chunk["CRS_ARR_TIME_UTC_MIN"],
            chunk["CRS_DEP_TIME_UTC_MIN"] + chunk["CRS_ELAPSED_TIME"],
        ).eq(0),
        "departure_from_schedule": modular_error(
            chunk["DEP_TIME_UTC_MIN"],
            chunk["CRS_DEP_TIME_UTC_MIN"] + chunk["DEP_DELAY"],
        ).eq(0),
        "arrival_from_schedule": modular_error(
            chunk["ARR_TIME_UTC_MIN"],
            chunk["CRS_ARR_TIME_UTC_MIN"] + chunk["ARR_DELAY"],
        ).eq(0),
        "elapsed_from_components": (
            chunk["ELAPSED_TIME"]
            == chunk["TAXI_OUT"] + chunk["AIR_TIME"] + chunk["TAXI_IN"]
        ),
        "arrival_delay_from_duration": (
            chunk["ARR_DELAY"]
            == chunk["DEP_DELAY"]
            + chunk["ELAPSED_TIME"]
            - chunk["CRS_ELAPSED_TIME"]
        ),
        "arrival_from_departure_and_elapsed": modular_error(
            chunk["ARR_TIME_UTC_MIN"],
            chunk["DEP_TIME_UTC_MIN"] + chunk["ELAPSED_TIME"],
        ).eq(0),
    }

    valid = pd.Series(True, index=chunk.index)
    for name, check in checks.items():
        counters[f"equation_violations_{name}"] += int((~check).sum())
        valid &= check
    counters["removed_time_equation_violations"] += int((~valid).sum())
    return valid


def process_chunk(
    chunk: pd.DataFrame,
    airport_timezones: dict[str, str],
    top_location_values: dict[str, set[str]],
    counters: Counter,
) -> pd.DataFrame:
    counters["input_rows"] += len(chunk)

    completed = (chunk["CANCELLED"] == 0) & (chunk["DIVERTED"] == 0)
    counters["removed_cancelled_or_diverted"] += int((~completed).sum())
    chunk = chunk.loc[completed].copy()

    in_top_locations = pd.Series(True, index=chunk.index)
    for column, values in top_location_values.items():
        in_top_locations &= chunk[column].isin(values)
    counters["removed_outside_top_location_values"] += int((~in_top_locations).sum())
    chunk = chunk.loc[in_top_locations].copy()

    chunk["FL_DATE"] = pd.to_datetime(chunk["FL_DATE"], errors="coerce")
    chunk["ORIGIN_TZ"] = chunk["ORIGIN"].map(airport_timezones)
    chunk["DEST_TZ"] = chunk["DEST"].map(airport_timezones)

    valid_times = pd.Series(True, index=chunk.index)
    for column in CLOCK_COLUMNS:
        valid_times &= valid_hhmm(chunk[column])

    required = [
        "FL_DATE",
        "AIRLINE",
        "AIRLINE_CODE",
        "ORIGIN",
        "ORIGIN_CITY",
        "DEST",
        "DEST_CITY",
        "ORIGIN_TZ",
        "DEST_TZ",
        "DEP_DELAY",
        "TAXI_OUT",
        "TAXI_IN",
        "ARR_DELAY",
        "CRS_ELAPSED_TIME",
        "ELAPSED_TIME",
        "AIR_TIME",
        "DISTANCE",
    ]
    complete = chunk[required].notna().all(axis=1) & valid_times
    counters["removed_missing_invalid_or_unmapped"] += int((~complete).sum())
    chunk = chunk.loc[complete].copy()

    chunk["YEAR"] = chunk["FL_DATE"].dt.year.astype("int16")
    chunk["MONTH"] = chunk["FL_DATE"].dt.month.astype("int8")
    chunk["DAY"] = chunk["FL_DATE"].dt.day.astype("int8")

    for column in DEPARTURE_CLOCK_COLUMNS:
        chunk[f"{column}_UTC_MIN"] = utc_minutes(
            chunk["FL_DATE"], chunk[column], chunk["ORIGIN_TZ"]
        )
    for column in ARRIVAL_CLOCK_COLUMNS:
        chunk[f"{column}_UTC_MIN"] = utc_minutes(
            chunk["FL_DATE"], chunk[column], chunk["DEST_TZ"]
        )

    valid_equations = valid_time_equations(chunk, counters)
    chunk = chunk.loc[valid_equations].copy()

    output = chunk.loc[:, OUTPUT_COLUMNS].copy()
    for column in NUMERICAL_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="raise")

    counters["output_rows"] += len(output)
    return output


def main() -> None:
    args = parse_args()
    source = resolve_input(args.input)
    destination = resolve_output(args.output)
    airports = args.airports.expanduser()
    if not airports.is_file() and args.airports == DEFAULT_AIRPORTS:
        airports = Path("dataset/flights/raw/airports_labeled.dat")

    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    if not airports.is_file():
        raise FileNotFoundError(f"Airport metadata not found: {airports}")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {destination}; use --force")

    airport_timezones = load_timezones(airports)
    top_location_values = find_top_location_values(source, args.chunksize)
    counters: Counter = Counter()
    category_values = {column: set() for column in CATEGORICAL_COLUMNS}
    temporary = destination.with_name(f".{destination.name}.partial")
    if temporary.exists():
        temporary.unlink()

    wrote_header = False
    for chunk in pd.read_csv(source, usecols=SOURCE_COLUMNS, chunksize=args.chunksize):
        processed = process_chunk(
            chunk, airport_timezones, top_location_values, counters
        )
        if processed.empty:
            continue
        for column in CATEGORICAL_COLUMNS:
            category_values[column].update(processed[column].astype(str).unique())
        processed.to_csv(temporary, mode="a", header=not wrote_header, index=False)
        wrote_header = True

    if not wrote_header:
        raise RuntimeError("No rows remained after preprocessing")
    temporary.replace(destination)

    metadata = {
        "source": str(source.resolve()),
        "airport_metadata": str(airports.resolve()),
        "output": str(destination.resolve()),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numerical_columns": list(NUMERICAL_COLUMNS),
        "location_filter": {
            "columns": list(LOCATION_COLUMNS),
            "top_values_per_column": TOP_LOCATION_VALUES,
            "selection_population": "completed, non-diverted input rows",
            "allowed_values": {
                column: sorted(values) for column, values in top_location_values.items()
            },
        },
        "strict_time_equation_filter": {
            "comparison": "exact equality; clock equations use modulo 1440",
            "equations": TIME_EQUATION_DESCRIPTIONS,
        },
        "clock_conversion": {
            "output_unit": "UTC minute of day, integer in [0, 1439]",
            "origin_timezone_columns": list(DEPARTURE_CLOCK_COLUMNS),
            "destination_timezone_columns": list(ARRIVAL_CLOCK_COLUMNS),
            "date_source": "FL_DATE",
            "dst_policy": {
                "ambiguous": "standard-time occurrence",
                "nonexistent": "shift forward",
            },
        },
        "removed_columns": [
            "FL_DATE",
            "AIRLINE_DOT",
            "DOT_CODE",
            "FL_NUMBER",
            "CANCELLED",
            "CANCELLATION_CODE",
            "DIVERTED",
            "DELAY_DUE_CARRIER",
            "DELAY_DUE_WEATHER",
            "DELAY_DUE_NAS",
            "DELAY_DUE_SECURITY",
            "DELAY_DUE_LATE_AIRCRAFT",
        ],
        "row_counts": dict(counters),
        "categorical_cardinalities": {
            column: len(values) for column, values in category_values.items()
        },
    }
    metadata_path = destination.parent / "preprocessing_info.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {counters['output_rows']:,} rows to {destination}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
