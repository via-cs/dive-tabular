"""Download and preprocess 30 seasons of NBA player scoring statistics.

The default invocation downloads one per-100-possessions CSV for every NBA
regular season from 1996-97 through 2025-26, then merges, validates, shuffles,
and writes the numeric table used by the synthetic-data experiments.

Examples:
    uv run python dataset/nba/raw/download_all_seasons.py
    uv run python dataset/nba/raw/download_all_seasons.py --force-output
    uv run python dataset/nba/raw/download_all_seasons.py --preprocess-only
    uv run python dataset/nba/raw/download_all_seasons.py /tmp/nba-seasons \
        --output /tmp/nba-data.csv --info-output /tmp/preprocessing.txt
    uv run python dataset/nba/raw/download_all_seasons.py \
        --preprocess-only --season-file /tmp/96-97.csv \
        --season-file /tmp/97-98.csv --output /tmp/nba-data.csv

The positional season source may be a directory or one explicit CSV file.
Repeated ``--season-file`` arguments override directory discovery, satisfying
workflows where specifically named files must replace the default paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


API_URL = "https://api.pbpstats.com/get-totals/nba"
FIRST_SEASON_START = 1996
LAST_SEASON_START = 2025
DEFAULT_SEASON_DIRECTORY = Path("dataset/nba/raw/all_seasons")
DEFAULT_OUTPUT = Path("dataset/nba/data.csv")
DEFAULT_INFO_OUTPUT = Path("dataset/nba/raw/preprocessing.txt")
OUTPUT_FILENAME = "data.csv"
INFO_FILENAME = "preprocessing.txt"
SHUFFLE_SEED = 42
CONSTRAINT_ATOL = 1e-9
CONSTRAINT_RTOL = 1e-9

IDENTIFIER_AND_EXPOSURE_COLUMNS = (
    "Name",
    "TeamAbbreviation",
    "GamesPlayed",
    "Minutes",
    "OffPoss",
)

RATE_COLUMNS = (
    "Points",
    "FG2M",
    "FG2A",
    "FG3M",
    "FG3A",
    "FtPoints",
    "PtsAssisted2s",
    "PtsUnassisted2s",
    "PtsAssisted3s",
    "PtsUnassisted3s",
    "PtsPutbacks",
    "Fg2aBlocked",
)

UNIT_INTERVAL_COLUMNS = (
    "Fg2Pct",
    "Fg3Pct",
    "NonHeaveFg3Pct",
    "Assisted2sPct",
    "NonPutbacksAssisted2sPct",
    "Assisted3sPct",
    "FG3APct",
    "ShotQualityAvg",
    "FG2APctBlocked",
    "FG3APctBlocked",
)

RAW_OUTPUT_COLUMNS = (
    "Name",
    "TeamAbbreviation",
    "GamesPlayed",
    "Minutes",
    "OffPoss",
    "Points",
    "FG2M",
    "FG2A",
    "Fg2Pct",
    "FG3M",
    "FG3A",
    "Fg3Pct",
    "NonHeaveFg3Pct",
    "FtPoints",
    "PtsAssisted2s",
    "PtsUnassisted2s",
    "PtsAssisted3s",
    "PtsUnassisted3s",
    "Assisted2sPct",
    "NonPutbacksAssisted2sPct",
    "Assisted3sPct",
    "FG3APct",
    "ShotQualityAvg",
    "EfgPct",
    "TsPct",
    "PtsPutbacks",
    "Fg2aBlocked",
    "FG2APctBlocked",
    "Fg3aBlocked",
    "FG3APctBlocked",
    "Usage",
)

MODEL_COLUMNS = tuple(
    column
    for column in RAW_OUTPUT_COLUMNS
    if column not in IDENTIFIER_AND_EXPOSURE_COLUMNS
)

NUMERIC_RAW_COLUMNS = tuple(
    column
    for column in RAW_OUTPUT_COLUMNS
    if column not in {"Name", "TeamAbbreviation"}
)


@dataclass(frozen=True)
class FileAudit:
    path: Path
    rows_read: int
    null_cells: int
    rows_kept: int
    rows_dropped: int


def season_name(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


SEASONS = tuple(
    season_name(year)
    for year in range(FIRST_SEASON_START, LAST_SEASON_START + 1)
)


def short_season_name(season: str) -> str:
    return f"{season[2:]}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "season_source",
        nargs="?",
        type=Path,
        default=DEFAULT_SEASON_DIRECTORY,
        help=(
            "Directory for/discovery of per-season CSV files, or one explicit "
            "CSV file. Defaults to dataset/nba/raw/all_seasons."
        ),
    )
    parser.add_argument(
        "--season-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit seasonal CSV to preprocess. Repeat for multiple files. "
            "When supplied, these files replace season-directory discovery."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Processed CSV path or directory. Defaults to dataset/nba/data.csv."
        ),
    )
    parser.add_argument(
        "--info-output",
        type=Path,
        default=DEFAULT_INFO_OUTPUT,
        help=(
            "Preprocessing report path or directory. Defaults to "
            "dataset/nba/raw/preprocessing.txt."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Download/validate all seasonal CSV files without preprocessing.",
    )
    mode.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Use existing seasonal CSV files without making API requests.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download and atomically replace existing seasonal CSV files.",
    )
    parser.add_argument(
        "--force-output",
        action="store_true",
        help="Replace existing data.csv and preprocessing.txt outputs.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=SHUFFLE_SEED,
        help=f"Deterministic merged-row shuffle seed (default: {SHUFFLE_SEED}).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Maximum API attempts per season (default: 8).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for each API attempt (default: 180).",
    )
    return parser.parse_args()


def resolve_named_output(path: Path, filename: str, suffix: str) -> Path:
    path = path.expanduser()
    return path if path.suffix.lower() == suffix else path / filename


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.partial")
    frame.to_csv(staging, index=False)
    staging.replace(destination)


def atomic_write_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.partial")
    staging.write_text(text, encoding="utf-8")
    staging.replace(destination)


def validate_season_csv(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Seasonal CSV is missing or empty: {path}")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != RAW_OUTPUT_COLUMNS:
        raise ValueError(
            f"Unexpected schema in {path}. Expected {list(RAW_OUTPUT_COLUMNS)}, "
            f"got {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError(f"Seasonal CSV contains no player rows: {path}")
    return len(frame)


def request_json(
    season: str,
    *,
    max_attempts: int,
    timeout: float,
) -> dict[str, Any]:
    query = urlencode(
        {
            "Season": season,
            "SeasonType": "Regular Season",
            "Type": "Player",
        }
    )
    url = f"{API_URL}?{query}"
    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        try:
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "LLM-Tabular-NBA-dataset-builder/1.0",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            retryable = error.code in retryable_statuses
            last_error: Exception = error
            retry_after = error.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = float(retry_after)
            else:
                delay = min(2 ** (attempt - 1), 30)
        except (URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as error:
            retryable = True
            last_error = error
            delay = min(2 ** (attempt - 1), 30)

        if not retryable or attempt == max_attempts:
            raise RuntimeError(
                f"Failed to download {season} after {attempt} attempt(s): "
                f"{last_error}"
            ) from last_error

        print(
            f"  {season}: attempt {attempt} failed ({last_error}); "
            f"retrying in {delay:g}s",
            flush=True,
        )
        time.sleep(delay)

    raise AssertionError("unreachable")


def transform_api_rows(payload: dict[str, Any], season: str) -> pd.DataFrame:
    rows = payload.get("multi_row_table_data")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{season}: API returned no player rows")

    raw = pd.DataFrame(rows)
    required = set(RAW_OUTPUT_COLUMNS).difference({"Fg3aBlocked"})
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{season}: API response is missing fields: {missing}")

    off_possessions = pd.to_numeric(raw["OffPoss"], errors="coerce")
    # Extremely brief appearances can have zero recorded offensive possessions.
    # Preserve them with undefined rates; preprocessing rejects those rows.
    normalization_denominator = off_possessions.where(off_possessions > 0)

    output = raw[
        [column for column in RAW_OUTPUT_COLUMNS if column != "Fg3aBlocked"]
    ].copy()

    for column in RATE_COLUMNS:
        counts = pd.to_numeric(raw[column], errors="coerce")
        output[column] = 100.0 * counts / normalization_denominator

    blocked_three_pct = pd.to_numeric(
        raw["FG3APctBlocked"], errors="coerce"
    )
    three_attempts = pd.to_numeric(raw["FG3A"], errors="coerce")
    output["Fg3aBlocked"] = (
        100.0
        * blocked_three_pct
        * three_attempts
        / normalization_denominator
    )

    return output.loc[:, RAW_OUTPUT_COLUMNS]


def download_all_seasons(
    directory: Path,
    *,
    force: bool,
    max_attempts: int,
    timeout: float,
) -> list[Path]:
    directory = directory.expanduser()
    if directory.suffix.lower() == ".csv":
        raise ValueError(
            "Downloading all seasons requires a destination directory, not "
            f"an individual CSV path: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for season in SEASONS:
        destination = directory / short_season_name(season)
        paths.append(destination)

        if destination.exists() and not force:
            row_count = validate_season_csv(destination)
            print(
                f"Cached {season}: {row_count:,} rows at {destination}",
                flush=True,
            )
            continue

        print(f"Downloading {season}...", flush=True)
        payload = request_json(
            season,
            max_attempts=max_attempts,
            timeout=timeout,
        )
        frame = transform_api_rows(payload, season)
        atomic_write_csv(frame, destination)
        validate_season_csv(destination)
        print(
            f"Downloaded {season}: {len(frame):,} rows to {destination}",
            flush=True,
        )
        time.sleep(1)

    return paths


def close_enough(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.Series(
        np.isclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            atol=CONSTRAINT_ATOL,
            rtol=CONSTRAINT_RTOL,
            equal_nan=False,
        ),
        index=left.index,
    )


def constraint_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    tolerance = CONSTRAINT_ATOL
    numeric = frame.loc[:, NUMERIC_RAW_COLUMNS]

    masks = {
        "all retained numeric values are finite": pd.Series(
            np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1),
            index=frame.index,
        ),
        "all retained numeric values are nonnegative": (
            numeric >= -tolerance
        ).all(axis=1),
        "GamesPlayed >= 1": frame["GamesPlayed"] >= 1 - tolerance,
        "OffPoss > 0": frame["OffPoss"] > 0,
        "Points = 2*FG2M + 3*FG3M + FtPoints": close_enough(
            frame["Points"],
            2 * frame["FG2M"] + 3 * frame["FG3M"] + frame["FtPoints"],
        ),
        "2*FG2M = PtsAssisted2s + PtsUnassisted2s": close_enough(
            2 * frame["FG2M"],
            frame["PtsAssisted2s"] + frame["PtsUnassisted2s"],
        ),
        "3*FG3M = PtsAssisted3s + PtsUnassisted3s": close_enough(
            3 * frame["FG3M"],
            frame["PtsAssisted3s"] + frame["PtsUnassisted3s"],
        ),
        "Fg2Pct*FG2A = FG2M": close_enough(
            frame["Fg2Pct"] * frame["FG2A"], frame["FG2M"]
        ),
        "Fg3Pct*FG3A = FG3M": close_enough(
            frame["Fg3Pct"] * frame["FG3A"], frame["FG3M"]
        ),
        "FG3APct*(FG2A+FG3A) = FG3A": close_enough(
            frame["FG3APct"] * (frame["FG2A"] + frame["FG3A"]),
            frame["FG3A"],
        ),
        "EfgPct*(FG2A+FG3A) = FG2M + 1.5*FG3M": close_enough(
            frame["EfgPct"] * (frame["FG2A"] + frame["FG3A"]),
            frame["FG2M"] + 1.5 * frame["FG3M"],
        ),
        "Assisted2sPct*(2*FG2M) = PtsAssisted2s": close_enough(
            frame["Assisted2sPct"] * (2 * frame["FG2M"]),
            frame["PtsAssisted2s"],
        ),
        "Assisted3sPct*(3*FG3M) = PtsAssisted3s": close_enough(
            frame["Assisted3sPct"] * (3 * frame["FG3M"]),
            frame["PtsAssisted3s"],
        ),
        (
            "NonPutbacksAssisted2sPct*(2*FG2M-PtsPutbacks) "
            "= PtsAssisted2s"
        ): close_enough(
            frame["NonPutbacksAssisted2sPct"]
            * (2 * frame["FG2M"] - frame["PtsPutbacks"]),
            frame["PtsAssisted2s"],
        ),
        "FG2APctBlocked*FG2A = Fg2aBlocked": close_enough(
            frame["FG2APctBlocked"] * frame["FG2A"],
            frame["Fg2aBlocked"],
        ),
        "FG3APctBlocked*FG3A = Fg3aBlocked": close_enough(
            frame["FG3APctBlocked"] * frame["FG3A"],
            frame["Fg3aBlocked"],
        ),
        "probability-like columns are within [0, 1]": (
            (frame.loc[:, UNIT_INTERVAL_COLUMNS] >= -tolerance)
            & (frame.loc[:, UNIT_INTERVAL_COLUMNS] <= 1 + tolerance)
        ).all(axis=1),
        "EfgPct and TsPct are within [0, 1.5]": (
            (frame[["EfgPct", "TsPct"]] >= -tolerance)
            & (frame[["EfgPct", "TsPct"]] <= 1.5 + tolerance)
        ).all(axis=1),
        "0 <= Usage <= 100": frame["Usage"].between(
            -tolerance, 100 + tolerance
        ),
        "FG2M <= FG2A": frame["FG2M"] <= frame["FG2A"] + tolerance,
        "FG3M <= FG3A": frame["FG3M"] <= frame["FG3A"] + tolerance,
        "Fg2aBlocked <= FG2A": (
            frame["Fg2aBlocked"] <= frame["FG2A"] + tolerance
        ),
        "Fg3aBlocked <= FG3A": (
            frame["Fg3aBlocked"] <= frame["FG3A"] + tolerance
        ),
        "PtsPutbacks <= PtsUnassisted2s": (
            frame["PtsPutbacks"] <= frame["PtsUnassisted2s"] + tolerance
        ),
        "Assisted2sPct <= NonPutbacksAssisted2sPct": (
            frame["Assisted2sPct"]
            <= frame["NonPutbacksAssisted2sPct"] + tolerance
        ),
        "scoring components do not exceed Points": (
            (frame["FtPoints"] <= frame["Points"] + tolerance)
            & (2 * frame["FG2M"] <= frame["Points"] + tolerance)
            & (3 * frame["FG3M"] <= frame["Points"] + tolerance)
        ),
    }
    return masks


def resolve_preprocessing_paths(args: argparse.Namespace) -> list[Path]:
    if args.season_file:
        return [path.expanduser() for path in args.season_file]

    source = args.season_source.expanduser()
    if source.suffix.lower() == ".csv":
        return [source]

    return [source / short_season_name(season) for season in SEASONS]


def preprocess_seasons(
    paths: list[Path],
    *,
    shuffle_seed: int,
) -> tuple[
    pd.DataFrame,
    list[FileAudit],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    frames: list[pd.DataFrame] = []
    file_audits: list[FileAudit] = []
    all_violation_counts: Counter[str] = Counter()
    first_violation_counts: Counter[str] = Counter()
    null_counts: Counter[str] = Counter()

    if not paths:
        raise ValueError("No seasonal CSV paths were supplied")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing seasonal CSV: {path}")

        raw = pd.read_csv(path)
        if tuple(raw.columns) != RAW_OUTPUT_COLUMNS:
            raise ValueError(
                f"Unexpected schema in {path}. Expected "
                f"{list(RAW_OUTPUT_COLUMNS)}, got {list(raw.columns)}"
            )

        raw_rows = len(raw)
        per_column_nulls = raw.loc[:, NUMERIC_RAW_COLUMNS].isna().sum()
        for column, count in per_column_nulls.items():
            null_counts[column] += int(count)

        frame = raw.copy()
        frame.loc[:, NUMERIC_RAW_COLUMNS] = (
            frame.loc[:, NUMERIC_RAW_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        masks = constraint_masks(frame)
        valid = pd.Series(True, index=frame.index)
        first_unassigned = pd.Series(True, index=frame.index)
        for description, mask in masks.items():
            mask = mask.fillna(False)
            failures = ~mask
            all_violation_counts[description] += int(failures.sum())
            first_failures = failures & first_unassigned
            first_violation_counts[description] += int(first_failures.sum())
            first_unassigned &= mask
            valid &= mask

        kept = frame.loc[valid, MODEL_COLUMNS].copy()
        frames.append(kept)
        file_audits.append(
            FileAudit(
                path=path,
                rows_read=raw_rows,
                null_cells=int(per_column_nulls.sum()),
                rows_kept=len(kept),
                rows_dropped=int((~valid).sum()),
            )
        )
        print(
            f"Preprocessed {path.name}: kept {len(kept):,}/{raw_rows:,} rows",
            flush=True,
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    merged = merged.loc[:, MODEL_COLUMNS]

    if merged.isna().any().any():
        raise AssertionError("Processed output unexpectedly contains null values")
    if not np.isfinite(merged.to_numpy(dtype=float)).all():
        raise AssertionError("Processed output unexpectedly contains nonfinite values")

    return (
        merged,
        file_audits,
        all_violation_counts,
        first_violation_counts,
        null_counts,
    )


def build_report(
    *,
    output_path: Path,
    output: pd.DataFrame,
    file_audits: list[FileAudit],
    all_violation_counts: Counter[str],
    first_violation_counts: Counter[str],
    null_counts: Counter[str],
    shuffle_seed: int,
) -> str:
    total_read = sum(audit.rows_read for audit in file_audits)
    total_kept = sum(audit.rows_kept for audit in file_audits)
    total_dropped = sum(audit.rows_dropped for audit in file_audits)
    timestamp = datetime.now(timezone.utc).isoformat()

    lines = [
        "NBA player scoring per 100 possessions",
        "======================================",
        "",
        "Source",
        "------",
        "",
        f"API endpoint: {API_URL}",
        "Parameters: Season=<YYYY-YY>, SeasonType=Regular Season, Type=Player",
        "Default seasons: 1996-97 through 2025-26 (30 seasons)",
        f"Processed at: {timestamp}",
        "",
        "Raw acquisition and normalization",
        "---------------------------------",
        "",
        "Each season is requested separately because a multi-season player request",
        "aggregates seasons. The API response rows come from multi_row_table_data.",
        "The raw API volume statistics are counts. The seasonal CSVs convert the",
        "following fields to rates per 100 offensive possessions:",
        "",
        "    per_100_value = 100 * raw_count / OffPoss",
        "",
        f"    {', '.join(RATE_COLUMNS)}",
        "",
        "The API does not expose Fg3aBlocked directly, so it is reconstructed as:",
        "",
        "    Fg3aBlocked = 100 * FG3APctBlocked * raw_FG3A / OffPoss",
        "",
        "Preprocessing",
        "-------------",
        "",
        "1. Read every selected seasonal CSV and validate its exact 31-column schema.",
        "2. Coerce retained numeric fields to numbers and replace every numeric null",
        "   with 0. Older API responses use null for many sparse zero-count fields.",
        "3. Evaluate every equality and inequality listed below using absolute and",
        f"   relative tolerances of {CONSTRAINT_ATOL:g}.",
        "4. Drop any row violating at least one constraint.",
        "5. Remove Name, TeamAbbreviation, GamesPlayed, Minutes, and OffPoss.",
        "6. Concatenate all valid rows without a Season column, shuffle the complete",
        f"   table with pandas random_state={shuffle_seed}, and reset the row index.",
        "7. Keep Usage as the final column for downstream regression.",
        "",
        "Enforced constraints",
        "--------------------",
        "",
    ]

    for description in all_violation_counts:
        lines.append(f"* {description}")

    lines.extend(
        [
            "",
            "Non-enforced relationships",
            "--------------------------",
            "",
            "NonHeaveFg3Pct == Fg3Pct is not enforced. It happened to hold in the",
            "2025-26 file, but legitimate heave attempts can make the values differ.",
            "EfgPct <= TsPct and ShotQualityAvg <= TsPct are not enforced because",
            "they had valid counterexamples in the inspected data.",
            "",
            "Audit summary",
            "-------------",
            "",
            f"Seasonal files: {len(file_audits)}",
            f"Rows read: {total_read}",
            f"Rows kept: {total_kept}",
            f"Rows dropped: {total_dropped}",
            f"Output columns: {len(output.columns)}",
            f"Output path: {output_path}",
            f"Output SHA-256: {sha256(output_path)}",
            "",
            "Output column order:",
            "",
            f"    {', '.join(output.columns)}",
            "",
            "Per-file audit",
            "--------------",
            "",
        ]
    )

    for audit in file_audits:
        lines.append(
            f"* {audit.path}: read={audit.rows_read}, null_cells={audit.null_cells}, "
            f"kept={audit.rows_kept}, dropped={audit.rows_dropped}"
        )

    lines.extend(["", "Nulls replaced with zero by column", "----------------------------------", ""])
    nonzero_nulls = {column: count for column, count in null_counts.items() if count}
    if nonzero_nulls:
        for column, count in nonzero_nulls.items():
            lines.append(f"* {column}: {count}")
    else:
        lines.append("* None")

    lines.extend(["", "Constraint violation counts", "---------------------------", ""])
    lines.append(
        "Counts below overlap when one row violates multiple constraints. The first-"
        "failure count assigns each rejected row only to its earliest failed check."
    )
    lines.append("")
    for description, count in all_violation_counts.items():
        lines.append(
            f"* {description}: all_failures={count}, "
            f"first_failures={first_violation_counts[description]}"
        )

    lines.append("")
    return "\n".join(lines)


def ensure_new_outputs(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {rendered}; use --force-output")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_path = resolve_named_output(args.output, OUTPUT_FILENAME, ".csv")
    info_path = resolve_named_output(args.info_output, INFO_FILENAME, ".txt")

    explicit_files = bool(args.season_file) or (
        args.season_source.suffix.lower() == ".csv"
    )
    if args.download_only and explicit_files:
        raise ValueError(
            "--download-only requires a season destination directory; explicit "
            "season files are preprocessing inputs"
        )

    if not args.preprocess_only and not explicit_files:
        season_paths = download_all_seasons(
            args.season_source,
            force=args.force_download,
            max_attempts=args.max_attempts,
            timeout=args.request_timeout,
        )
    else:
        season_paths = resolve_preprocessing_paths(args)

    if args.download_only:
        print(f"Validated {len(season_paths)} seasonal CSV files")
        return

    ensure_new_outputs([output_path, info_path], args.force_output)
    (
        output,
        file_audits,
        all_violation_counts,
        first_violation_counts,
        null_counts,
    ) = preprocess_seasons(season_paths, shuffle_seed=args.shuffle_seed)

    atomic_write_csv(output, output_path)
    report = build_report(
        output_path=output_path,
        output=output,
        file_audits=file_audits,
        all_violation_counts=all_violation_counts,
        first_violation_counts=first_violation_counts,
        null_counts=null_counts,
        shuffle_seed=args.shuffle_seed,
    )
    atomic_write_text(report, info_path)

    print(
        f"Wrote {len(output):,} shuffled rows and {len(output.columns)} columns "
        f"to {output_path}"
    )
    print(f"Wrote preprocessing audit to {info_path}")


if __name__ == "__main__":
    main()
