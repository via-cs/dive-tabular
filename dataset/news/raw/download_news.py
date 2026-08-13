"""Download the UCI Online News Popularity dataset.

Examples:
    uv run python dataset/news/raw/download_news.py
    uv run python dataset/news/raw/download_news.py dataset/news/raw
    uv run python dataset/news/raw/download_news.py /tmp/news.csv --force

The output may be either a directory or an explicit CSV filename. The
accompanying ``.names`` file is installed beside the CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/332/"
    "online%2Bnews%2Bpopularity.zip"
)
ARCHIVE_CSV = "OnlineNewsPopularity/OnlineNewsPopularity.csv"
ARCHIVE_NAMES = "OnlineNewsPopularity/OnlineNewsPopularity.names"
OUTPUT_FILENAME = "OnlineNewsPopularity.csv"
EXPECTED_ROWS = 39_644
DEFAULT_OUTPUT = Path(__file__).resolve().parent

EXPECTED_COLUMNS = (
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
    "shares",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output directory or explicit CSV filename. Defaults to "
            "dataset/news/raw/OnlineNewsPopularity.csv."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the CSV and names destinations if they already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Download timeout in seconds (default: 180).",
    )
    return parser.parse_args(argv)


def destination_from_output(output: Path) -> Path:
    """Interpret a .csv path as a file and every other path as a directory."""
    output = output.expanduser()
    return output if output.suffix.lower() == ".csv" else output / OUTPUT_FILENAME


def names_destination(csv_destination: Path) -> Path:
    return csv_destination.with_suffix(".names")


def validate_csv(path: Path) -> None:
    """Require the complete official UCI CSV schema and row count."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded CSV is empty or missing: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = tuple(column.strip() for column in next(reader, []))
        row_count = sum(1 for row in reader if row)

    if header != EXPECTED_COLUMNS:
        raise RuntimeError(
            "Downloaded CSV has an unexpected schema. "
            f"Expected {list(EXPECTED_COLUMNS)}, got {list(header)}."
        )
    if row_count != EXPECTED_ROWS:
        raise RuntimeError(
            f"Downloaded CSV has {row_count:,} rows; expected {EXPECTED_ROWS:,}."
        )


def _extract_member(
    archive: zipfile.ZipFile,
    member: str,
    destination: Path,
) -> None:
    staging = destination.with_name(f".{destination.name}.partial")
    with archive.open(member) as source, staging.open("wb") as output:
        shutil.copyfileobj(source, output)
    os.replace(staging, destination)


def download(
    destination: Path,
    *,
    force: bool = False,
    timeout: float = 180.0,
) -> Path:
    """Download, validate, and atomically install the official UCI files."""
    destination = destination.resolve()
    names_path = names_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        validate_csv(destination)
        print(f"Already downloaded: {destination}")
        return destination

    with tempfile.TemporaryDirectory(
        prefix=".online-news-", dir=destination.parent
    ) as temporary_directory:
        archive_path = Path(temporary_directory) / "online_news_popularity.zip"
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "LLM-Tabular Online News downloader"},
        )
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            archive_path.open("wb") as output,
        ):
            while block := response.read(1024 * 1024):
                output.write(block)

        with zipfile.ZipFile(archive_path) as archive:
            members = set(archive.namelist())
            required = {ARCHIVE_CSV, ARCHIVE_NAMES}
            missing = sorted(required - members)
            if missing:
                raise RuntimeError(
                    f"UCI archive is missing expected members: {missing}"
                )
            _extract_member(archive, ARCHIVE_CSV, destination)
            _extract_member(archive, ARCHIVE_NAMES, names_path)

    validate_csv(destination)
    size_mib = destination.stat().st_size / (1024 * 1024)
    print(f"Downloaded {destination} ({size_mib:.1f} MiB)")
    print(f"Downloaded {names_path}")
    return destination


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    download(
        destination_from_output(args.output),
        force=args.force,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
