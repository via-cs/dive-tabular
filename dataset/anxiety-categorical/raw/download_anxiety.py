"""Download the Social Anxiety Dataset CSV from Kaggle.

Examples:
    uv run python dataset/anxiety-categorical/raw/download_anxiety.py
    uv run python dataset/anxiety-categorical/raw/download_anxiety.py /tmp/anxiety
    uv run python dataset/anxiety-categorical/raw/download_anxiety.py \
        /tmp/anxiety.csv --force

The output may be either a directory or an explicit CSV filename. The default
destination is ``dataset/anxiety-categorical/raw/data.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import kagglehub


DATASET_HANDLE = "natezhang123/social-anxiety-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_HANDLE}"
KAGGLE_FILE = "enhanced_anxiety_dataset.csv"
OUTPUT_FILENAME = "data.csv"
EXPECTED_ROWS = 11_000
EXPECTED_COLUMNS = (
    "Age",
    "Gender",
    "Occupation",
    "Sleep Hours",
    "Physical Activity (hrs/week)",
    "Caffeine Intake (mg/day)",
    "Alcohol Consumption (drinks/week)",
    "Smoking",
    "Family History of Anxiety",
    "Stress Level (1-10)",
    "Heart Rate (bpm)",
    "Breathing Rate (breaths/min)",
    "Sweating Level (1-5)",
    "Dizziness",
    "Medication",
    "Therapy Sessions (per month)",
    "Recent Major Life Event",
    "Diet Quality (1-10)",
    "Anxiety Level (1-10)",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output directory or explicit CSV filename. Defaults to "
            "dataset/anxiety-categorical/raw/data.csv."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination if it already exists.",
    )
    return parser.parse_args(argv)


def destination_from_output(output: Path) -> Path:
    """Interpret a .csv path as a file and every other path as a directory."""
    output = output.expanduser()
    return output if output.suffix.lower() == ".csv" else output / OUTPUT_FILENAME


def validate_csv(path: Path) -> None:
    """Require the complete Kaggle Social Anxiety schema and row count."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded CSV is empty or missing: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = tuple(next(reader, []))
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


def download(destination: Path, force: bool = False) -> Path:
    """Download, validate, and atomically install the source CSV."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        validate_csv(destination)
        print(f"Already downloaded: {destination}")
        return destination

    with tempfile.TemporaryDirectory(
        prefix=".social-anxiety-", dir=destination.parent
    ) as temporary_directory:
        downloaded = Path(
            kagglehub.dataset_download(
                DATASET_HANDLE,
                path=KAGGLE_FILE,
                output_dir=temporary_directory,
                force_download=True,
            )
        )
        if downloaded.is_dir():
            downloaded = downloaded / KAGGLE_FILE
        if not downloaded.exists():
            matches = list(Path(temporary_directory).rglob(KAGGLE_FILE))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Kaggle download did not produce {KAGGLE_FILE}: {matches}"
                )
            downloaded = matches[0]

        validate_csv(downloaded)
        staging = destination.with_name(f".{destination.name}.partial")
        shutil.copyfile(downloaded, staging)
        os.replace(staging, destination)

    validate_csv(destination)
    size_mib = destination.stat().st_size / (1024 * 1024)
    print(f"Downloaded {destination} ({size_mib:.1f} MiB)")
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    download(destination_from_output(args.output), force=args.force)


if __name__ == "__main__":
    main()
