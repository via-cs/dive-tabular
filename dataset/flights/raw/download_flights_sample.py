"""Download the 3-million-row flight sample from Kaggle.

Examples:
    uv run python dataset/flights/raw/download_flights_sample.py
    uv run python dataset/flights/raw/download_flights_sample.py dataset/flights/raw
    uv run python dataset/flights/raw/download_flights_sample.py /tmp/flights.csv --force
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

import kagglehub


DATASET_HANDLE = "patrickzel/flight-delay-and-cancellation-dataset-2019-2023"
KAGGLE_FILE = "flights_sample_3m.csv"
EXPECTED_COLUMNS = {
    "FL_DATE",
    "AIRLINE_CODE",
    "FL_NUMBER",
    "ORIGIN",
    "DEST",
    "ARR_DELAY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("dataset/flights/raw"),
        help=(
            "Output directory or explicit CSV filename. "
            "Defaults to dataset/flights/raw/flights_sample_3m.csv."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination if it already exists.",
    )
    return parser.parse_args()


def destination_from_output(output: Path) -> Path:
    """Interpret a .csv path as a file and every other path as a directory."""
    output = output.expanduser()
    if output.suffix.lower() == ".csv":
        return output
    return output / KAGGLE_FILE


def validate_csv(path: Path) -> None:
    """Check that the downloaded file looks like the expected flight dataset."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty or missing: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        header = next(csv.reader(file), [])

    missing = EXPECTED_COLUMNS.difference(header)
    if missing:
        raise RuntimeError(
            f"Downloaded CSV is missing expected columns: {sorted(missing)}"
        )


def download(destination: Path, force: bool = False) -> Path:
    """Download the Kaggle file and atomically install it at destination."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        validate_csv(destination)
        print(f"Already downloaded: {destination}")
        return destination

    with tempfile.TemporaryDirectory(
        prefix=".flights_sample_3m-", dir=destination.parent
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


def main() -> None:
    args = parse_args()
    download(destination_from_output(args.output), force=args.force)


if __name__ == "__main__":
    main()
