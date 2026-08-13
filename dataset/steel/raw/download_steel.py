"""Download the Steel Industry Energy Consumption CSV from Kaggle.

Examples:
    uv run python data/steel/raw/download_steel.py
    uv run python data/steel/raw/download_steel.py data/steel/raw
    uv run python data/steel/raw/download_steel.py /tmp/steel.csv --force

The output may be either a directory or an explicit CSV filename.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

import kagglehub


DATASET_HANDLE = "csafrit2/steel-industry-energy-consumption"
KAGGLE_FILE = "Steel_industry_data.csv"
EXPECTED_COLUMNS = {
    "date",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("data/steel/raw"),
        help=(
            "Output directory or explicit CSV filename. Defaults to "
            "data/steel/raw/Steel_industry_data.csv."
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
    return output if output.suffix.lower() == ".csv" else output / KAGGLE_FILE


def validate_csv(path: Path) -> None:
    """Check that a file has the expected Steel dataset columns."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty or missing: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
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
        prefix=".steel-industry-", dir=destination.parent
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
