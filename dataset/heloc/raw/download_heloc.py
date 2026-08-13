"""Download the FICO HELOC dataset from Hugging Face as a CSV.

Examples:
    uv run python dataset/heloc/raw/download_heloc.py
    uv run python dataset/heloc/raw/download_heloc.py dataset/heloc/raw
    uv run python dataset/heloc/raw/download_heloc.py /tmp/heloc.csv --force

The output may be either a directory or an explicit CSV filename.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


DATASET_HANDLE = "mstz/heloc"
DATASET_REVISION = "edbc126972334cf5203739cf519782a23d46c5ea"
SOURCE_URL = (
    "https://huggingface.co/datasets/mstz/heloc/raw/"
    f"{DATASET_REVISION}/heloc.csv"
)
OUTPUT_FILE = "heloc_dataset_v1.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent

EXPECTED_COLUMNS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output directory or explicit CSV filename. Defaults to "
            "dataset/heloc/raw/heloc_dataset_v1.csv."
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
    return output if output.suffix.lower() == ".csv" else output / OUTPUT_FILE


def validate_csv(path: Path) -> None:
    """Require a non-empty CSV with the original FICO HELOC schema."""
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty or missing: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = tuple(next(reader, []))
        row_count = sum(1 for _ in reader)

    if header != EXPECTED_COLUMNS:
        raise RuntimeError(
            "Downloaded CSV has an unexpected schema. "
            f"Expected {list(EXPECTED_COLUMNS)}, got {list(header)}."
        )
    if row_count != 10_459:
        raise RuntimeError(
            f"Downloaded HELOC CSV has {row_count:,} rows; expected 10,459."
        )


def download(destination: Path, force: bool = False) -> Path:
    """Download the pinned public CSV and atomically install it."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        validate_csv(destination)
        print(f"Already downloaded: {destination}")
        return destination

    with tempfile.TemporaryDirectory(
        prefix=".heloc-", dir=destination.parent
    ) as temporary_directory:
        downloaded = Path(temporary_directory) / OUTPUT_FILE
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "LLM-Tabular HELOC downloader"},
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            downloaded.open("wb") as output_file,
        ):
            while block := response.read(1024 * 1024):
                output_file.write(block)

        validate_csv(downloaded)
        staging = destination.with_name(f".{destination.name}.partial")
        shutil.copyfile(downloaded, staging)
        os.replace(staging, destination)

    validate_csv(destination)
    size_kib = destination.stat().st_size / 1024
    print(f"Downloaded {destination} ({size_kib:.1f} KiB)")
    return destination


def main() -> None:
    args = parse_args()
    download(destination_from_output(args.output), force=args.force)


if __name__ == "__main__":
    main()
