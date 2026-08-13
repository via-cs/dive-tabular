"""Restore packaged split constraints into a newly prepared experiment.

Directory mode discovers the three supported JSON files under every
``splits/split_*/constraints`` directory. Exact-file mode bypasses directory
discovery and accepts repeated explicit SOURCE TARGET pairs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


CONSTRAINT_FILENAMES = (
    "categorical_dependency_constraint.json",
    "equational_constraint.json",
    "linear_constraint.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference_experiment_dir",
        nargs="?",
        type=Path,
        help="Packaged experiment directory containing splits/split_*/constraints.",
    )
    parser.add_argument(
        "target_experiment_dir",
        nargs="?",
        type=Path,
        help="Prepared experiment directory that will receive the constraints.",
    )
    parser.add_argument(
        "--constraint-file",
        action="append",
        nargs=2,
        type=Path,
        default=[],
        metavar=("SOURCE", "TARGET"),
        help=(
            "Copy one exact source JSON to one exact target path. Repeat for "
            "additional files. When supplied, directory discovery is bypassed."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace target constraint files that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned copies without writing files.",
    )
    return parser.parse_args(argv)


def validate_constraint_file(path: Path) -> None:
    if path.name not in CONSTRAINT_FILENAMES:
        raise ValueError(
            f"Unsupported constraint filename {path.name!r}; expected one of "
            f"{CONSTRAINT_FILENAMES}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Constraint file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "categorical_dependency_constraint.json":
        if not isinstance(value, dict) or not isinstance(value.get("constraints"), list):
            raise ValueError(
                "Categorical constraints must contain an object with a "
                f"constraints list: {path}"
            )
        constraints = value["constraints"]
    else:
        if not isinstance(value, list):
            raise ValueError(f"Constraint file must contain a JSON list: {path}")
        constraints = value
    if not all(isinstance(item, dict) for item in constraints):
        raise ValueError(f"Every constraint entry must be an object: {path}")


def directory_pairs(reference: Path, target: Path) -> list[tuple[Path, Path]]:
    reference = reference.expanduser().resolve()
    target = target.expanduser().resolve()
    source_split_root = reference / "splits"
    target_split_root = target / "splits"
    if not source_split_root.is_dir():
        raise FileNotFoundError(f"Reference splits directory not found: {source_split_root}")
    if not target_split_root.is_dir():
        raise FileNotFoundError(
            f"Target splits directory not found: {target_split_root}; run the "
            "multirun prepare stage first."
        )

    split_dirs = sorted(path for path in source_split_root.glob("split_*") if path.is_dir())
    if not split_dirs:
        raise FileNotFoundError(f"No reference split directories found: {source_split_root}")

    pairs: list[tuple[Path, Path]] = []
    for source_split in split_dirs:
        target_split = target_split_root / source_split.name
        if not target_split.is_dir():
            raise FileNotFoundError(f"Matching target split not found: {target_split}")
        for filename in CONSTRAINT_FILENAMES:
            pairs.append(
                (
                    source_split / "constraints" / filename,
                    target_split / "constraints" / filename,
                )
            )
    return pairs


def selected_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.constraint_file:
        if args.reference_experiment_dir is not None or args.target_experiment_dir is not None:
            raise ValueError(
                "Use either the two experiment directories or repeated "
                "--constraint-file SOURCE TARGET pairs, not both."
            )
        return [
            (source.expanduser().resolve(), target.expanduser().resolve())
            for source, target in args.constraint_file
        ]
    if args.reference_experiment_dir is None or args.target_experiment_dir is None:
        raise ValueError(
            "Directory mode requires REFERENCE_EXPERIMENT_DIR and "
            "TARGET_EXPERIMENT_DIR."
        )
    return directory_pairs(args.reference_experiment_dir, args.target_experiment_dir)


def restore(
    pairs: list[tuple[Path, Path]], *, force: bool, dry_run: bool
) -> int:
    seen_targets: set[Path] = set()
    for source, target in pairs:
        validate_constraint_file(source)
        if target.name != source.name:
            raise ValueError(
                f"Source and target filenames must match: {source.name!r} != {target.name!r}"
            )
        if source == target:
            raise ValueError(f"Source and target are the same file: {source}")
        if target in seen_targets:
            raise ValueError(f"Duplicate target path: {target}")
        seen_targets.add(target)
        if target.exists() and not force:
            raise FileExistsError(f"Target already exists: {target}; use --force to replace it")

    for source, target in pairs:
        print(f"{'would copy' if dry_run else 'copy'}: {source} -> {target}")
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.partial")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    return len(pairs)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        pairs = selected_pairs(args)
        count = restore(pairs, force=args.force, dry_run=args.dry_run)
    except (FileExistsError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{'validated' if args.dry_run else 'restored'} {count} constraint files")


if __name__ == "__main__":
    main()
