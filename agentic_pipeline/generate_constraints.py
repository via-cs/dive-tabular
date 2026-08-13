"""Generate categorical, equational, and linear constraints in one run.

The family pipelines run in strict categorical -> equational -> linear order.
By default, linear constraints are removed when every involved column is already
involved in at least one generated equational constraint. Experiments can retain
those constraints explicitly with ``--keep-equationally-covered-linear``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from agentic_pipeline.categorical.config import DEFAULT_MODEL
from agentic_pipeline.categorical.dataset_io import (
    write_json,
    write_json_atomic,
)
from agentic_pipeline.model_backends import SUPPORTED_PROVIDERS


FAMILY_ORDER = ("categorical", "equational", "linear")
CONSTRAINT_FILENAMES = {
    "categorical": "categorical_dependency_constraint.json",
    "equational": "equational_constraint.json",
    "linear": "linear_constraint.json",
}
REPORT_FILENAME = "run_report_all_constraints.json"
MANAGED_OPTIONS = {
    "output_dir",
    "model",
    "provider",
    "seed",
    "dry_run",
}


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"{label} must be a JSON object: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{label} must be a JSON object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meta",
        type=Path,
        help="Path to meta.json or a directory containing meta.json.",
    )
    parser.add_argument(
        "data",
        type=Path,
        help="Path to data.csv or a directory containing data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agentic_constraints/all"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default="openai",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-equationally-covered-linear",
        action="store_true",
        help=(
            "Preserve verified linear constraints even when all of their "
            "columns occur in generated equational constraints."
        ),
    )
    parser.add_argument(
        "--skip-equational-fix-generation",
        action="store_true",
        help=(
            "Run equational constraint discovery without downstream repair "
            "code generation."
        ),
    )
    for family in FAMILY_ORDER:
        parser.add_argument(f"--{family}-model", default=None)
        parser.add_argument(
            f"--{family}-provider",
            choices=SUPPORTED_PROVIDERS,
            default=None,
        )
        parser.add_argument(
            f"--{family}-arguments",
            type=lambda value, name=family: _json_object(
                value, f"--{name}-arguments"
            ),
            default={},
            metavar="JSON",
            help=(
                f"Additional {family} pipeline options as a JSON object. "
                "Managed options cannot be overridden."
            ),
        )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write all three family request previews without model calls.",
    )
    return parser.parse_args()


def append_options(
    command: list[str],
    options: dict[str, Any],
    *,
    reserved: Iterable[str] = MANAGED_OPTIONS,
) -> None:
    reserved_set = set(reserved)
    for name, value in options.items():
        if name in reserved_set:
            raise ValueError(f"Option {name!r} is managed by the orchestrator")
        if value is None or value is False:
            continue
        option = "--" + name.replace("_", "-")
        command.append(option)
        if value is True:
            continue
        if isinstance(value, list):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))


def family_command(
    family: str,
    *,
    meta: Path,
    data: Path,
    output_dir: Path,
    model: str,
    provider: str,
    seed: int,
    arguments: dict[str, Any],
    dry_run: bool,
    skip_equational_fix_generation: bool = False,
) -> list[str]:
    if family not in FAMILY_ORDER:
        raise ValueError(f"Unknown constraint family: {family}")
    command = [
        sys.executable,
        "-m",
        f"agentic_pipeline.{family}",
        str(meta),
        str(data),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--provider",
        provider,
        "--seed",
        str(seed),
    ]
    reserved = set(MANAGED_OPTIONS)
    if family == "equational":
        reserved.add("skip_fix_generation")
    append_options(command, arguments, reserved=reserved)
    if family == "equational" and skip_equational_fix_generation:
        command.append("--skip-fix-generation")
    if dry_run:
        command.append("--dry-run")
    return command


def load_constraint_list(path: Path, family: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{family} constraint file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{family} constraint file must contain a JSON list")
    for index, constraint in enumerate(value):
        if not isinstance(constraint, dict):
            raise ValueError(
                f"{family} constraint at index {index} must be an object"
            )
        columns = constraint.get("columns")
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) and column for column in columns)
        ):
            raise ValueError(
                f"{family} constraint at index {index} has invalid columns"
            )
    return value


def filter_linear_constraints(
    equational: list[dict[str, Any]],
    linear: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop linear constraints fully covered by the union of equation columns."""
    equational_columns = {
        column
        for constraint in equational
        for column in constraint["columns"]
    }
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for constraint in linear:
        columns = set(constraint["columns"])
        if columns and columns.issubset(equational_columns):
            covering_ids = [
                equation["id"]
                for equation in equational
                if columns & set(equation["columns"])
            ]
            dropped.append(
                {
                    "id": constraint.get("id"),
                    "columns": constraint["columns"],
                    "covering_equational_constraint_ids": covering_ids,
                }
            )
        else:
            kept.append(constraint)
    return kept, {
        "policy": "drop_if_columns_subset_of_equational_column_union",
        "equational_columns": sorted(equational_columns),
        "input_linear_constraints": len(linear),
        "kept_linear_constraints": len(kept),
        "dropped_linear_constraints": dropped,
    }


def categorical_discovery_skip_reason(meta: Path) -> str | None:
    meta_path = meta / "meta.json" if meta.is_dir() else meta
    info_path = meta_path.with_name("info.json")
    if not info_path.is_file():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    column_types = info.get("col_types")
    if not isinstance(column_types, dict):
        return None
    categorical_columns = [
        name
        for name, specification in column_types.items()
        if isinstance(specification, dict)
        and specification.get("type") == "cat"
    ]
    if len(categorical_columns) < 2:
        return (
            "fewer_than_two_categorical_columns: "
            f"{categorical_columns}"
        )
    return None


def generate_all_constraints(
    *,
    meta: Path,
    data: Path,
    output_dir: Path,
    family_settings: dict[str, dict[str, Any]],
    seed: int = 42,
    dry_run: bool = False,
    keep_equationally_covered_linear: bool = False,
    skip_equational_fix_generation: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    categorical_skip = categorical_discovery_skip_reason(meta)

    for family in FAMILY_ORDER:
        output = output_dir / CONSTRAINT_FILENAMES[family]
        if family == "categorical" and categorical_skip is not None:
            if dry_run:
                write_json(
                    output_dir / "request_preview_categorical.json",
                    {
                        "skipped": True,
                        "reason": categorical_skip,
                        "meta_path": str(meta),
                        "data_path": str(data),
                    },
                )
            else:
                write_json_atomic(
                    output,
                    {"dsl_version": "2.0", "constraints": []},
                )
                write_json(
                    output_dir / "run_report_categorical.json",
                    {
                        "skipped": True,
                        "reason": categorical_skip,
                        "published_constraints": 0,
                    },
                )
            commands.append(
                {
                    "family": family,
                    "status": "skipped",
                    "reason": categorical_skip,
                    "command": None,
                    "output": None if dry_run else str(output),
                }
            )
            continue

        settings = family_settings[family]
        command = family_command(
            family,
            meta=meta,
            data=data,
            output_dir=output_dir,
            model=str(settings["model"]),
            provider=str(settings.get("provider", "openai")),
            seed=seed,
            arguments=dict(settings.get("arguments", {})),
            dry_run=dry_run,
            skip_equational_fix_generation=skip_equational_fix_generation,
        )
        subprocess.run(command, check=True)
        if not dry_run and not output.is_file():
            raise RuntimeError(
                f"{family} pipeline completed without writing {output}"
            )
        commands.append(
            {
                "family": family,
                "status": "completed",
                "command": command,
                "output": None if dry_run else str(output),
            }
        )

    if keep_equationally_covered_linear:
        linear_filter: dict[str, Any] = {
            "status": "disabled",
            "policy": "keep_all_verified_linear_constraints",
            "reason": "requested_by_caller",
        }
    elif dry_run:
        linear_filter: dict[str, Any] = {"status": "skipped_dry_run"}
    else:
        equational_path = output_dir / CONSTRAINT_FILENAMES["equational"]
        linear_path = output_dir / CONSTRAINT_FILENAMES["linear"]
        equational = load_constraint_list(equational_path, "equational")
        linear = load_constraint_list(linear_path, "linear")
        filtered, linear_filter = filter_linear_constraints(
            equational, linear
        )
        write_json_atomic(linear_path, filtered)

        linear_report_path = output_dir / "run_report_linear.json"
        if linear_report_path.is_file():
            linear_report = json.loads(
                linear_report_path.read_text(encoding="utf-8")
            )
            if isinstance(linear_report, dict):
                linear_report["combined_pipeline_filter"] = linear_filter
                linear_report["published_constraints_after_filter"] = len(
                    filtered
                )
                write_json_atomic(linear_report_path, linear_report)

    report = {
        "family_order": list(FAMILY_ORDER),
        "meta_path": str(meta),
        "data_path": str(data),
        "output_dir": str(output_dir),
        "seed": seed,
        "dry_run": dry_run,
        "keep_equationally_covered_linear": (
            keep_equationally_covered_linear
        ),
        "skip_equational_fix_generation": skip_equational_fix_generation,
        "commands": commands,
        "linear_filter": linear_filter,
    }
    write_json(output_dir / REPORT_FILENAME, report)
    return report


def main() -> None:
    args = parse_args()
    settings = {
        family: {
            "model": getattr(args, f"{family}_model") or args.model,
            "provider": (
                getattr(args, f"{family}_provider") or args.provider
            ),
            "arguments": getattr(args, f"{family}_arguments"),
        }
        for family in FAMILY_ORDER
    }
    report = generate_all_constraints(
        meta=args.meta,
        data=args.data,
        output_dir=args.output_dir,
        family_settings=settings,
        seed=args.seed,
        dry_run=args.dry_run,
        keep_equationally_covered_linear=(
            args.keep_equationally_covered_linear
        ),
        skip_equational_fix_generation=(
            args.skip_equational_fix_generation
        ),
    )
    if args.dry_run:
        print(
            "Wrote all family request previews and "
            f"{args.output_dir / REPORT_FILENAME}"
        )
    else:
        if args.keep_equationally_covered_linear:
            print(
                "Wrote categorical, equational, and unfiltered linear "
                f"constraints to {args.output_dir}"
            )
        else:
            print(
                "Wrote categorical, equational, and filtered linear "
                f"constraints to {args.output_dir}"
            )
            dropped = report["linear_filter"]["dropped_linear_constraints"]
            print(
                f"Removed {len(dropped)} equationally covered linear "
                "constraints"
            )


if __name__ == "__main__":
    main()
