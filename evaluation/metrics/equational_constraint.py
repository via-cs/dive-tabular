"""Evaluate row-level equational constraints on one or more CSV files.

Each constraint must define ``check(df)`` and return a Boolean pandas Series
where True means that the row passes the constraint.

Examples:
    uv run python -m evaluation.metrics.equational_constraint \
        data/flights/constraints_expert/equational_constraint.json
    uv run python -m evaluation.metrics.equational_constraint \
        data/flights/constraints_expert/equational_constraint.json \
        experiments/flights/tvae/unconstrained/synthetic/synthetic_0.csv
    uv run python -m evaluation.metrics.equational_constraint \
        data/flights/constraints_expert/equational_constraint.json \
        experiments/flights/tvae/repair_test --output report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA = Path("data/flights/data.csv")
DEFAULT_REPORT_NAME = "equational_check_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "constraints",
        type=Path,
        help="Path to equational_constraint.json.",
    )
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA,
        help=(
            "CSV file or directory to evaluate. Directories are searched "
            f"recursively for CSV files (default: {DEFAULT_DATA})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON report path or output directory. A directory uses "
            f"the filename {DEFAULT_REPORT_NAME}."
        ),
    )
    return parser.parse_args()


def load_constraints(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Constraint JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Constraint JSON must contain a list.")
    if not payload:
        return []

    required = {"id", "description", "columns", "check_code"}
    seen_ids: set[str] = set()
    for index, constraint in enumerate(payload):
        if not isinstance(constraint, dict):
            raise ValueError(f"Constraint at index {index} is not an object.")
        missing = sorted(required - set(constraint))
        if missing:
            raise ValueError(
                f"Constraint at index {index} is missing entries: {missing}"
            )
        constraint_id = constraint["id"]
        if not isinstance(constraint_id, str) or not constraint_id:
            raise ValueError(f"Constraint at index {index} has an invalid id.")
        if constraint_id in seen_ids:
            raise ValueError(f"Duplicate constraint id: {constraint_id}")
        seen_ids.add(constraint_id)
        if not isinstance(constraint["columns"], list) or not constraint["columns"]:
            raise ValueError(f"Constraint {constraint_id} has invalid columns.")
        if not isinstance(constraint["check_code"], str):
            raise ValueError(f"Constraint {constraint_id} has invalid check_code.")
    return payload


def resolve_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Data file must be a CSV: {path}")
        return [path]
    if path.is_dir():
        files = sorted(candidate for candidate in path.rglob("*.csv") if candidate.is_file())
        if not files:
            raise FileNotFoundError(f"No CSV files found under: {path}")
        return files
    raise FileNotFoundError(f"Data path not found: {path}")


def load_check(constraint: dict[str, Any]):
    namespace: dict[str, Any] = {"pd": pd, "np": np}
    try:
        exec(constraint["check_code"], namespace)
    except Exception as exc:
        raise ValueError(
            f"Could not compile check_code for {constraint['id']}: {exc}"
        ) from exc
    check = namespace.get("check")
    if not callable(check):
        raise ValueError(
            f"Constraint {constraint['id']} check_code must define check(df)."
        )
    return check


def evaluate_dataframe_with_masks(
    data: pd.DataFrame,
    constraints: list[dict[str, Any]],
    source: str = "Dataframe",
) -> tuple[dict[str, Any], list[pd.Series]]:
    violation_masks: list[pd.Series] = []
    per_constraint = []

    for constraint in constraints:
        constraint_id = constraint["id"]
        missing = [column for column in constraint["columns"] if column not in data]
        if missing:
            raise ValueError(
                f"{source}: constraint {constraint_id} is missing columns {missing}"
            )

        check = load_check(constraint)
        try:
            pass_mask = check(data.copy())
        except Exception as exc:
            raise RuntimeError(
                f"{source}: check {constraint_id} failed: {exc}"
            ) from exc
        if not isinstance(pass_mask, pd.Series):
            raise TypeError(
                f"{source}: check {constraint_id} did not return a pandas Series."
            )
        if len(pass_mask) != len(data) or not pass_mask.index.equals(data.index):
            raise ValueError(
                f"{source}: check {constraint_id} returned a misaligned mask."
            )
        if pass_mask.isna().any():
            raise ValueError(
                f"{source}: check {constraint_id} returned missing mask values."
            )
        if not pd.api.types.is_bool_dtype(pass_mask.dtype):
            raise TypeError(
                f"{source}: check {constraint_id} did not return a Boolean mask."
            )

        violation_mask = ~pass_mask
        violation_masks.append(violation_mask)
        violations = int(violation_mask.sum())
        rows = int(len(data))
        per_constraint.append(
            {
                "id": constraint_id,
                "description": constraint["description"],
                "columns": constraint["columns"],
                "rows_checked": rows,
                "passes": rows - violations,
                "violations": violations,
                "violation_rate": float(violations / rows) if rows else None,
            }
        )

    rows = int(len(data))
    if violation_masks:
        violation_frame = pd.concat(violation_masks, axis=1)
        any_violation = violation_frame.any(axis=1)
        all_pass = ~any_violation
        total_violations = int(violation_frame.to_numpy().sum())
    else:
        any_violation = pd.Series(False, index=data.index)
        all_pass = pd.Series(True, index=data.index)
        total_violations = 0
    total_checks = rows * len(constraints)

    report = {
        "rows": rows,
        "per_constraint": per_constraint,
        "overall": {
            "constraints_evaluated": len(constraints),
            "rows_checked": rows,
            "rows_passing_all_constraints": int(all_pass.sum()),
            "rows_with_any_violation": int(any_violation.sum()),
            "row_violation_rate": (
                float(any_violation.mean()) if rows else None
            ),
            "total_constraint_violations": total_violations,
            "total_constraint_checks": total_checks,
            "check_violation_rate": (
                float(total_violations / total_checks) if total_checks else None
            ),
        },
    }
    return report, violation_masks


def evaluate_dataframe(
    data: pd.DataFrame,
    constraints: list[dict[str, Any]],
    source: str = "Dataframe",
) -> dict[str, Any]:
    report, _ = evaluate_dataframe_with_masks(data, constraints, source)
    return report


def evaluate_csv(
    csv_path: Path, constraints: list[dict[str, Any]]
) -> dict[str, Any]:
    data = pd.read_csv(csv_path, float_precision="round_trip")
    report = evaluate_dataframe(data, constraints, str(csv_path))
    return {"csv_path": str(csv_path), **report}


def print_result(result: dict[str, Any]) -> None:
    print(f"\nCSV: {result['csv_path']}")
    print(f"Rows: {result['rows']:,}")
    print(f"{'Constraint':48} {'Violations':>12} {'Rate':>12}")
    print("-" * 74)
    for entry in result["per_constraint"]:
        rate = entry["violation_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.6%}"
        print(f"{entry['id'][:48]:48} {entry['violations']:12,} {rate_text:>12}")

    overall = result["overall"]
    row_rate = overall["row_violation_rate"]
    check_rate = overall["check_violation_rate"]
    print("-" * 74)
    print(
        "Rows violating at least one constraint: "
        f"{overall['rows_with_any_violation']:,} "
        f"({row_rate:.6%})" if row_rate is not None else "n/a"
    )
    print(
        "Violations across all row-constraint checks: "
        f"{overall['total_constraint_violations']:,} / "
        f"{overall['total_constraint_checks']:,} "
        f"({check_rate:.6%})" if check_rate is not None else "n/a"
    )


def resolve_output(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path / DEFAULT_REPORT_NAME


def main() -> None:
    args = parse_args()
    constraints = load_constraints(args.constraints)
    csv_files = resolve_csv_files(args.data)
    results = [evaluate_csv(path, constraints) for path in csv_files]
    report = {
        "constraints_path": str(args.constraints),
        "mask_semantics": "true_means_pass",
        "datasets": results,
    }

    for result in results:
        print_result(result)

    if args.output is not None:
        output_path = resolve_output(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {output_path}")


if __name__ == "__main__":
    main()
