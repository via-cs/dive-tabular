"""Evaluate continuous consistency with trusted equational ``fix_code``.

For each constraint, every per-column ``fix(df)`` predicts that column from
the other columns in the same synthetic dataframe. The constraint consistency
score is the maximum R2 across those prediction directions. The overall score
is the unweighted mean of the defined per-constraint maxima.

The constraint JSON is trusted input because its Python code is executed.

Examples:
    uv run python -m evaluation.metrics.equational_consistency \
        dataset/flights/constraints_expert/equational_constraint.json \
        experiments/flights/model/synthetic/synthetic_0.csv
    uv run python -m evaluation.metrics.equational_consistency \
        dataset/flights/constraints_expert/equational_constraint.json \
        experiments/flights/model/synthetic \
        --output equational_consistency_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


DEFAULT_REPORT_NAME = "equational_consistency_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "constraints",
        type=Path,
        help="Path to equational_constraint.json.",
    )
    parser.add_argument(
        "data",
        type=Path,
        help=(
            "Synthetic CSV file or directory. Directories are searched "
            "recursively for CSV files."
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

    constraints = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(constraints, list):
        raise ValueError("Constraint JSON must contain a list.")
    if not constraints:
        return []

    seen_ids: set[str] = set()
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            raise ValueError(f"Constraint at index {index} is not an object.")

        missing = {"id", "description", "columns", "fix_code"} - set(constraint)
        if missing:
            raise ValueError(
                f"Constraint at index {index} is missing entries: {sorted(missing)}"
            )

        constraint_id = constraint["id"]
        if not isinstance(constraint_id, str) or not constraint_id:
            raise ValueError(f"Constraint at index {index} has an invalid id.")
        if constraint_id in seen_ids:
            raise ValueError(f"Duplicate constraint id: {constraint_id}")
        seen_ids.add(constraint_id)

        columns = constraint["columns"]
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) and column for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise ValueError(f"Constraint {constraint_id} has invalid columns.")

        fixes = constraint["fix_code"]
        if not isinstance(fixes, list) or not fixes:
            raise ValueError(f"Constraint {constraint_id} has invalid fix_code.")

        fix_columns: list[str] = []
        for fix_index, fix_record in enumerate(fixes):
            if not isinstance(fix_record, dict):
                raise ValueError(
                    f"Constraint {constraint_id} fix {fix_index} is not an object."
                )
            column = fix_record.get("column")
            code = fix_record.get("code")
            if not isinstance(column, str) or not column:
                raise ValueError(
                    f"Constraint {constraint_id} fix {fix_index} has no column."
                )
            # ``code: null`` marks an unavailable repair path for that column.
            if code is not None and (
                not isinstance(code, str) or not code.strip()
            ):
                raise ValueError(
                    f"Constraint {constraint_id} fix for {column} has invalid code."
                )
            fix_columns.append(column)

        if fix_columns != columns:
            raise ValueError(
                f"Constraint {constraint_id} fix columns must exactly match columns "
                f"in the same order; got {fix_columns}, expected {columns}."
            )

    return constraints


def resolve_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Data file must be a CSV: {path}")
        return [path]
    if path.is_dir():
        files = sorted(
            candidate for candidate in path.rglob("*.csv") if candidate.is_file()
        )
        if not files:
            raise FileNotFoundError(f"No CSV files found under: {path}")
        return files
    raise FileNotFoundError(f"Data path not found: {path}")


def load_fix(constraint_id: str, column: str, code: str):
    namespace: dict[str, Any] = {"np": np, "pd": pd}
    try:
        exec(code, namespace)
    except Exception as exc:
        raise ValueError(
            f"Could not compile fix_code for {constraint_id}/{column}: {exc}"
        ) from exc

    fix = namespace.get("fix")
    if not callable(fix):
        raise ValueError(
            f"fix_code for {constraint_id}/{column} must define fix(df)."
        )
    return fix


def evaluate_direction(
    data: pd.DataFrame,
    constraint_id: str,
    fix_record: dict[str, str],
) -> dict[str, Any]:
    column = fix_record["column"]
    fix = load_fix(constraint_id, column, fix_record["code"])
    try:
        predicted = fix(data.copy())
    except Exception as exc:
        raise RuntimeError(
            f"Prediction {constraint_id}/{column} failed while executing: {exc}"
        ) from exc

    if not isinstance(predicted, pd.Series):
        raise TypeError(
            f"Prediction {constraint_id}/{column} returned "
            f"{type(predicted).__name__}; expected pandas.Series."
        )
    if len(predicted) != len(data) or not predicted.index.equals(data.index):
        raise ValueError(
            f"Prediction {constraint_id}/{column} returned a misaligned Series."
        )

    actual_values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual_values) & np.isfinite(predicted_values)
    finite_pairs = int(finite.sum())

    excluded_rows = int(len(data) - finite_pairs)
    result: dict[str, Any] = {
        "column": column,
        "rows": int(len(data)),
        "finite_pairs": finite_pairs,
        "excluded_rows": excluded_rows,
    }
    if excluded_rows > 0:
        print(
            f"Warning: {constraint_id}/{column} dropped {excluded_rows} of "
            f"{len(data)} rows with non-finite actual or predicted values "
            f"before R2 ({finite_pairs} finite pairs remain).",
            flush=True,
        )
    if finite_pairs < 2:
        result.update(
            {
                "r2": None,
                "undefined_reason": "R2 requires at least two finite pairs.",
            }
        )
        return result

    score = float(
        r2_score(
            actual_values[finite],
            predicted_values[finite],
            force_finite=True,
        )
    )
    if not np.isfinite(score):
        result.update(
            {
                "r2": None,
                "undefined_reason": "R2 calculation returned a non-finite value.",
            }
        )
        return result

    result["r2"] = score
    return result


def evaluate_dataframe(
    data: pd.DataFrame,
    constraints: list[dict[str, Any]],
    source: str = "Dataframe",
) -> dict[str, Any]:
    per_constraint: list[dict[str, Any]] = []

    for constraint in constraints:
        constraint_id = constraint["id"]
        missing = [column for column in constraint["columns"] if column not in data]
        if missing:
            raise ValueError(
                f"{source}: constraint {constraint_id} is missing columns {missing}"
            )

        column_scores = [
            evaluate_direction(data, constraint_id, fix_record)
            for fix_record in constraint["fix_code"]
            if isinstance(fix_record.get("code"), str)
            and fix_record["code"].strip()
        ]
        defined_scores = [entry for entry in column_scores if entry["r2"] is not None]
        if defined_scores:
            selected = max(defined_scores, key=lambda entry: entry["r2"])
            consistency = selected["r2"]
            selected_column = selected["column"]
        else:
            consistency = None
            selected_column = None

        per_constraint.append(
            {
                "id": constraint_id,
                "description": constraint["description"],
                "columns": constraint["columns"],
                "column_scores": column_scores,
                "r2_consistency": consistency,
                "selected_column": selected_column,
            }
        )

    defined_constraints = [
        entry for entry in per_constraint if entry["r2_consistency"] is not None
    ]
    average = (
        float(np.mean([entry["r2_consistency"] for entry in defined_constraints]))
        if defined_constraints
        else None
    )
    return {
        "rows": int(len(data)),
        "per_constraint": per_constraint,
        "overall": {
            "aggregation": "unweighted_mean_of_per_constraint_maximum_r2",
            "constraints_evaluated": len(constraints),
            "constraints_with_defined_score": len(defined_constraints),
            "constraints_with_undefined_score": len(constraints)
            - len(defined_constraints),
            "average_r2_consistency": average,
        },
    }


def evaluate_csv(
    csv_path: Path,
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    data = pd.read_csv(csv_path, float_precision="round_trip")
    report = evaluate_dataframe(data, constraints, str(csv_path))
    return {"csv_path": str(csv_path), **report}


def resolve_output(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path / DEFAULT_REPORT_NAME


def print_result(result: dict[str, Any]) -> None:
    print(f"\nCSV: {result['csv_path']}")
    print(f"Rows: {result['rows']:,}")
    print(f"{'Constraint':48} {'Selected column':32} {'Maximum R2':>14}")
    print("-" * 98)
    for entry in result["per_constraint"]:
        score = entry["r2_consistency"]
        score_text = "n/a" if score is None else f"{score:.8f}"
        selected = entry["selected_column"] or "n/a"
        print(f"{entry['id'][:48]:48} {selected[:32]:32} {score_text:>14}")
    print("-" * 98)
    average = result["overall"]["average_r2_consistency"]
    average_text = "n/a" if average is None else f"{average:.8f}"
    print(f"Average R2 consistency: {average_text}")


def main() -> None:
    args = parse_args()
    constraints = load_constraints(args.constraints)
    csv_files = resolve_csv_files(args.data)
    datasets = [evaluate_csv(path, constraints) for path in csv_files]
    report = {
        "constraints_path": str(args.constraints),
        "data_path": str(args.data),
        "datasets": datasets,
    }

    for result in datasets:
        print_result(result)

    if args.output is not None:
        output_path = resolve_output(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {output_path}")


if __name__ == "__main__":
    main()
