"""Score equational repair candidates by their change in KS complement.

For every constraint and target column, this script executes the corresponding
``fix(df)`` function against the full synthetic dataframe. It then compares the
target column with the training column before and after replacement. Each
candidate is evaluated independently from the original synthetic data.

The constraint file is trusted input: its ``fix_code`` strings are executed as
Python code. This module is internal; use
``constraints.expert_constraints_fix`` for repair workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sdmetrics.single_column import KSComplement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synthetic", type=Path, help="Synthetic CSV file to repair.")
    parser.add_argument("train", type=Path, help="Real training CSV file.")
    parser.add_argument(
        "constraints",
        type=Path,
        help="Equational-constraint JSON containing per-column fix_code entries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path at which to save the JSON report.",
    )
    return parser.parse_args()


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"{label} data must be a CSV file: {path}")
    return pd.read_csv(path, float_precision="round_trip")


def load_constraints(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Constraint JSON file not found: {path}")

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


def has_executable_fix(fix_record: dict[str, Any]) -> bool:
    """Return True when ``fix_code`` entry has non-empty repair source."""
    code = fix_record.get("code")
    return isinstance(code, str) and bool(code.strip())


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


def finite_values(series: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = numeric[np.isfinite(numeric)]
    if values.size == 0:
        raise ValueError(f"{label} has no finite numerical values for the KS test.")
    return values


def ks_metrics(real: np.ndarray, synthetic: np.ndarray) -> tuple[float, float]:
    complement = float(KSComplement.compute(real, synthetic))
    if not np.isfinite(complement):
        raise ValueError("SDMetrics KSComplement returned a non-finite score.")
    return 1.0 - complement, complement


def evaluate_repair(
    synthetic: pd.DataFrame,
    real_values: np.ndarray,
    constraint_id: str,
    fix_record: dict[str, str],
) -> dict[str, Any]:
    column = fix_record["column"]
    before_values = finite_values(
        synthetic[column], f"synthetic column {column!r} before repair"
    )
    before_statistic, before_complement = ks_metrics(real_values, before_values)

    fix = load_fix(constraint_id, column, fix_record["code"])
    try:
        repaired = fix(synthetic.copy())
    except Exception as exc:
        raise RuntimeError(
            f"Repair {constraint_id}/{column} failed while executing: {exc}"
        ) from exc

    if not isinstance(repaired, pd.Series):
        raise TypeError(
            f"Repair {constraint_id}/{column} returned {type(repaired).__name__}; "
            "expected pandas.Series."
        )
    if len(repaired) != len(synthetic) or not repaired.index.equals(synthetic.index):
        raise ValueError(
            f"Repair {constraint_id}/{column} returned a misaligned pandas Series."
        )

    after_values = finite_values(
        repaired, f"synthetic column {column!r} after repair"
    )
    after_statistic, after_complement = ks_metrics(real_values, after_values)

    return {
        "column": column,
        "train_finite_values": int(real_values.size),
        "synthetic_finite_values_before": int(before_values.size),
        "synthetic_finite_values_after": int(after_values.size),
        "ks_statistic_before": before_statistic,
        "ks_complement_before": before_complement,
        "ks_statistic_after": after_statistic,
        "ks_complement_after": after_complement,
        "delta_ks_complement": after_complement - before_complement,
    }


def evaluate_constraints(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    constraints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []

    for constraint in constraints:
        constraint_id = constraint["id"]
        missing_synthetic = [
            column for column in constraint["columns"] if column not in synthetic
        ]
        missing_train = [
            column for column in constraint["columns"] if column not in train
        ]
        if missing_synthetic:
            raise ValueError(
                f"Synthetic data is missing columns for {constraint_id}: "
                f"{missing_synthetic}"
            )
        if missing_train:
            raise ValueError(
                f"Training data is missing columns for {constraint_id}: {missing_train}"
            )

        repairs = []
        for fix_record in constraint["fix_code"]:
            if not has_executable_fix(fix_record):
                continue
            column = fix_record["column"]
            real_values = finite_values(
                train[column], f"training column {column!r}"
            )
            repairs.append(
                evaluate_repair(
                    synthetic,
                    real_values,
                    constraint_id,
                    fix_record,
                )
            )

        repairs.sort(key=lambda repair: repair["delta_ks_complement"], reverse=True)
        for rank, repair in enumerate(repairs, start=1):
            repair["rank"] = rank

        reports.append(
            {
                "id": constraint_id,
                "description": constraint["description"],
                "columns": constraint["columns"],
                "repairs": repairs,
            }
        )

    return reports


def main() -> None:
    args = parse_args()
    synthetic = load_csv(args.synthetic, "Synthetic")
    train = load_csv(args.train, "Training")
    constraints = load_constraints(args.constraints)

    report = {
        "synthetic_file": str(args.synthetic),
        "train_file": str(args.train),
        "constraints_file": str(args.constraints),
        "synthetic_rows": int(len(synthetic)),
        "train_rows": int(len(train)),
        "delta_definition": "ks_complement_after - ks_complement_before",
        "constraints": evaluate_constraints(synthetic, train, constraints),
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)

