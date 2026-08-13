"""Evaluate categorical dependency constraints expressed in DSL 2.0.

Each constraint maps one or more determinant columns to an allowed set for one
categorical dependent column. Rows whose determinant configuration is absent
from a value table are outside that constraint's scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from agentic_pipeline.categorical.graph import cycle_details, add_dependency
from agentic_pipeline.categorical.models import (
    CategoricalConstraintProposal,
    scalar_key,
)
from agentic_pipeline.categorical.verifier import (
    CategoricalConstraintError,
    atomic_mapping,
    json_scalar,
    typed_tuple,
)


DEFAULT_REPORT_NAME = "categorical_dependency_check_report.json"
SUPPORTED_DSL_VERSION = "2.0"
PROPOSAL_FIELDS = (
    "id",
    "description",
    "rationale",
    "determinants",
    "dependent",
    "value_table",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "constraints",
        type=Path,
        help="Path to a DSL 2.0 categorical_dependency_constraint.json.",
    )
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "CSV file or directory to evaluate. Directories are searched "
            "recursively. The default is data.csv beside the dataset's "
            "constraints_expert directory."
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


def _proposal(raw: Any, index: int) -> CategoricalConstraintProposal:
    if not isinstance(raw, dict):
        raise ValueError(f"Constraint at index {index} must be an object.")
    missing = [field for field in PROPOSAL_FIELDS if field not in raw]
    if missing:
        raise ValueError(
            f"Constraint at index {index} is missing entries: {missing}"
        )
    try:
        proposal = CategoricalConstraintProposal.model_validate(
            {field: raw[field] for field in PROPOSAL_FIELDS}
        )
        if not proposal.value_table:
            raise ValueError("value_table must not be empty")
        atomic_mapping(proposal)
        return proposal
    except (ValidationError, CategoricalConstraintError, ValueError) as exc:
        constraint_id = raw.get("id", index)
        raise ValueError(f"Invalid constraint {constraint_id!r}: {exc}") from exc


def load_constraint_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Constraint JSON not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Constraint JSON must contain an object.")
    if document.get("dsl_version") != SUPPORTED_DSL_VERSION:
        raise ValueError(
            f"Unsupported dsl_version {document.get('dsl_version')!r}; "
            f"expected {SUPPORTED_DSL_VERSION!r}. DSL 1.0 is no longer supported."
        )
    constraints = document.get("constraints")
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list.")

    graph: dict[str, set[str]] = {}
    seen_ids: set[str] = set()
    normalized = []
    for index, raw in enumerate(constraints):
        proposal = _proposal(raw, index)
        if proposal.id in seen_ids:
            raise ValueError(f"Duplicate constraint id: {proposal.id}")
        cycles = cycle_details(
            graph, proposal.determinants, proposal.dependent
        )
        if cycles:
            raise ValueError(
                f"Constraint {proposal.id!r} creates a dependency cycle: {cycles}"
            )
        add_dependency(graph, proposal.determinants, proposal.dependent)
        seen_ids.add(proposal.id)
        normalized.append({**raw, **proposal.model_dump()})
    return {"dsl_version": SUPPORTED_DSL_VERSION, "constraints": normalized}


def default_data_path(constraint_path: Path) -> Path:
    constraints_dir = constraint_path.parent
    if constraints_dir.name == "constraints_expert":
        return constraints_dir.parent / "data.csv"
    return constraints_dir / "data.csv"


def resolve_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Data file must be a CSV: {path}")
        return [path]
    if path.is_dir():
        files = sorted(
            candidate
            for candidate in path.rglob("*.csv")
            if candidate.is_file()
        )
        if not files:
            raise FileNotFoundError(f"No CSV files found under: {path}")
        return files
    raise FileNotFoundError(f"Data path not found: {path}")


def load_proposals(
    document: dict[str, Any],
) -> list[CategoricalConstraintProposal]:
    if document.get("dsl_version") != SUPPORTED_DSL_VERSION:
        raise ValueError(
            f"Categorical constraints require DSL {SUPPORTED_DSL_VERSION}."
        )
    constraints = document.get("constraints")
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list.")
    return [_proposal(raw, index) for index, raw in enumerate(constraints)]


def constraint_masks(
    data: pd.DataFrame,
    constraint: CategoricalConstraintProposal,
) -> dict[str, pd.Series]:
    columns = [*constraint.determinants, constraint.dependent]
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(
            f"Constraint {constraint.id} is missing columns {missing}."
        )

    mapping = atomic_mapping(constraint)
    applicable_values: list[bool] = []
    violation_values: list[bool] = []
    null_values: list[bool] = []
    for values in data[columns].itertuples(index=False, name=None):
        contains_null = any(pd.isna(value) for value in values)
        if contains_null:
            applicable_values.append(True)
            violation_values.append(True)
            null_values.append(True)
            continue
        try:
            key = typed_tuple(list(values[:-1]))
            dependent = scalar_key(json_scalar(values[-1]))
        except CategoricalConstraintError as exc:
            raise ValueError(
                f"Constraint {constraint.id} found an invalid categorical value: "
                f"{exc}"
            ) from exc
        allowed = mapping.get(key)
        applicable = allowed is not None
        applicable_values.append(applicable)
        violation_values.append(
            bool(applicable and dependent not in allowed)
        )
        null_values.append(False)

    applicable = pd.Series(applicable_values, index=data.index, dtype=bool)
    violation = pd.Series(violation_values, index=data.index, dtype=bool)
    null_value = pd.Series(null_values, index=data.index, dtype=bool)
    return {
        "applicable": applicable,
        "violation": violation,
        "passes": ~violation,
        "null_value": null_value,
        "dependent_mismatch": violation & ~null_value,
    }


def evaluate_dataframe_with_masks(
    data: pd.DataFrame,
    document: dict[str, Any],
) -> tuple[dict[str, Any], list[pd.Series]]:
    proposals = load_proposals(document)
    violation_masks: list[pd.Series] = []
    per_constraint: list[dict[str, Any]] = []
    rows = int(len(data))
    for proposal in proposals:
        masks = constraint_masks(data, proposal)
        violation = masks["violation"]
        applicable = int(masks["applicable"].sum())
        violations = int(violation.sum())
        violation_masks.append(violation)
        per_constraint.append(
            {
                "id": proposal.id,
                "description": proposal.description,
                "determinants": proposal.determinants,
                "dependent": proposal.dependent,
                "columns": [*proposal.determinants, proposal.dependent],
                "rows_checked": rows,
                "support_count": applicable,
                "support": float(applicable / rows) if rows else 0.0,
                "passes": rows - violations,
                "violations": violations,
                "violation_rate": (
                    float(violations / applicable) if applicable else None
                ),
                "violation_reasons": {
                    "null_value": int(masks["null_value"].sum()),
                    "dependent_mismatch": int(
                        masks["dependent_mismatch"].sum()
                    ),
                },
            }
        )

    if violation_masks:
        violation_frame = pd.concat(violation_masks, axis=1)
        any_violation = violation_frame.any(axis=1)
        total_violations = int(violation_frame.to_numpy().sum())
    else:
        any_violation = pd.Series(False, index=data.index, dtype=bool)
        total_violations = 0
    total_checks = rows * len(proposals)
    report = {
        "rows": rows,
        "per_constraint": per_constraint,
        "overall": {
            "constraints_evaluated": len(proposals),
            "rows_checked": rows,
            "rows_passing_all_constraints": int((~any_violation).sum()),
            "rows_with_any_violation": int(any_violation.sum()),
            "row_violation_rate": float(any_violation.mean()) if rows else None,
            "total_constraint_violations": total_violations,
            "total_constraint_checks": total_checks,
            "check_violation_rate": (
                float(total_violations / total_checks)
                if total_checks
                else None
            ),
        },
    }
    return report, violation_masks


def evaluate_dataframe(
    data: pd.DataFrame,
    document: dict[str, Any],
) -> dict[str, Any]:
    report, _ = evaluate_dataframe_with_masks(data, document)
    return report


def evaluate_csv(
    csv_path: Path,
    document: dict[str, Any],
) -> dict[str, Any]:
    result = evaluate_dataframe(
        pd.read_csv(csv_path, float_precision="round_trip"),
        document,
    )
    return {"csv_path": str(csv_path), **result}


def print_result(result: dict[str, Any]) -> None:
    print(f"\nCSV: {result['csv_path']}")
    print(f"Rows: {result['rows']:,}")
    print(f"{'Constraint':48} {'Violations':>12} {'Rate':>12}")
    print("-" * 74)
    for entry in result["per_constraint"]:
        rate = entry["violation_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.6%}"
        print(
            f"{entry['id'][:48]:48} "
            f"{entry['violations']:12,} {rate_text:>12}"
        )
    overall = result["overall"]
    row_rate = overall["row_violation_rate"]
    rate_text = "n/a" if row_rate is None else f"{row_rate:.6%}"
    print("-" * 74)
    print(
        "Rows violating at least one constraint: "
        f"{overall['rows_with_any_violation']:,} ({rate_text})"
    )


def resolve_output(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path / DEFAULT_REPORT_NAME


def main() -> None:
    args = parse_args()
    document = load_constraint_document(args.constraints)
    data_path = args.data or default_data_path(args.constraints)
    csv_files = resolve_csv_files(data_path)
    results = [evaluate_csv(path, document) for path in csv_files]
    report = {
        "constraints_path": str(args.constraints),
        "dsl_version": document["dsl_version"],
        "mask_semantics": "true_means_violation",
        "datasets": results,
    }
    for result in results:
        print_result(result)

    if args.output is not None:
        output_path = resolve_output(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote JSON report: {output_path}")


if __name__ == "__main__":
    main()
