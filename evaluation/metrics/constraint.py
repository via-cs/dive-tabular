"""Expert-constraint metrics for real and synthetic tabular data.

The supplied ``constraints_expert`` directory may contain categorical,
equational, and/or linear constraint JSON files.
Available families are evaluated independently; missing families are reported
as unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from constraints.linear.projector import (
    evaluate_dataframe as evaluate_linear_violations,
)
from constraints.linear.projector import project_dataframe as project_linear_dataframe
from constraints.linear.schema import LinearConstraint
from constraints.linear.schema import load_constraints as load_linear_constraints

from .categorical_constraint import (
    evaluate_dataframe_with_masks as evaluate_categorical_violations,
)
from .categorical_constraint import load_constraint_document
from .equational_constraint import (
    evaluate_dataframe_with_masks as evaluate_equational_violations,
)
from .equational_constraint import load_constraints as load_check_constraints

from .equational_consistency import evaluate_dataframe as evaluate_equational_r2
from .equational_consistency import load_constraints as load_r2_constraints


CATEGORICAL_CONSTRAINT_NAME = "categorical_dependency_constraint.json"
EQUATIONAL_CONSTRAINT_NAME = "equational_constraint.json"
LINEAR_CONSTRAINT_NAME = "linear_constraint.json"
LFD_PROJECTION_SOLVER = "CLARABEL"
LFD_PROJECTION_TOLERANCE = 1e-7


def _load_linear_constraints_if_nonempty(
    path: Path,
) -> list[LinearConstraint] | None:
    document = json.loads(path.read_text(encoding="utf-8"))
    constraint_list = (
        document.get("constraints") if isinstance(document, dict) else document
    )
    if constraint_list == []:
        return None
    return load_linear_constraints(path)


def load_constraint_bundle(constraints_expert: Path | str) -> dict[str, Any]:
    """Load whichever supported constraint families exist in a directory."""
    directory = Path(constraints_expert)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Expert constraints directory not found: {directory}"
        )

    categorical_path = directory / CATEGORICAL_CONSTRAINT_NAME
    equational_path = directory / EQUATIONAL_CONSTRAINT_NAME
    linear_path = directory / LINEAR_CONSTRAINT_NAME
    categorical_document = (
        load_constraint_document(categorical_path)
        if categorical_path.is_file()
        else None
    )
    equational_constraints = None
    if equational_path.is_file():
        check_constraints = load_check_constraints(equational_path)
        equational_constraints = load_r2_constraints(equational_path)
        check_ids = [constraint["id"] for constraint in check_constraints]
        r2_ids = [constraint["id"] for constraint in equational_constraints]
        if check_ids != r2_ids:
            raise ValueError(
                "Equational check and R2 constraint definitions are misaligned."
            )
    linear_constraints = (
        _load_linear_constraints_if_nonempty(linear_path)
        if linear_path.is_file()
        else None
    )

    if (
        categorical_document is None
        and equational_constraints is None
        and linear_constraints is None
    ):
        raise FileNotFoundError(
            f"No supported constraint JSON files found in: {directory}"
        )

    return {
        "directory": directory,
        "categorical_path": categorical_path if categorical_document else None,
        "equational_path": equational_path if equational_constraints else None,
        "linear_path": linear_path if linear_constraints else None,
        "categorical_document": categorical_document,
        "equational_constraints": equational_constraints,
        "linear_constraints": linear_constraints,
    }


def _violation_summary(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    return {
        "constraints_evaluated": overall["constraints_evaluated"],
        "rows_checked": overall["rows_checked"],
        "rows_with_any_violation": overall["rows_with_any_violation"],
        "cvr": overall["row_violation_rate"],
        "total_constraint_violations": overall[
            "total_constraint_violations"
        ],
        "total_constraint_checks": overall["total_constraint_checks"],
        "scvc": overall["check_violation_rate"],
    }


def evaluate_constraint_dataframe(
    data: pd.DataFrame,
    bundle: dict[str, Any],
    source: str = "Dataframe",
    linear_reference_data: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a compact summary and detailed constraint report for one table."""
    summary: dict[str, Any] = {
        "rows": int(len(data)),
        "average_r2_consistency": None,
        "equational_cvr": None,
        "equational_scvc": None,
        "categorical_dependency_cvr": None,
        "categorical_dependency_scvc": None,
        "linear_cvr": None,
        "linear_scvc": None,
        "linear_feasibility_distance": None,
        "family_status": {
            "equational": "not_available",
            "categorical_dependency": "not_available",
            "linear": "not_available",
        },
    }
    details: dict[str, Any] = {
        "source": source,
        "rows": int(len(data)),
        "summary": summary,
        "equational": {"status": "not_available"},
        "categorical_dependency": {"status": "not_available"},
        "linear": {"status": "not_available"},
    }

    categorical_document = bundle["categorical_document"]
    if categorical_document is not None:
        categorical_report, _ = evaluate_categorical_violations(
            data,
            categorical_document,
        )
        categorical_summary = _violation_summary(categorical_report)
        summary["categorical_dependency_cvr"] = categorical_summary["cvr"]
        summary["categorical_dependency_scvc"] = categorical_summary["scvc"]
        summary["family_status"]["categorical_dependency"] = "evaluated"
        details["categorical_dependency"] = {
            "status": "evaluated",
            "summary": categorical_summary,
            "per_constraint": categorical_report["per_constraint"],
        }

    equational_constraints = bundle["equational_constraints"]
    if equational_constraints:
        equational_report, _ = evaluate_equational_violations(
            data,
            equational_constraints,
            source,
        )
        r2_report = evaluate_equational_r2(
            data,
            equational_constraints,
            source,
        )
        equational_summary = _violation_summary(equational_report)
        average_r2 = r2_report["overall"]["average_r2_consistency"]
        summary["average_r2_consistency"] = average_r2
        summary["equational_cvr"] = equational_summary["cvr"]
        summary["equational_scvc"] = equational_summary["scvc"]
        summary["family_status"]["equational"] = "evaluated"
        details["equational"] = {
            "status": "evaluated",
            "violation_summary": equational_summary,
            "per_constraint_violations": equational_report["per_constraint"],
            "r2_summary": r2_report["overall"],
            "per_constraint_r2": r2_report["per_constraint"],
        }

    linear_constraints = bundle["linear_constraints"]
    if linear_constraints is not None:
        linear_report = evaluate_linear_violations(data, linear_constraints)
        projection = project_linear_dataframe(
            data,
            linear_constraints,
            reference_data=(
                data if linear_reference_data is None else linear_reference_data
            ),
            solver=LFD_PROJECTION_SOLVER,
            tolerance=LFD_PROJECTION_TOLERANCE,
        )
        feasibility_distance = projection.report["changes"][
            "mean_normalized_l2"
        ]
        summary["linear_cvr"] = linear_report["cvr"]
        summary["linear_scvc"] = linear_report["scvc"]
        summary["linear_feasibility_distance"] = feasibility_distance
        summary["family_status"]["linear"] = "evaluated"
        details["linear"] = {
            "status": "evaluated",
            "summary": {
                key: value
                for key, value in linear_report.items()
                if key != "per_constraint"
            },
            "feasibility_distance": {
                "value": feasibility_distance,
                "definition": (
                    "mean normalized L2 distance to the joint linear feasible "
                    "region; lower is better and zero is fully feasible"
                ),
                "normalization": "real training population standard deviation",
                "solver": LFD_PROJECTION_SOLVER,
                "projection_tolerance": LFD_PROJECTION_TOLERANCE,
                "scales": projection.report["scales"],
                "mutable_columns": projection.report["mutable_columns"],
            },
            "per_constraint": linear_report["per_constraint"],
        }

    return summary, details


def evaluate_real_constraint_dataset(
    real_train: pd.DataFrame,
    real_train_path: Path | str,
    constraints_expert: Path | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate real training data against a constraint bundle once."""
    bundle = load_constraint_bundle(constraints_expert)
    real_path = str(real_train_path)
    summary, details = evaluate_constraint_dataframe(
        real_train,
        bundle,
        real_path,
        linear_reference_data=real_train,
    )
    return {"path": real_path, **summary}, details


def evaluate_constraint_datasets(
    real_train: pd.DataFrame,
    real_train_path: Path | str,
    synthetic_datasets: list[tuple[Path | str, pd.DataFrame]],
    constraints_expert: Path | str,
    real_summary: dict[str, Any] | None = None,
    real_details: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate synthetic data, optionally reusing real-data metrics."""
    bundle = load_constraint_bundle(constraints_expert)
    if (real_summary is None) != (real_details is None):
        raise ValueError(
            "real_summary and real_details must be provided together"
        )
    if real_summary is None:
        real_path = str(real_train_path)
        summary, real_details = evaluate_constraint_dataframe(
            real_train, bundle, real_path
        )
        real_summary = {"path": real_path, **summary}

    synthetic_summaries: list[dict[str, Any]] = []
    synthetic_details: list[dict[str, Any]] = []
    for path, data in synthetic_datasets:
        path_text = str(path)
        summary, details = evaluate_constraint_dataframe(
            data,
            bundle,
            path_text,
            linear_reference_data=real_train,
        )
        synthetic_summaries.append({"path": path_text, **summary})
        synthetic_details.append(details)

    available_families = [
        family
        for family, available in (
            ("categorical_dependency", bundle["categorical_document"] is not None),
            ("equational", bundle["equational_constraints"] is not None),
            ("linear", bundle["linear_constraints"] is not None),
        )
        if available
    ]
    paths = {
        "constraints_expert": str(bundle["directory"]),
        "categorical_constraints": (
            str(bundle["categorical_path"])
            if bundle["categorical_path"] is not None
            else None
        ),
        "equational_constraints": (
            str(bundle["equational_path"])
            if bundle["equational_path"] is not None
            else None
        ),
        "linear_constraints": (
            str(bundle["linear_path"])
            if bundle["linear_path"] is not None
            else None
        ),
    }
    summary_report = {
        "available_families": available_families,
        "real_train": real_summary,
        "synthetic": synthetic_summaries,
    }
    details_report = {
        "paths": paths,
        "available_families": available_families,
        "metric_definitions": {
            "cvr": "fraction of rows violating at least one family constraint",
            "scvc": (
                "total row-constraint violations divided by rows times "
                "family constraints"
            ),
            "average_r2_consistency": (
                "unweighted mean of each equation's maximum column-direction R2"
            ),
            "linear_feasibility_distance": (
                "mean normalized L2 distance to the joint linear feasible region; "
                "normalized by real training population standard deviations"
            ),
        },
        "real_train": real_details,
        "synthetic": synthetic_details,
    }
    return summary_report, details_report
