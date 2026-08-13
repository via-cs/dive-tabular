"""Euclidean projection of tabular rows onto linear feasible regions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from .schema import LinearConstraint


@dataclass
class LinearProjectionResult:
    """Projected dataframe plus JSON-serializable diagnostics."""

    data: pd.DataFrame
    report: dict[str, Any]


def _constraint_columns(constraints: Sequence[LinearConstraint]) -> list[str]:
    return list(dict.fromkeys(
        column for constraint in constraints for column in constraint.columns
    ))


def _mutable_columns(constraints: Sequence[LinearConstraint]) -> list[str]:
    return list(dict.fromkeys(
        column for constraint in constraints for column in constraint.mutable_columns
    ))


def _numeric_values(data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Data is missing constrained columns: {missing}")
    numeric = data[list(columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        raise ValueError(
            "Constrained columns must contain finite numeric values; found "
            f"{data.iloc[row][columns[column]]!r} in column {columns[column]!r} "
            f"at row position {row}."
        )
    return values


def _matrices(
    constraints: Sequence[LinearConstraint],
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    column_index = {column: index for index, column in enumerate(columns)}
    coefficients = np.zeros((len(constraints), len(columns)), dtype=float)
    for row, constraint in enumerate(constraints):
        for column, value in constraint.coefficients.items():
            coefficients[row, column_index[column]] = value
    rhs = np.asarray([constraint.rhs for constraint in constraints], dtype=float)
    return coefficients, rhs


def evaluate_dataframe(
    data: pd.DataFrame,
    constraints: Sequence[LinearConstraint],
    *,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Measure per-constraint and joint row violations."""
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative.")
    columns = _constraint_columns(constraints)
    values = _numeric_values(data, columns)
    coefficients, rhs = _matrices(constraints, columns)
    margins = values @ coefficients.T - rhs
    violations = margins < -tolerance
    any_violation = violations.any(axis=1)
    per_constraint = []
    for index, constraint in enumerate(constraints):
        count = int(violations[:, index].sum())
        per_constraint.append({
            "id": constraint.id,
            "description": constraint.description,
            "violations": count,
            "violation_rate": float(count / len(data)) if len(data) else 0.0,
            "minimum_margin": float(margins[:, index].min()) if len(data) else None,
        })
    total = int(violations.sum())
    return {
        "rows": int(len(data)),
        "constraints": len(constraints),
        "rows_with_any_violation": int(any_violation.sum()),
        "cvr": float(any_violation.mean()) if len(data) else 0.0,
        "total_constraint_violations": total,
        "scvc": float(total / violations.size) if violations.size else 0.0,
        "minimum_margin": float(margins.min()) if margins.size else None,
        "per_constraint": per_constraint,
    }


def _scales(
    data: pd.DataFrame,
    mutable_columns: Sequence[str],
    scale_mode: str,
    reference_data: pd.DataFrame | None,
) -> np.ndarray:
    if scale_mode == "none":
        return np.ones(len(mutable_columns), dtype=float)
    if scale_mode != "std":
        raise ValueError("scale_mode must be 'none' or 'std'.")
    source = data if reference_data is None else reference_data
    values = _numeric_values(source, mutable_columns)
    scales = np.std(values, axis=0, ddof=0)
    scales[~np.isfinite(scales) | (scales <= 1e-12)] = 1.0
    return scales


def _solve_batch(
    original_all: np.ndarray,
    all_coefficients: np.ndarray,
    rhs: np.ndarray,
    mutable_indices: np.ndarray,
    scales: np.ndarray,
    solver: str,
    tolerance: float,
) -> np.ndarray:
    original_mutable = original_all[:, mutable_indices]
    mutable_coefficients = all_coefficients[:, mutable_indices]
    original_margins = original_all @ all_coefficients.T - rhs
    scaled_coefficients = mutable_coefficients * scales
    constraint_scales = np.linalg.norm(scaled_coefficients, axis=1)
    constraint_scales[constraint_scales <= np.finfo(float).eps] = 1.0
    normalized_coefficients = scaled_coefficients / constraint_scales[:, None]
    normalized_original_margins = original_margins / constraint_scales

    # Solve for the standardized displacement directly. This is equivalent to
    # optimizing over values in the original units, but avoids presenting the
    # solver with objective coefficients whose magnitudes can differ by many
    # orders (for example, polarity and share-count columns in News).
    displacement = cp.Variable(original_mutable.shape)
    projected_margins = (
        normalized_original_margins
        + displacement @ normalized_coefficients.T
    )
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(displacement)),
        [projected_margins >= 0],
    )
    solver_name = solver.upper()
    options: dict[str, Any] = {"solver": solver_name, "verbose": False}
    if solver_name == "OSQP":
        options.update({
            "eps_abs": max(tolerance * 0.1, 1e-9),
            "eps_rel": max(tolerance * 0.1, 1e-9),
            "max_iter": 100_000,
            "polishing": True,
        })
    problem.solve(**options)
    if (
        problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        or displacement.value is None
    ):
        raise RuntimeError(
            f"Convex projection failed with solver status {problem.status!r}."
        )
    return original_mutable + np.asarray(displacement.value, dtype=float) * scales


def project_dataframe(
    data: pd.DataFrame,
    constraints: Sequence[LinearConstraint],
    *,
    reference_data: pd.DataFrame | None = None,
    scale_mode: str = "std",
    solver: str = "OSQP",
    batch_size: int = 1000,
    tolerance: float = 1e-7,
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
) -> LinearProjectionResult:
    """Project violating rows onto the joint linear feasible region.

    The default standardized objective minimizes ``sum(((z-x)/sigma)^2)`` to
    prevent large-unit columns from dominating. Set ``scale_mode='none'`` for
    the paper's literal ``argmin ||z-x||_2^2`` objective. Constraints always
    remain in their original units.

    With ``invalid_row_policy='drop'``, individually infeasible rows and rows
    whose projected margins still exceed ``tolerance`` are removed up to
    ``max_drop_fraction`` of the stage input.
    """
    if not constraints:
        raise ValueError("At least one linear constraint is required.")
    if invalid_row_policy not in {"error", "drop"}:
        raise ValueError(
            "invalid_row_policy must be either 'error' or 'drop'; "
            f"got {invalid_row_policy!r}."
        )
    if not 0.0 <= max_drop_fraction <= 1.0:
        raise ValueError(
            "max_drop_fraction must be between 0 and 1 inclusive; "
            f"got {max_drop_fraction}."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative.")

    started = time.perf_counter()
    input_rows = len(data)
    all_columns = _constraint_columns(constraints)
    mutable_columns = _mutable_columns(constraints)
    all_values = _numeric_values(data, all_columns)
    all_coefficients, rhs = _matrices(constraints, all_columns)
    column_index = {column: index for index, column in enumerate(all_columns)}
    mutable_indices = np.asarray(
        [column_index[column] for column in mutable_columns], dtype=int
    )
    scales = _scales(data, mutable_columns, scale_mode, reference_data)
    before = evaluate_dataframe(data, constraints, tolerance=tolerance)
    violating = (all_values @ all_coefficients.T - rhs < -tolerance).any(axis=1)

    projected = data.copy()
    projected_mutable = all_values[:, mutable_indices].copy()
    violating_positions = np.flatnonzero(violating)
    solver_failed_positions: list[int] = []

    def enforce_drop_limit(row_count: int) -> None:
        drop_fraction = row_count / input_rows if input_rows else 0.0
        if drop_fraction > max_drop_fraction:
            raise RuntimeError(
                "Linear projection would raise the row-drop fraction to "
                f"{drop_fraction:.6%}, exceeding --max-drop-fraction "
                f"{max_drop_fraction:.6%}."
            )

    def solve_positions(positions: np.ndarray) -> None:
        if not len(positions):
            return
        if not len(mutable_indices):
            if invalid_row_policy == "error":
                raise RuntimeError(
                    "Convex projection is infeasible because violating rows "
                    "have no mutable linear columns."
                )
            solver_failed_positions.extend(int(position) for position in positions)
            enforce_drop_limit(len(solver_failed_positions))
            return
        try:
            projected_mutable[positions] = _solve_batch(
                all_values[positions],
                all_coefficients,
                rhs,
                mutable_indices,
                scales,
                solver,
                tolerance,
            )
        except RuntimeError:
            if invalid_row_policy == "error":
                raise
            if len(positions) == 1:
                solver_failed_positions.append(int(positions[0]))
                enforce_drop_limit(len(solver_failed_positions))
                return
            midpoint = len(positions) // 2
            solve_positions(positions[:midpoint])
            solve_positions(positions[midpoint:])

    for start in range(0, len(violating_positions), batch_size):
        positions = violating_positions[start:start + batch_size]
        solve_positions(positions)

    for index, column in enumerate(mutable_columns):
        projected[column] = projected_mutable[:, index]
    after_projection = evaluate_dataframe(
        projected,
        constraints,
        tolerance=tolerance,
    )
    projected_values = _numeric_values(projected, all_columns)
    residual_positions = np.flatnonzero(
        (projected_values @ all_coefficients.T - rhs < -tolerance).any(axis=1)
    )
    solver_failed = np.asarray(sorted(set(solver_failed_positions)), dtype=int)
    numerical_residual_positions = np.setdiff1d(
        residual_positions,
        solver_failed,
        assume_unique=True,
    )
    invalid_positions = np.union1d(solver_failed, residual_positions)
    if len(invalid_positions) and invalid_row_policy == "error":
        raise RuntimeError(
            "Projection completed but numerical residuals still violate "
            f"constraints in {len(invalid_positions)} row(s)."
        )

    enforce_drop_limit(len(invalid_positions))
    retained = np.ones(input_rows, dtype=bool)
    retained[invalid_positions] = False
    if len(invalid_positions):
        projected = projected.iloc[retained].copy()
    after = evaluate_dataframe(projected, constraints, tolerance=tolerance)

    original_mutable = all_values[:, mutable_indices]
    delta = (projected_mutable - original_mutable)[retained]
    changed = np.abs(delta) > tolerance
    row_l2 = np.linalg.norm(delta, axis=1)
    standardized_l2 = np.linalg.norm(delta / scales, axis=1)
    report = {
        "method": "minimum_normalized_l2_convex_projection",
        "objective": (
            "sum(((z-x)/reference_std)^2)"
            if scale_mode == "std"
            else "sum((z-x)^2)"
        ),
        "solver": solver.upper(),
        "scale_mode": scale_mode,
        "tolerance": tolerance,
        "invalid_row_policy": invalid_row_policy,
        "max_drop_fraction": max_drop_fraction,
        "input_rows": input_rows,
        "output_rows": len(projected),
        "rows_dropped": int(len(invalid_positions)),
        "row_drop_fraction": (
            float(len(invalid_positions) / input_rows) if input_rows else 0.0
        ),
        "dropped_row_indices": data.index[invalid_positions].tolist(),
        "solver_failed_row_indices": data.index[solver_failed].tolist(),
        "numerical_residual_row_indices": data.index[
            numerical_residual_positions
        ].tolist(),
        "mutable_columns": mutable_columns,
        "scales": {
            column: float(scale) for column, scale in zip(mutable_columns, scales)
        },
        "before": before,
        "after_projection_before_drop": after_projection,
        "after": after,
        "changes": {
            "rows_changed": int(changed.any(axis=1).sum()),
            "cells_changed": int(changed.sum()),
            "mean_l2_original_units": float(row_l2.mean()) if len(row_l2) else 0.0,
            "max_l2_original_units": float(row_l2.max()) if len(row_l2) else 0.0,
            "mean_normalized_l2": (
                float(standardized_l2.mean()) if len(standardized_l2) else 0.0
            ),
            "mean_normalized_l2_changed_rows": (
                float(standardized_l2[changed.any(axis=1)].mean())
                if changed.any()
                else 0.0
            ),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return LinearProjectionResult(data=projected, report=report)
