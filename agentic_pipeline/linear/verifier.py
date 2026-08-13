"""Deterministic full-data verification for proposed linear inequalities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_NUMERICAL_TOLERANCE,
    DEFAULT_VIOLATION_THRESHOLD,
)
from .models import LinearConstraintProposal


class LinearConstraintError(ValueError):
    """Raised when a proposal violates the verifier contract."""


def canonical_geometry(
    constraint: LinearConstraintProposal,
) -> tuple[tuple[tuple[str, int, int], ...], tuple[int, int]]:
    """Return a positive-scale-invariant exact geometry key."""
    ordered = sorted(constraint.coefficients.items())
    scale = abs(Fraction(str(ordered[0][1])))
    direction = tuple(
        (
            column,
            (normalized := Fraction(str(value)) / scale).numerator,
            normalized.denominator,
        )
        for column, value in ordered
    )
    rhs = Fraction(str(constraint.rhs)) / scale
    return direction, (rhs.numerator, rhs.denominator)


def constraint_geometry_fingerprint(
    constraint: LinearConstraintProposal,
) -> str:
    payload = json.dumps(
        canonical_geometry(constraint),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def proposal_fingerprint(constraint: LinearConstraintProposal) -> str:
    payload = json.dumps(
        constraint.model_dump(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_constraint(
    constraint: LinearConstraintProposal,
    numerical_columns: set[str],
) -> None:
    unknown = set(constraint.coefficients) - numerical_columns
    if unknown:
        raise LinearConstraintError(
            f"unknown or non-numerical columns: {sorted(unknown)}"
        )


def _json_records(
    data: pd.DataFrame,
    positions: np.ndarray,
    columns: list[str],
    lhs: np.ndarray,
    rhs: float,
    margins: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position in positions:
        row = {
            column: data.iloc[int(position)][column] for column in columns
        }
        row.update(
            {
                "_lhs": float(lhs[position]),
                "_rhs": float(rhs),
                "_margin": float(margins[position]),
            }
        )
        records.append(json.loads(json.dumps(row, default=_json_default)))
    return records


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


@dataclass
class LinearConstraintVerifier:
    """Verify proposed inequalities against one immutable full dataframe."""

    data: pd.DataFrame
    numerical_columns: set[str]
    violation_threshold: float = DEFAULT_VIOLATION_THRESHOLD
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE
    max_counterexamples: int = 30
    sample_seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.violation_threshold <= 1:
            raise ValueError("violation_threshold must be between 0 and 1")
        if self.numerical_tolerance < 0:
            raise ValueError("numerical_tolerance cannot be negative")
        if self.max_counterexamples < 0:
            raise ValueError("max_counterexamples cannot be negative")

    def verify(self, constraint: LinearConstraintProposal) -> dict[str, Any]:
        fingerprint = proposal_fingerprint(constraint)
        geometry_fingerprint = constraint_geometry_fingerprint(constraint)
        base = {
            "id": constraint.id,
            "fingerprint": fingerprint,
            "geometry_fingerprint": geometry_fingerprint,
        }
        try:
            validate_constraint(constraint, self.numerical_columns)
        except LinearConstraintError as exc:
            return {
                **base,
                "status": "invalid_constraint",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        columns = list(constraint.coefficients)
        coefficients = np.asarray(
            [constraint.coefficients[column] for column in columns],
            dtype=float,
        )
        values = self.data[columns].to_numpy(dtype=float)
        lhs = values @ coefficients
        margins = lhs - constraint.rhs
        violation_positions = np.flatnonzero(
            margins < -self.numerical_tolerance
        )
        violations = int(violation_positions.size)
        rows = int(len(self.data))
        violation_rate = float(violations / rows) if rows else 0.0

        if violations > self.max_counterexamples:
            stable_seed = (
                int(geometry_fingerprint[:8], 16) ^ self.sample_seed
            )
            generator = np.random.default_rng(stable_seed)
            example_positions = np.sort(
                generator.choice(
                    violation_positions,
                    size=self.max_counterexamples,
                    replace=False,
                )
            )
        else:
            example_positions = violation_positions

        return {
            **base,
            "status": (
                "accepted"
                if violation_rate <= self.violation_threshold
                else "high_violation_rate"
            ),
            "rows_checked": rows,
            "passes": rows - violations,
            "violations": violations,
            "violation_rate": violation_rate,
            "threshold": self.violation_threshold,
            "numerical_tolerance": self.numerical_tolerance,
            "minimum_margin": (
                float(np.min(margins)) if margins.size else None
            ),
            "counterexamples": _json_records(
                self.data,
                example_positions,
                columns,
                lhs,
                constraint.rhs,
                margins,
            ),
        }
