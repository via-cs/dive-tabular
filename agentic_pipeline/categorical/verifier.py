"""Full-data verification for unified categorical dependency constraints."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_MAX_ATOMIC_CONFIGURATIONS,
    DEFAULT_MAX_COUNTEREXAMPLES,
    DEFAULT_MAX_DETERMINANTS,
    DEFAULT_VIOLATION_THRESHOLD,
)
from .models import CategoricalConstraintProposal, JsonScalar, scalar_key


class CategoricalConstraintError(ValueError):
    """Raised when a proposed constraint violates the verifier contract."""


def json_scalar(value: Any) -> JsonScalar:
    """Convert a dataframe scalar into its JSON representation."""
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (str, int, float, bool)):
        raise CategoricalConstraintError(
            f"value {value!r} is not a supported categorical JSON scalar"
        )
    if isinstance(value, float) and not np.isfinite(value):
        raise CategoricalConstraintError("categorical values must be finite")
    return value


def typed_tuple(values: tuple[Any, ...] | list[Any]) -> tuple[str, ...]:
    return tuple(scalar_key(json_scalar(value)) for value in values)


def atomic_mapping(
    constraint: CategoricalConstraintProposal,
    *,
    max_configurations: int = DEFAULT_MAX_ATOMIC_CONFIGURATIONS,
) -> dict[tuple[str, ...], frozenset[str]]:
    """Expand set-valued rows into a scalar-safe exact-tuple mapping."""
    mapping: dict[tuple[str, ...], frozenset[str]] = {}
    expanded = 0
    for row_index, row in enumerate(constraint.value_table):
        dependent = frozenset(scalar_key(value) for value in row.dependent_values)
        for values in itertools.product(*row.determinant_values):
            expanded += 1
            if expanded > max_configurations:
                raise CategoricalConstraintError(
                    "value table expands beyond the maximum of "
                    f"{max_configurations:,} atomic determinant configurations"
                )
            key = typed_tuple(values)
            previous = mapping.get(key)
            if previous is not None and previous != dependent:
                raise CategoricalConstraintError(
                    "overlapping value-table rows assign conflicting admissible "
                    f"dependent values at row {row_index}"
                )
            mapping[key] = dependent
    return mapping


def constraint_signature(
    constraint: CategoricalConstraintProposal,
) -> tuple[tuple[str, ...], str]:
    return tuple(sorted(constraint.determinants)), constraint.dependent


def canonical_payload(constraint: CategoricalConstraintProposal) -> dict[str, Any]:
    """Return an order-independent semantic representation."""
    order = sorted(
        range(len(constraint.determinants)),
        key=lambda index: constraint.determinants[index],
    )
    mapping = atomic_mapping(constraint)
    canonical_rows = []
    for key, allowed in mapping.items():
        reordered = [key[index] for index in order]
        canonical_rows.append([reordered, sorted(allowed)])
    canonical_rows.sort(key=lambda item: (item[0], item[1]))
    return {
        "determinants": sorted(constraint.determinants),
        "dependent": constraint.dependent,
        "mapping": canonical_rows,
    }


def constraint_fingerprint(constraint: CategoricalConstraintProposal) -> str:
    payload = json.dumps(
        canonical_payload(constraint),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_record(row: pd.Series, columns: list[str]) -> dict[str, JsonScalar]:
    return {column: json_scalar(row[column]) for column in columns}


@dataclass
class CategoricalConstraintVerifier:
    """Verify a value-table constraint against one immutable dataframe."""

    data: pd.DataFrame
    categorical_columns: set[str]
    violation_threshold: float = DEFAULT_VIOLATION_THRESHOLD
    max_determinants: int = DEFAULT_MAX_DETERMINANTS
    max_counterexamples: int = DEFAULT_MAX_COUNTEREXAMPLES
    max_atomic_configurations: int = DEFAULT_MAX_ATOMIC_CONFIGURATIONS

    def __post_init__(self) -> None:
        if not 0 <= self.violation_threshold <= 1:
            raise ValueError("violation_threshold must be between 0 and 1")
        if self.max_determinants < 1:
            raise ValueError("max_determinants must be positive")
        if self.max_counterexamples < 0:
            raise ValueError("max_counterexamples cannot be negative")
        if self.max_atomic_configurations < 1:
            raise ValueError("max_atomic_configurations must be positive")

    def _validate(self, constraint: CategoricalConstraintProposal) -> None:
        if not constraint.value_table:
            raise CategoricalConstraintError("value_table must not be empty")
        columns = set(constraint.determinants) | {constraint.dependent}
        unknown = columns - self.categorical_columns
        if unknown:
            raise CategoricalConstraintError(
                f"unknown or non-categorical columns: {sorted(unknown)}"
            )
        if len(constraint.determinants) > self.max_determinants:
            raise CategoricalConstraintError(
                f"at most {self.max_determinants} determinants are allowed"
            )

        domains = {
            column: {
                scalar_key(json_scalar(value))
                for value in self.data[column].drop_duplicates().tolist()
            }
            for column in columns
        }
        for row_index, row in enumerate(constraint.value_table):
            for index, values in enumerate(row.determinant_values):
                column = constraint.determinants[index]
                invalid = [
                    value
                    for value in values
                    if scalar_key(value) not in domains[column]
                ]
                if invalid:
                    raise CategoricalConstraintError(
                        f"value_table[{row_index}] contains unknown values for "
                        f"{column!r}: {invalid}"
                    )
            invalid_dependent = [
                value
                for value in row.dependent_values
                if scalar_key(value) not in domains[constraint.dependent]
            ]
            if invalid_dependent:
                raise CategoricalConstraintError(
                    f"value_table[{row_index}] contains unknown dependent values: "
                    f"{invalid_dependent}"
                )

    def verify(self, constraint: CategoricalConstraintProposal) -> dict[str, Any]:
        fingerprint: str | None = None
        try:
            self._validate(constraint)
            mapping = atomic_mapping(
                constraint,
                max_configurations=self.max_atomic_configurations,
            )
            fingerprint = constraint_fingerprint(constraint)
        except (CategoricalConstraintError, ValueError) as exc:
            return {
                "id": constraint.id,
                "status": "invalid_constraint",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        determinant_values = [
            typed_tuple(list(values))
            for values in self.data[constraint.determinants].itertuples(
                index=False, name=None
            )
        ]
        dependent_values = [
            scalar_key(json_scalar(value))
            for value in self.data[constraint.dependent].tolist()
        ]
        applicable_positions: list[int] = []
        violation_positions: list[int] = []
        for position, (key, dependent) in enumerate(
            zip(determinant_values, dependent_values, strict=True)
        ):
            allowed = mapping.get(key)
            if allowed is None:
                continue
            applicable_positions.append(position)
            if dependent not in allowed:
                violation_positions.append(position)

        rows = len(self.data)
        applicable = len(applicable_positions)
        violations = len(violation_positions)
        support = float(applicable / rows) if rows else 0.0
        violation_rate = float(violations / applicable) if applicable else None
        accepted = (
            applicable > 0
            and violation_rate is not None
            and violation_rate < self.violation_threshold
        )
        example_positions = violation_positions[: self.max_counterexamples]
        columns = [*constraint.determinants, constraint.dependent]
        return {
            "id": constraint.id,
            "fingerprint": fingerprint,
            "status": (
                "accepted"
                if accepted
                else "zero_support"
                if applicable == 0
                else "high_violation_rate"
            ),
            "rows_checked": rows,
            "support_count": applicable,
            "support": support,
            "violations": violations,
            "violation_rate": violation_rate,
            "threshold": self.violation_threshold,
            "atomic_configurations": len(mapping),
            "counterexamples": [
                _json_record(self.data.iloc[position], columns)
                for position in example_positions
            ],
        }
