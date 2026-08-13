"""Deterministically consolidate verified equational constraints.

Consolidation first keeps one verified rule per exact unordered column set.
It then solves a binary optimization problem whose retained rules must each
have at least one column unused by every other retained rule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import inf, isfinite
from typing import Any, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix, vstack


Record = dict[str, Any]


def column_signature(record: Record) -> tuple[str, ...]:
    """Return the canonical unordered column-set signature for one rule."""
    return tuple(sorted(record["constraint"]["columns"]))


def unique_columns_by_rule(records: Sequence[Record]) -> dict[str, list[str]]:
    """Return columns used by exactly one rule in ``records``."""
    counts = Counter(
        column
        for record in records
        for column in record["constraint"]["columns"]
    )
    return {
        record["constraint"]["id"]: [
            column
            for column in record["constraint"]["columns"]
            if counts[column] == 1
        ]
        for record in records
    }


def _finite_number(value: Any, default: float = inf) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _preference_key(index: int, record: Record) -> tuple[Any, ...]:
    """Prefer stronger verification, simpler rules, then acceptance order."""
    verification = record.get("verification", {})
    return (
        _finite_number(verification.get("violation_rate")),
        _finite_number(verification.get("violations")),
        len(record["constraint"]["columns"]),
        record.get("discovery_phase", inf),
        record.get("refinement_round", inf),
        index,
        record["constraint"]["id"],
    )


def deduplicate_column_sets(
    records: Sequence[Record],
) -> tuple[list[Record], list[dict[str, Any]]]:
    """Keep one preferred rule for each exact unordered column set."""
    indexed_by_signature: dict[tuple[str, ...], list[tuple[int, Record]]] = (
        defaultdict(list)
    )
    for index, record in enumerate(records):
        indexed_by_signature[column_signature(record)].append((index, record))

    kept_indices: set[int] = set()
    dropped: list[dict[str, Any]] = []
    for signature, members in indexed_by_signature.items():
        kept_index, kept_record = min(
            members, key=lambda item: _preference_key(item[0], item[1])
        )
        kept_indices.add(kept_index)
        kept_id = kept_record["constraint"]["id"]
        for index, record in members:
            if index == kept_index:
                continue
            dropped.append(
                {
                    "id": record["constraint"]["id"],
                    "reason": "duplicate_column_set",
                    "duplicate_of": kept_id,
                    "column_signature": list(signature),
                }
            )

    deduplicated = [
        record for index, record in enumerate(records) if index in kept_indices
    ]
    input_order = {
        record["constraint"]["id"]: index
        for index, record in enumerate(records)
    }
    dropped.sort(key=lambda item: input_order[item["id"]])
    return deduplicated, dropped


def _optimization_constraints(
    records: Sequence[Record],
) -> tuple[csr_matrix, np.ndarray, np.ndarray, int]:
    """Build the private-column witness constraints for scipy.milp."""
    occurrences: dict[str, list[int]] = defaultdict(list)
    witnesses: list[tuple[int, str]] = []
    for index, record in enumerate(records):
        for column in record["constraint"]["columns"]:
            occurrences[column].append(index)
            witnesses.append((index, column))

    rule_count = len(records)
    variable_count = rule_count + len(witnesses)
    row_count = (
        len(witnesses)
        + sum(len(occurrences[column]) - 1 for _, column in witnesses)
        + rule_count
    )
    matrix = lil_matrix((row_count, variable_count), dtype=float)
    lower = np.full(row_count, -np.inf)
    upper = np.zeros(row_count)
    witness_variables: dict[int, list[int]] = defaultdict(list)
    row = 0

    for offset, (rule_index, column) in enumerate(witnesses):
        witness_index = rule_count + offset
        witness_variables[rule_index].append(witness_index)

        # A column can witness only a retained rule.
        matrix[row, witness_index] = 1.0
        matrix[row, rule_index] = -1.0
        row += 1

        # A witnessed column cannot occur in another retained rule.
        for other_index in occurrences[column]:
            if other_index == rule_index:
                continue
            matrix[row, witness_index] = 1.0
            matrix[row, other_index] = 1.0
            upper[row] = 1.0
            row += 1

    # Every retained rule needs at least one private-column witness.
    for rule_index in range(rule_count):
        matrix[row, rule_index] = 1.0
        for witness_index in witness_variables[rule_index]:
            matrix[row, witness_index] = -1.0
        row += 1

    return matrix.tocsr(), lower, upper, variable_count


def _solve(
    objective: np.ndarray,
    matrix: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    bounds: Bounds,
) -> Any:
    result = milp(
        objective,
        integrality=np.ones(len(objective)),
        bounds=bounds,
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(
            "equational constraint consolidation optimization failed: "
            f"{result.message}"
        )
    return result


def maximum_unique_column_subset(
    records: Sequence[Record],
) -> tuple[list[Record], dict[str, Any]]:
    """Return an exact maximum-cardinality private-column subset.

    Among maximum-cardinality solutions, inclusion is decided
    lexicographically by verification quality, rule simplicity, acceptance
    order, and ID. Repeated feasibility solves avoid tolerance-sensitive tiny
    objective coefficients and make the selected optimum reproducible.
    """
    records = list(records)
    if not records:
        return [], {
            "solver": "scipy.optimize.milp (HiGHS)",
            "status": "not_required",
            "optimal_retained_constraints": 0,
            "tie_break": "verification quality, simplicity, acceptance order, ID",
        }

    existing_unique = unique_columns_by_rule(records)
    if all(existing_unique.values()):
        return records, {
            "solver": "scipy.optimize.milp (HiGHS)",
            "status": "not_required",
            "optimal_retained_constraints": len(records),
            "tie_break": "verification quality, simplicity, acceptance order, ID",
        }

    matrix, lower, upper, variable_count = _optimization_constraints(records)
    rule_count = len(records)
    bounds_lower = np.zeros(variable_count)
    bounds_upper = np.ones(variable_count)
    objective = np.zeros(variable_count)
    objective[:rule_count] = -1.0
    result = _solve(
        objective,
        matrix,
        lower,
        upper,
        Bounds(bounds_lower, bounds_upper),
    )
    optimum = int(round(float(np.sum(result.x[:rule_count]))))

    cardinality_row = lil_matrix((1, variable_count), dtype=float)
    cardinality_row[0, :rule_count] = 1.0
    tied_matrix = vstack([matrix, cardinality_row.tocsr()], format="csr")
    tied_lower = np.append(lower, float(optimum))
    tied_upper = np.append(upper, float(optimum))
    feasibility_objective = np.zeros(variable_count)

    preferred_indices = sorted(
        range(rule_count),
        key=lambda index: _preference_key(index, records[index]),
    )
    for index in preferred_indices:
        trial_lower = bounds_lower.copy()
        trial_upper = bounds_upper.copy()
        trial_lower[index] = 1.0
        trial_upper[index] = 1.0
        trial = milp(
            feasibility_objective,
            integrality=np.ones(variable_count),
            bounds=Bounds(trial_lower, trial_upper),
            constraints=LinearConstraint(
                tied_matrix, tied_lower, tied_upper
            ),
            options={"presolve": True},
        )
        selected = bool(trial.success)
        bounds_lower[index] = float(selected)
        bounds_upper[index] = float(selected)

    selected_records = [
        record
        for index, record in enumerate(records)
        if bounds_lower[index] == 1.0
    ]
    return selected_records, {
        "solver": "scipy.optimize.milp (HiGHS)",
        "status": "optimal",
        "optimal_retained_constraints": optimum,
        "tie_break": "verification quality, simplicity, acceptance order, ID",
    }


def consolidate_records(
    accepted_records: Sequence[Record],
) -> tuple[list[Record], dict[str, Any]]:
    """Deduplicate exact column sets, then select an exact optimal subset."""
    records = list(accepted_records)
    ids = [record["constraint"]["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("accepted equational constraint IDs must be unique")

    deduplicated, duplicate_drops = deduplicate_column_sets(records)
    retained, optimization = maximum_unique_column_subset(deduplicated)
    retained_ids = {
        record["constraint"]["id"] for record in retained
    }
    retained_by_column: dict[str, list[str]] = defaultdict(list)
    for record in retained:
        for column in record["constraint"]["columns"]:
            retained_by_column[column].append(record["constraint"]["id"])

    optimization_drops = []
    for record in deduplicated:
        constraint = record["constraint"]
        if constraint["id"] in retained_ids:
            continue
        optimization_drops.append(
            {
                "id": constraint["id"],
                "reason": "column_overlap_optimization",
                "column_conflicts": {
                    column: retained_by_column.get(column, [])
                    for column in constraint["columns"]
                },
            }
        )

    order = {constraint_id: index for index, constraint_id in enumerate(ids)}
    dropped_rules = duplicate_drops + optimization_drops
    dropped_rules.sort(key=lambda item: order[item["id"]])
    unique_columns = unique_columns_by_rule(retained)
    if any(not columns for columns in unique_columns.values()):
        raise RuntimeError(
            "internal error: optimized equational constraints violate the "
            "private-column invariant"
        )

    kept_ids = [record["constraint"]["id"] for record in retained]
    return retained, {
        "method": "exact_column_deduplication_then_binary_optimization",
        "input_constraints": len(records),
        "deduplicated_constraints": len(deduplicated),
        "published_constraints": len(retained),
        "kept_ids": kept_ids,
        "dropped_ids": [item["id"] for item in dropped_rules],
        "dropped_rules": dropped_rules,
        "unique_columns": unique_columns,
        "optimization": optimization,
    }
