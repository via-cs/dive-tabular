"""Internal equational repair used by the unified expert fixer.

The default static-global strategy scores every repair once against the
original synthetic dataframe, finds the highest KSComplement-delta threshold
that permits a complete safe repair schedule, and then maximizes total score
subject to that threshold. The legacy dynamic-greedy strategy remains
available for comparisons. Both strategies apply the selected repairs
sequentially with a single-writer and frozen-source guarantee.

The constraint JSON is trusted input: its ``check_code`` and ``fix_code``
strings are executed as Python code. Entries with ``code: null`` are skipped;
those columns are never repair targets. This module is internal; use
``constraints.expert_constraints_fix`` for repair workflows.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics.equational_constraint import load_check

from .scoring import (
    evaluate_repair,
    finite_values,
    has_executable_fix,
    ks_metrics,
    load_constraints,
    load_csv,
    load_fix,
)


DEFAULT_EQUATIONAL_STRATEGY = "static-global"
RANDOM_EQUATIONAL_STRATEGY = "random-feasible"
EQUATIONAL_STRATEGIES = (
    DEFAULT_EQUATIONAL_STRATEGY,
    "dynamic-greedy",
    RANDOM_EQUATIONAL_STRATEGY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "synthetic",
        type=Path,
        help="Synthetic CSV file or directory of CSV files to repair.",
    )
    parser.add_argument(
        "train",
        type=Path,
        help="Real training CSV used as the KSComplement reference.",
    )
    parser.add_argument(
        "constraints",
        type=Path,
        help="Equational-constraint JSON with check_code and fix_code entries.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Repaired CSV file or output directory. A file path is allowed for a "
            "single synthetic input. Defaults to "
            "<synthetic_stem>_equational_fixed.csv beside each input."
        ),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help=(
            "JSON report file or output directory. A file path is allowed for a "
            "single synthetic input. Defaults to "
            "<synthetic_stem>_equational_fix_report.json beside each input."
        ),
    )
    parser.add_argument(
        "--invalid-row-policy",
        choices=("error", "drop"),
        default="error",
        help=(
            "How to handle rows for which the selected fix returns NaN or "
            "infinity. 'error' preserves the previous fail-fast behavior; "
            "'drop' removes those rows before constraint validation."
        ),
    )
    parser.add_argument(
        "--max-drop-fraction",
        type=float,
        default=0.05,
        help=(
            "Maximum cumulative fraction of input rows that --invalid-row-policy "
            "drop may remove (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--strategy",
        "--equational-strategy",
        dest="strategy",
        choices=EQUATIONAL_STRATEGIES,
        default=DEFAULT_EQUATIONAL_STRATEGY,
        help=(
            "Equational repair planner. static-global is the default; "
            "dynamic-greedy preserves the legacy stepwise selection."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help=(
            "Seed used by random-feasible planning (default: 0). Ignored by "
            "the other strategies."
        ),
    )
    return parser.parse_args()


def default_output_path(source: Path, suffix: str) -> Path:
    return source.with_name(f"{source.stem}{suffix}")


def synthetic_csv_files(source: Path) -> list[Path]:
    """Return one input file or the CSV files immediately inside a directory."""
    if source.is_file():
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Synthetic input does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"Synthetic input is not a file or directory: {source}")

    csv_files = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )
    if not csv_files:
        raise ValueError(f"Synthetic input directory contains no CSV files: {source}")
    return csv_files


def output_path_for_source(
    source: Path,
    requested: Path | None,
    suffix: str,
    multiple_inputs: bool,
) -> Path:
    """Resolve an optional file/directory output for one synthetic source."""
    if requested is None:
        return default_output_path(source, suffix)
    if multiple_inputs or requested.is_dir():
        if requested.exists() and not requested.is_dir():
            raise ValueError(
                f"Output must be a directory when processing multiple files: {requested}"
            )
        return requested / f"{source.stem}{suffix}"
    return requested


def validate_dataframe_columns(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    constraints: list[dict[str, Any]],
) -> None:
    required_columns = {
        column for constraint in constraints for column in constraint["columns"]
    }
    missing_synthetic = sorted(required_columns - set(synthetic.columns))
    missing_train = sorted(required_columns - set(train.columns))
    if missing_synthetic:
        raise ValueError(
            f"Synthetic data is missing required columns: {missing_synthetic}"
        )
    if missing_train:
        raise ValueError(f"Training data is missing required columns: {missing_train}")


def validate_check_code(constraints: list[dict[str, Any]]) -> None:
    for constraint in constraints:
        code = constraint.get("check_code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(
                f"Constraint {constraint['id']} must contain non-empty check_code."
            )
        load_check(constraint)


def private_target_summary(
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(
        column for constraint in constraints for column in constraint["columns"]
    )
    per_constraint = []
    for constraint in constraints:
        eligible = {
            fix["column"]
            for fix in constraint["fix_code"]
            if has_executable_fix(fix)
        }
        private_targets = [
            column
            for column in constraint["columns"]
            if counts[column] == 1 and column in eligible
        ]
        per_constraint.append(
            {
                "id": constraint["id"],
                "private_eligible_targets": private_targets,
            }
        )

    return {
        "satisfied": all(entry["private_eligible_targets"] for entry in per_constraint),
        "constraints": per_constraint,
    }


def constraint_pass_mask(
    data: pd.DataFrame,
    constraint: dict[str, Any],
) -> pd.Series:
    check = load_check(constraint)
    try:
        pass_mask = check(data.copy())
    except Exception as exc:
        raise RuntimeError(
            f"Constraint check {constraint['id']} failed: {exc}"
        ) from exc

    if not isinstance(pass_mask, pd.Series):
        raise TypeError(
            f"Constraint {constraint['id']} check returned "
            f"{type(pass_mask).__name__}; expected pandas.Series."
        )
    if len(pass_mask) != len(data) or not pass_mask.index.equals(data.index):
        raise ValueError(
            f"Constraint {constraint['id']} check returned a misaligned Series."
        )
    if pass_mask.isna().any():
        raise ValueError(
            f"Constraint {constraint['id']} check returned missing values."
        )
    if not pd.api.types.is_bool_dtype(pass_mask.dtype):
        raise TypeError(
            f"Constraint {constraint['id']} check did not return Boolean values."
        )
    return pass_mask


def constraint_violations(
    data: pd.DataFrame,
    constraint: dict[str, Any],
) -> int:
    return int((~constraint_pass_mask(data, constraint)).sum())


def apply_fix(
    data: pd.DataFrame,
    constraint: dict[str, Any],
    fix_record: dict[str, str],
) -> pd.DataFrame:
    constraint_id = constraint["id"]
    column = fix_record["column"]
    fix = load_fix(constraint_id, column, fix_record["code"])
    try:
        repaired_values = fix(data.copy())
    except Exception as exc:
        raise RuntimeError(
            f"Repair {constraint_id}/{column} failed while executing: {exc}"
        ) from exc

    if not isinstance(repaired_values, pd.Series):
        raise TypeError(
            f"Repair {constraint_id}/{column} returned "
            f"{type(repaired_values).__name__}; expected pandas.Series."
        )
    if len(repaired_values) != len(data) or not repaired_values.index.equals(data.index):
        raise ValueError(
            f"Repair {constraint_id}/{column} returned a misaligned pandas Series."
        )

    repaired = data.copy()
    repaired[column] = repaired_values
    return repaired


def score_available_candidates(
    data: pd.DataFrame,
    train_values: dict[str, Any],
    constraints: list[dict[str, Any]],
    unresolved: set[int],
    forbidden_targets: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    available: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []

    for constraint_index, constraint in enumerate(constraints):
        if constraint_index not in unresolved:
            continue
        for fix_index, fix_record in enumerate(constraint["fix_code"]):
            if not has_executable_fix(fix_record):
                continue
            column = fix_record["column"]
            if column in forbidden_targets:
                blocked.append(
                    {
                        "constraint_id": constraint["id"],
                        "column": column,
                        "reason": "column was previously read or written",
                    }
                )
                continue

            metrics = evaluate_repair(
                data,
                train_values[column],
                constraint["id"],
                fix_record,
            )
            available.append(
                {
                    "constraint_index": constraint_index,
                    "fix_index": fix_index,
                    "constraint_id": constraint["id"],
                    **metrics,
                }
            )

    available.sort(
        key=lambda candidate: (
            -candidate["delta_ks_complement"],
            candidate["constraint_index"],
            candidate["fix_index"],
        )
    )
    return available, blocked


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"constraint_index", "fix_index"}
    }


def feasible_resolution_order(
    constraints: list[dict[str, Any]],
    unresolved: set[int],
    forbidden_targets: set[str],
) -> tuple[int, ...] | None:
    """Return one complete safe constraint order, or ``None`` if impossible."""

    @lru_cache(maxsize=None)
    def search(
        remaining: tuple[int, ...],
        forbidden: frozenset[str],
    ) -> tuple[int, ...] | None:
        if not remaining:
            return ()

        for constraint_index in remaining:
            constraint = constraints[constraint_index]
            eligible_targets = {
                fix_record["column"]
                for fix_record in constraint["fix_code"]
                if has_executable_fix(fix_record)
            }
            if not eligible_targets - forbidden:
                continue

            next_remaining = tuple(
                index for index in remaining if index != constraint_index
            )
            next_forbidden = forbidden | frozenset(constraint["columns"])
            continuation = search(next_remaining, next_forbidden)
            if continuation is not None:
                return (constraint_index, *continuation)
        return None

    return search(
        tuple(sorted(unresolved)),
        frozenset(forbidden_targets),
    )


def available_static_candidates(
    candidates: list[dict[str, Any]],
    unresolved: set[int],
    forbidden_targets: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Filter original-data candidate scores for one planned execution step."""
    available: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    for candidate in candidates:
        if candidate["constraint_index"] not in unresolved:
            continue
        if candidate["column"] in forbidden_targets:
            blocked.append(
                {
                    "constraint_id": candidate["constraint_id"],
                    "column": candidate["column"],
                    "reason": "column was previously read or written",
                }
            )
        else:
            available.append(candidate)
    return available, blocked


def _best_static_plan_for_threshold(
    constraints: list[dict[str, Any]],
    repairable_indices: tuple[int, ...],
    candidates: list[dict[str, Any]],
    threshold: float,
) -> tuple[float, tuple[tuple[int, int], ...]] | None:
    """Return the maximum-sum complete plan whose candidates meet threshold."""
    candidates_by_constraint: dict[int, list[dict[str, Any]]] = {
        index: [] for index in repairable_indices
    }
    for candidate in candidates:
        candidates_by_constraint[candidate["constraint_index"]].append(candidate)

    full_mask = (1 << len(repairable_indices)) - 1

    @lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[float, tuple[tuple[int, int], ...]] | None:
        if mask == full_mask:
            return 0.0, ()

        frozen: set[str] = set()
        for position, constraint_index in enumerate(repairable_indices):
            if mask & (1 << position):
                frozen.update(constraints[constraint_index]["columns"])

        best: tuple[float, tuple[tuple[int, int], ...]] | None = None
        for position, constraint_index in enumerate(repairable_indices):
            bit = 1 << position
            if mask & bit:
                continue
            for candidate in candidates_by_constraint[constraint_index]:
                if candidate["column"] in frozen:
                    continue
                score = float(candidate["delta_ks_complement"])
                if score < threshold:
                    continue
                suffix = solve(mask | bit)
                if suffix is None:
                    continue
                total = score + suffix[0]
                plan = (
                    (constraint_index, candidate["fix_index"]),
                    *suffix[1],
                )
                if best is None or total > best[0] or (
                    np.isclose(total, best[0], rtol=0.0, atol=1e-15)
                    and plan < best[1]
                ):
                    best = total, plan
        return best

    return solve(0)


def plan_static_global_repairs(
    synthetic: pd.DataFrame,
    train_values: dict[str, Any],
    constraints: list[dict[str, Any]],
    repairable_indices: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Plan a maximin-then-max-sum schedule from original-data KS scores."""
    repairable = tuple(sorted(repairable_indices))
    if not repairable:
        return [], {
            "objective": "maximize minimum predicted delta, then total predicted delta",
            "candidate_count": 0,
            "optimal_threshold": None,
            "predicted_total_delta_ks_complement": 0.0,
            "threshold_search": [],
            "candidate_scores": [],
        }, []

    candidates, _ = score_available_candidates(
        synthetic,
        train_values,
        constraints,
        set(repairable),
        set(),
    )
    candidate_lookup = {
        (candidate["constraint_index"], candidate["fix_index"]): candidate
        for candidate in candidates
    }
    thresholds = sorted(
        {float(candidate["delta_ks_complement"]) for candidate in candidates},
        reverse=True,
    )
    attempts: list[dict[str, Any]] = []
    for threshold in thresholds:
        result = _best_static_plan_for_threshold(
            constraints,
            repairable,
            candidates,
            threshold,
        )
        attempts.append({"threshold": threshold, "feasible": result is not None})
        if result is None:
            continue
        total, keys = result
        plan = [candidate_lookup[key] for key in keys]
        return plan, {
            "objective": "maximize minimum predicted delta, then total predicted delta",
            "candidate_count": len(candidates),
            "optimal_threshold": threshold,
            "predicted_min_delta_ks_complement": min(
                float(candidate["delta_ks_complement"]) for candidate in plan
            ),
            "predicted_total_delta_ks_complement": total,
            "threshold_search": attempts,
            "candidate_scores": [
                public_candidate(candidate) for candidate in candidates
            ],
        }, candidates

    unresolved_ids = [constraints[index]["id"] for index in repairable]
    raise RuntimeError(
        "No complete repair schedule exists under the single-write and "
        "frozen-source conditions, even at the lowest static KS threshold. "
        f"Repairable constraints: {unresolved_ids}."
    )



def plan_random_feasible_repairs(
    synthetic: pd.DataFrame,
    train_values: dict[str, Any],
    constraints: list[dict[str, Any]],
    repairable_indices: set[int],
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Randomly select a complete safe constraint order and repair targets.

    The depth-first search randomizes both the next constraint and its target,
    while backtracking away from orders that would leave a later constraint
    without an unfrozen executable target. Candidate KS scores are measured
    only after the random structural plan is fixed and never affect selection.
    """
    repairable = tuple(sorted(repairable_indices))
    rng = random.Random(random_seed)
    failed_states: set[tuple[tuple[int, ...], frozenset[str]]] = set()
    search_nodes = 0
    backtracks = 0

    def search(
        remaining: tuple[int, ...],
        forbidden: frozenset[str],
    ) -> tuple[tuple[int, int], ...] | None:
        nonlocal search_nodes, backtracks
        search_nodes += 1
        if not remaining:
            return ()

        state = (remaining, forbidden)
        if state in failed_states:
            return None

        randomized_constraints = list(remaining)
        rng.shuffle(randomized_constraints)
        for constraint_index in randomized_constraints:
            constraint = constraints[constraint_index]
            eligible_fix_indices = [
                fix_index
                for fix_index, fix_record in enumerate(constraint["fix_code"])
                if has_executable_fix(fix_record)
                and fix_record["column"] not in forbidden
            ]
            if not eligible_fix_indices:
                continue
            rng.shuffle(eligible_fix_indices)

            next_remaining = tuple(
                index for index in remaining if index != constraint_index
            )
            next_forbidden = forbidden | frozenset(constraint["columns"])
            continuation = search(next_remaining, next_forbidden)
            if continuation is not None:
                return (
                    (constraint_index, eligible_fix_indices[0]),
                    *continuation,
                )
            backtracks += 1

        failed_states.add(state)
        return None

    keys = search(repairable, frozenset())
    if keys is None:
        unresolved_ids = [constraints[index]["id"] for index in repairable]
        raise RuntimeError(
            "No complete randomized repair schedule exists under the "
            "single-write and frozen-source conditions. "
            f"Random seed: {random_seed}. Repairable constraints: "
            f"{unresolved_ids}."
        )

    plan: list[dict[str, Any]] = []
    for constraint_index, fix_index in keys:
        constraint = constraints[constraint_index]
        fix_record = constraint["fix_code"][fix_index]
        metrics = evaluate_repair(
            synthetic,
            train_values[fix_record["column"]],
            constraint["id"],
            fix_record,
        )
        plan.append(
            {
                "constraint_index": constraint_index,
                "fix_index": fix_index,
                "constraint_id": constraint["id"],
                **metrics,
            }
        )

    planning_report = {
        "objective": (
            "random complete schedule with random available repair targets; "
            "KSComplement scores do not affect selection"
        ),
        "random_seed": random_seed,
        "candidate_count": len(plan),
        "search_nodes": search_nodes,
        "backtracks": backtracks,
        "failed_states": len(failed_states),
        "predicted_min_delta_ks_complement": (
            min(float(candidate["delta_ks_complement"]) for candidate in plan)
            if plan
            else None
        ),
        "predicted_total_delta_ks_complement": sum(
            float(candidate["delta_ks_complement"]) for candidate in plan
        ),
        "candidate_scores": [public_candidate(candidate) for candidate in plan],
    }
    return plan, planning_report, plan


def applied_repair_metrics(
    before: pd.DataFrame,
    after: pd.DataFrame,
    real_values: np.ndarray,
    column: str,
) -> dict[str, Any]:
    """Measure the observed target-column KS change for an applied repair."""
    before_values = finite_values(before[column], f"column {column!r} before repair")
    after_values = finite_values(after[column], f"column {column!r} after repair")
    before_statistic, before_complement = ks_metrics(real_values, before_values)
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


def validate_resolved_constraints(
    data: pd.DataFrame,
    constraints: list[dict[str, Any]],
    resolved: list[int],
) -> dict[str, int]:
    violations = {
        constraints[index]["id"]: constraint_violations(data, constraints[index])
        for index in resolved
    }
    failed = {
        constraint_id: count
        for constraint_id, count in violations.items()
        if count
    }
    if failed:
        raise RuntimeError(
            "A selected repair failed to preserve resolved constraints; "
            f"violation counts: {failed}"
        )
    return violations


def _equational_fix(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    constraints: list[dict[str, Any]],
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
    strategy: str = DEFAULT_EQUATIONAL_STRATEGY,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    if strategy not in EQUATIONAL_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {EQUATIONAL_STRATEGIES}; got {strategy!r}."
        )

    validate_dataframe_columns(synthetic, train, constraints)
    validate_check_code(constraints)

    train_values = {
        column: finite_values(train[column], f"training column {column!r}")
        for column in {
            column for constraint in constraints for column in constraint["columns"]
        }
    }
    working = synthetic.copy()
    skipped_unfixable = [
        constraints[index]["id"]
        for index, constraint in enumerate(constraints)
        if not any(has_executable_fix(fix) for fix in constraint["fix_code"])
    ]
    unresolved = {
        index
        for index, constraint in enumerate(constraints)
        if any(has_executable_fix(fix) for fix in constraint["fix_code"])
    }
    resolved: list[int] = []
    forbidden_targets: set[str] = set()
    steps: list[dict[str, Any]] = []
    input_rows = len(synthetic)
    total_rows_dropped = 0
    planned_repairs: list[dict[str, Any]] = []
    planning_report: dict[str, Any] | None = None
    planned_candidates: list[dict[str, Any]] = []
    if strategy == DEFAULT_EQUATIONAL_STRATEGY:
        (
            planned_repairs,
            planning_report,
            planned_candidates,
        ) = plan_static_global_repairs(
            synthetic,
            train_values,
            constraints,
            unresolved,
        )
    elif strategy == RANDOM_EQUATIONAL_STRATEGY:
        (
            planned_repairs,
            planning_report,
            planned_candidates,
        ) = plan_random_feasible_repairs(
            synthetic,
            train_values,
            constraints,
            unresolved,
            random_seed,
        )

    repairable_count = len(unresolved)
    while unresolved:
        if strategy in {DEFAULT_EQUATIONAL_STRATEGY, RANDOM_EQUATIONAL_STRATEGY}:
            available, blocked = available_static_candidates(
                planned_candidates,
                unresolved,
                forbidden_targets,
            )
            selected = planned_repairs[len(steps)]
            dead_end_candidates: list[dict[str, Any]] = []
            selected_continuation = tuple(
                candidate["constraint_index"]
                for candidate in planned_repairs[len(steps) + 1 :]
            )
        else:
            available, blocked = score_available_candidates(
                working,
                train_values,
                constraints,
                unresolved,
                forbidden_targets,
            )
            if not available:
                unresolved_ids = [constraints[index]["id"] for index in sorted(unresolved)]
                raise RuntimeError(
                    "No feasible repair target remains under the single-write and "
                    "frozen-source conditions. "
                    f"Unresolved constraints: {unresolved_ids}. "
                    f"Forbidden targets: {sorted(forbidden_targets)}."
                )

            selected = None
            dead_end_candidates: list[dict[str, Any]] = []
            selected_continuation: tuple[int, ...] = ()
            for candidate in available:
                candidate_index = candidate["constraint_index"]
                remaining = unresolved - {candidate_index}
                next_forbidden = forbidden_targets | set(
                    constraints[candidate_index]["columns"]
                )
                continuation = feasible_resolution_order(
                    constraints,
                    remaining,
                    next_forbidden,
                )
                if continuation is not None:
                    selected = candidate
                    selected_continuation = continuation
                    break
                dead_end_candidates.append(public_candidate(candidate))

            if selected is None:
                unresolved_ids = [constraints[index]["id"] for index in sorted(unresolved)]
                raise RuntimeError(
                    "No candidate permits a complete repair schedule under the "
                    "single-write and frozen-source conditions. "
                    f"Unresolved constraints: {unresolved_ids}. "
                    f"Forbidden targets: {sorted(forbidden_targets)}."
                )

        constraint_index = selected["constraint_index"]
        fix_index = selected["fix_index"]
        constraint = constraints[constraint_index]
        fix_record = constraint["fix_code"][fix_index]
        forbidden_before = sorted(forbidden_targets)

        rows_before = len(working)
        before_repair = working.copy()
        repaired = apply_fix(working, constraint, fix_record)
        repaired_numeric = pd.to_numeric(
            repaired[selected["column"]], errors="coerce"
        ).to_numpy(dtype=float)
        nonfinite_mask = ~np.isfinite(repaired_numeric)
        nonfinite_row_indices = repaired.index[nonfinite_mask].tolist()

        finite_repaired = repaired.loc[~nonfinite_mask]
        selected_pass_mask = constraint_pass_mask(finite_repaired, constraint)
        unsatisfied_row_indices = selected_pass_mask.index[
            ~selected_pass_mask
        ].tolist()
        invalid_mask = pd.Series(nonfinite_mask, index=repaired.index)
        invalid_mask.loc[unsatisfied_row_indices] = True
        dropped_row_indices = repaired.index[invalid_mask].tolist()
        rows_dropped = len(dropped_row_indices)

        if nonfinite_row_indices and invalid_row_policy == "error":
            examples = nonfinite_row_indices[:10]
            raise RuntimeError(
                f"Repair {constraint['id']}/{selected['column']} produced "
                f"{len(nonfinite_row_indices):,} non-finite values. "
                "Example row indices: "
                f"{examples}. Re-run with --invalid-row-policy drop to remove "
                "mathematically unsatisfiable rows."
            )
        if unsatisfied_row_indices and invalid_row_policy == "error":
            examples = unsatisfied_row_indices[:10]
            raise RuntimeError(
                f"Repair {constraint['id']}/{selected['column']} left "
                f"{len(unsatisfied_row_indices):,} rows violating its own "
                f"constraint. Example row indices: {examples}. Re-run with "
                "--invalid-row-policy drop to remove mathematically "
                "unsatisfiable rows."
            )

        prospective_total_dropped = total_rows_dropped + rows_dropped
        prospective_drop_fraction = (
            prospective_total_dropped / input_rows if input_rows else 0.0
        )
        if (
            rows_dropped
            and invalid_row_policy == "drop"
            and prospective_drop_fraction > max_drop_fraction
        ):
            raise RuntimeError(
                f"Repair {constraint['id']}/{selected['column']} would raise "
                f"the cumulative row-drop fraction to "
                f"{prospective_drop_fraction:.6%}, exceeding "
                f"--max-drop-fraction {max_drop_fraction:.6%}."
            )

        if rows_dropped:
            working = repaired.loc[~invalid_mask.to_numpy()].copy()
            total_rows_dropped = prospective_total_dropped
            if working.empty:
                raise RuntimeError(
                    f"Repair {constraint['id']}/{selected['column']} removed "
                    "every synthetic row."
                )
        else:
            working = repaired

        actual_metrics = applied_repair_metrics(
            before_repair,
            working,
            train_values[selected["column"]],
            selected["column"],
        )
        prediction_error = (
            actual_metrics["delta_ks_complement"]
            - float(selected["delta_ks_complement"])
        )

        unresolved.remove(constraint_index)
        resolved.append(constraint_index)
        forbidden_targets.update(constraint["columns"])
        violation_counts = validate_resolved_constraints(
            working,
            constraints,
            resolved,
        )

        steps.append(
            {
                "step": len(steps) + 1,
                "selected": public_candidate(selected),
                "predicted_static": (
                    public_candidate(selected)
                    if strategy == DEFAULT_EQUATIONAL_STRATEGY
                    else None
                ),
                "actual_execution": actual_metrics,
                "prediction_error": prediction_error,
                "constraint_columns": constraint["columns"],
                "input_columns": [
                    column
                    for column in constraint["columns"]
                    if column != selected["column"]
                ],
                "forbidden_targets_before": forbidden_before,
                "forbidden_targets_after": sorted(forbidden_targets),
                "resolved_constraint_violations": violation_counts,
                "rows_before": rows_before,
                "rows_after": len(working),
                "rows_dropped": rows_dropped,
                "dropped_row_indices": dropped_row_indices,
                "nonfinite_row_indices": nonfinite_row_indices,
                "unsatisfied_row_indices": unsatisfied_row_indices,
                "available_candidates": [
                    public_candidate(candidate) for candidate in available
                ],
                "dead_end_candidates": dead_end_candidates,
                "feasible_continuation": [
                    constraints[index]["id"] for index in selected_continuation
                ],
                "blocked_candidates": blocked,
            }
        )
        drop_message = (
            f", dropped {rows_dropped:,} invalid rows" if rows_dropped else ""
        )
        print(
            f"Step {len(steps)}/{repairable_count}: {constraint['id']} -> "
            f"{selected['column']} "
            f"(predicted delta KSComplement="
            f"{selected['delta_ks_complement']:.12g}, actual="
            f"{actual_metrics['delta_ks_complement']:.12g}"
            f"{drop_message})"
        )

    final_violations = {
        constraint["id"]: constraint_violations(working, constraint)
        for constraint in constraints
        if any(has_executable_fix(fix) for fix in constraint["fix_code"])
    }
    if any(final_violations.values()):
        raise RuntimeError(
            f"Final constraint validation failed: {final_violations}"
        )

    constrained_columns = sorted(train_values)
    final_column_ks = {
        column: applied_repair_metrics(
            synthetic,
            working,
            train_values[column],
            column,
        )
        for column in constrained_columns
    }
    actual_step_deltas = [
        float(step["actual_execution"]["delta_ks_complement"])
        for step in steps
    ]
    report = {
        "algorithm": {
            DEFAULT_EQUATIONAL_STRATEGY: (
                "static global maximin-then-max-sum repair with frozen dependencies"
            ),
            "dynamic-greedy": (
                "safe greedy non-overwriting repair with frozen dependencies"
            ),
            RANDOM_EQUATIONAL_STRATEGY: (
                "random complete repair order and targets with frozen dependencies"
            ),
        }[strategy],
        "strategy": strategy,
        "random_seed": (
            random_seed if strategy == RANDOM_EQUATIONAL_STRATEGY else None
        ),
        "score": "SDMetrics KSComplement after repair minus before repair",
        "planning": planning_report,
        "repair_scope": "entire target column",
        "invalid_row_policy": invalid_row_policy,
        "max_drop_fraction": max_drop_fraction,
        "input_rows": input_rows,
        "output_rows": len(working),
        "rows_dropped": total_rows_dropped,
        "row_drop_fraction": (
            total_rows_dropped / input_rows if input_rows else 0.0
        ),
        "constraints_resolved": len(resolved),
        "constraints_skipped_unfixable": skipped_unfixable,
        "private_target_assumption": private_target_summary(constraints),
        "selected_order": [
            {
                "step": step["step"],
                "constraint_id": step["selected"]["constraint_id"],
                "target_column": step["selected"]["column"],
                "predicted_delta_ks_complement": step["selected"][
                    "delta_ks_complement"
                ],
                "actual_delta_ks_complement": step["actual_execution"][
                    "delta_ks_complement"
                ],
            }
            for step in steps
        ],
        "final_constraint_violations": final_violations,
        "actual_min_step_delta_ks_complement": (
            min(actual_step_deltas) if actual_step_deltas else None
        ),
        "actual_total_step_delta_ks_complement": sum(actual_step_deltas),
        "final_column_ks": final_column_ks,
        "steps": steps,
    }
    return working, report


def equational_fix(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    constraints: list[dict[str, Any]],
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
    strategy: str = DEFAULT_EQUATIONAL_STRATEGY,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _equational_fix(
        synthetic,
        train,
        constraints,
        invalid_row_policy=invalid_row_policy,
        max_drop_fraction=max_drop_fraction,
        strategy=strategy,
        random_seed=random_seed,
    )


def static_global_equational_fix(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    constraints: list[dict[str, Any]],
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return equational_fix(
        synthetic,
        train,
        constraints,
        invalid_row_policy=invalid_row_policy,
        max_drop_fraction=max_drop_fraction,
        strategy=DEFAULT_EQUATIONAL_STRATEGY,
    )


def greedy_equational_fix(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    constraints: list[dict[str, Any]],
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return equational_fix(
        synthetic,
        train,
        constraints,
        invalid_row_policy=invalid_row_policy,
        max_drop_fraction=max_drop_fraction,
        strategy="dynamic-greedy",
    )


def random_equational_fix(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    constraints: list[dict[str, Any]],
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return equational_fix(
        synthetic,
        train,
        constraints,
        invalid_row_policy=invalid_row_policy,
        max_drop_fraction=max_drop_fraction,
        strategy=RANDOM_EQUATIONAL_STRATEGY,
        random_seed=random_seed,
    )


def main() -> None:
    args = parse_args()
    train = load_csv(args.train, "Training")
    constraints = load_constraints(args.constraints)
    synthetic_files = synthetic_csv_files(args.synthetic)
    multiple_inputs = len(synthetic_files) > 1 or args.synthetic.is_dir()

    for file_index, synthetic_file in enumerate(synthetic_files, start=1):
        print(f"Processing {file_index}/{len(synthetic_files)}: {synthetic_file}")
        synthetic = load_csv(synthetic_file, "Synthetic")
        output_csv = output_path_for_source(
            synthetic_file,
            args.output_csv,
            "_equational_fixed.csv",
            multiple_inputs,
        )
        output_report = output_path_for_source(
            synthetic_file,
            args.output_report,
            "_equational_fix_report.json",
            multiple_inputs,
        )

        repaired, report = equational_fix(
            synthetic,
            train,
            constraints,
            invalid_row_policy=args.invalid_row_policy,
            max_drop_fraction=args.max_drop_fraction,
            strategy=args.strategy,
            random_seed=args.random_seed,
        )
        report.update(
            {
                "synthetic_file": str(synthetic_file),
                "train_file": str(args.train),
                "constraints_file": str(args.constraints),
                "output_csv": str(output_csv),
                "output_report": str(output_report),
                "synthetic_rows": int(len(synthetic)),
                "output_rows": int(len(repaired)),
                "train_rows": int(len(train)),
            }
        )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_report.parent.mkdir(parents=True, exist_ok=True)
        repaired.to_csv(output_csv, index=False)
        output_report.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote repaired CSV: {output_csv}")
        print(f"Wrote repair report: {output_report}")


if __name__ == "__main__":
    main()
