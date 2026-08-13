"""Topologically repair acyclic categorical constraints in DSL 2.0.

For each dependent column, all applicable constraints are combined by
intersecting their allowed sets. A violating value is sampled from the
conditional distribution in reference data. If the intersection is empty, the
row is dropped.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_pipeline.categorical.graph import (
    Adjacency,
    add_dependency,
    cycle_details,
    topological_order,
)
from agentic_pipeline.categorical.models import (
    CategoricalConstraintProposal,
    scalar_key,
)
from agentic_pipeline.categorical.verifier import (
    atomic_mapping,
    json_scalar,
    typed_tuple,
)
from evaluation.metrics.categorical_constraint import (
    evaluate_dataframe,
    load_constraint_document,
    load_proposals,
    resolve_csv_files,
)


FIXED_SUFFIX = "_categorical_dependency_fixed"
REPORT_SUFFIX = "_categorical_dependency_fix_report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        type=Path,
        help="Synthetic CSV file or directory containing CSV files to repair.",
    )
    parser.add_argument(
        "constraints",
        type=Path,
        help="Path to a DSL 2.0 categorical dependency document.",
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=None,
        help=(
            "Optional reference CSV used for conditional sampling. The input "
            "CSV itself is used when omitted."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path for one input file, or output directory. For a "
            "directory input, this must be a directory."
        ),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help=(
            "Output JSON path for one input file, or report directory. For a "
            "directory input, this must be a directory."
        ),
    )
    return parser.parse_args()


def _build_graph(
    proposals: list[CategoricalConstraintProposal],
) -> tuple[Adjacency, list[str]]:
    graph: Adjacency = {}
    for proposal in proposals:
        cycles = cycle_details(
            graph, proposal.determinants, proposal.dependent
        )
        if cycles:
            raise ValueError(
                f"Constraint {proposal.id!r} creates a dependency cycle: "
                f"{cycles}"
            )
        add_dependency(graph, proposal.determinants, proposal.dependent)
    return graph, topological_order(graph)


def _value_key(value: Any) -> str:
    return scalar_key(json_scalar(value))


def _sample_allowed_value(
    *,
    allowed: set[str],
    dependent: str,
    parent_columns: list[str],
    row: pd.Series,
    reference: pd.DataFrame,
    rng: random.Random,
) -> Any:
    ordered = sorted(allowed)
    if len(ordered) == 1:
        return json.loads(ordered[0])

    matching = reference
    for column in parent_columns:
        expected = _value_key(row[column])
        matching = matching[
            matching[column].map(lambda value: _value_key(value) == expected)
        ]
    counts = Counter(
        _value_key(value)
        for value in matching[dependent].tolist()
        if _value_key(value) in allowed
    )
    if not counts:
        counts = Counter(
            _value_key(value)
            for value in reference[dependent].tolist()
            if _value_key(value) in allowed
        )
    weights = [counts.get(value, 0) for value in ordered]
    if not any(weights):
        weights = [1] * len(ordered)
    return json.loads(rng.choices(ordered, weights=weights, k=1)[0])


def fix_dataframe(
    data: pd.DataFrame,
    document: dict[str, Any],
    *,
    reference_data: pd.DataFrame | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    proposals = load_proposals(document)
    _, order = _build_graph(proposals)
    required_columns = {
        column
        for proposal in proposals
        for column in [*proposal.determinants, proposal.dependent]
    }
    missing = required_columns - set(data)
    if missing:
        raise ValueError(f"Repair data is missing columns: {sorted(missing)}")
    reference = data if reference_data is None else reference_data
    missing_reference = required_columns - set(reference)
    if missing_reference:
        raise ValueError(
            "Reference data is missing columns: "
            f"{sorted(missing_reference)}"
        )
    required = sorted(required_columns)
    if data[required].isna().any().any():
        raise ValueError("Repair data contains missing categorical values.")
    if reference[required].isna().any().any():
        raise ValueError("Reference data contains missing categorical values.")

    before = evaluate_dataframe(data, document)
    by_dependent: dict[str, list[CategoricalConstraintProposal]] = {}
    mappings: dict[str, dict[tuple[str, ...], frozenset[str]]] = {}
    for proposal in proposals:
        by_dependent.setdefault(proposal.dependent, []).append(proposal)
        mappings[proposal.id] = atomic_mapping(proposal)

    working = data.copy().reset_index(drop=True)
    dropped = [False] * len(working)
    drop_events: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for dependent in order:
        constraints = by_dependent.get(dependent, [])
        if not constraints:
            continue
        rows_repaired = 0
        rows_dropped = 0
        for position in range(len(working)):
            if dropped[position]:
                continue
            row = working.iloc[position]
            applicable: list[
                tuple[CategoricalConstraintProposal, frozenset[str]]
            ] = []
            for constraint in constraints:
                key = typed_tuple(
                    [row[column] for column in constraint.determinants]
                )
                allowed = mappings[constraint.id].get(key)
                if allowed is not None:
                    applicable.append((constraint, allowed))
            if not applicable:
                continue

            allowable = set(applicable[0][1])
            for _, allowed in applicable[1:]:
                allowable.intersection_update(allowed)
            if not allowable:
                dropped[position] = True
                rows_dropped += 1
                drop_events.append(
                    {
                        "row_position": position,
                        "dependent": dependent,
                        "constraint_ids": [
                            constraint.id for constraint, _ in applicable
                        ],
                    }
                )
                continue
            if _value_key(row[dependent]) in allowable:
                continue

            parent_columns = sorted(
                {
                    column
                    for constraint, _ in applicable
                    for column in constraint.determinants
                }
            )
            working.at[position, dependent] = _sample_allowed_value(
                allowed=allowable,
                dependent=dependent,
                parent_columns=parent_columns,
                row=row,
                reference=reference,
                rng=rng,
            )
            rows_repaired += 1

        steps.append(
            {
                "dependent": dependent,
                "constraint_ids": [
                    constraint.id for constraint in constraints
                ],
                "rows_repaired": rows_repaired,
                "rows_dropped_empty_allowable_set": rows_dropped,
            }
        )

    repaired = working.loc[
        [not value for value in dropped]
    ].reset_index(drop=True)
    after = evaluate_dataframe(repaired, document)
    remaining = after["overall"]["total_constraint_violations"]
    if remaining:
        raise RuntimeError(
            "Topological categorical repair left "
            f"{remaining} constraint violations."
        )

    return repaired, {
        "algorithm": "topological allowed-set intersection repair",
        "sampling": (
            "reference conditional distribution, then reference marginal, "
            "then uniform"
        ),
        "seed": seed,
        "topological_order": order,
        "constraint_order": [
            proposal.id for proposal in proposals
        ],
        "input_rows": len(data),
        "output_rows": len(repaired),
        "rows_dropped_empty_allowable_set": sum(dropped),
        "drop_events": drop_events,
        "before": before["overall"],
        "after": after["overall"],
        "final_constraints": after["per_constraint"],
        "steps": steps,
    }


def _single_output_path(
    source: Path,
    requested: Path | None,
    suffix: str,
    extension: str,
) -> Path:
    default = source.with_name(f"{source.stem}{suffix}{extension}")
    if requested is None:
        return default
    if requested.suffix.lower() == extension:
        return requested
    return requested / default.name


def output_paths(
    input_path: Path,
    csv_files: list[Path],
    output: Path | None,
    output_report: Path | None,
) -> list[tuple[Path, Path, Path]]:
    if input_path.is_file():
        source = csv_files[0]
        return [
            (
                source,
                _single_output_path(
                    source, output, FIXED_SUFFIX, ".csv"
                ),
                _single_output_path(
                    source, output_report, REPORT_SUFFIX, ".json"
                ),
            )
        ]

    if output is not None and output.suffix.lower() == ".csv":
        raise ValueError("--output must be a directory for directory input.")
    if output_report is not None and output_report.suffix.lower() == ".json":
        raise ValueError(
            "--output-report must be a directory for directory input."
        )
    csv_root = output or input_path.with_name(
        f"{input_path.name}{FIXED_SUFFIX}"
    )
    report_root = output_report or input_path.with_name(
        f"{input_path.name}{REPORT_SUFFIX}"
    )
    return [
        (
            source,
            csv_root / source.relative_to(input_path),
            report_root / source.relative_to(input_path).with_suffix(".json"),
        )
        for source in csv_files
    ]


def fix_csv(
    source: Path,
    csv_output: Path,
    report_output: Path,
    constraint_path: Path,
    document: dict[str, Any],
    *,
    reference_data: pd.DataFrame | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    if csv_output.resolve() == source.resolve():
        raise ValueError(f"Refusing to overwrite input CSV: {source}")
    data = pd.read_csv(source)
    repaired, report = fix_dataframe(
        data,
        document,
        reference_data=reference_data,
        seed=seed,
    )
    report.update(
        {
            "source_csv": str(source),
            "output_csv": str(csv_output),
            "output_report": str(report_output),
            "constraints_path": str(constraint_path),
            "dsl_version": document["dsl_version"],
        }
    )
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(csv_output, index=False)
    report_output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def print_report(report: dict[str, Any]) -> None:
    before = report["before"]
    after = report["after"]
    print(f"\nSource: {report['source_csv']}")
    print(f"Output: {report['output_csv']}")
    print(
        "Rows with any violation: "
        f"{before['rows_with_any_violation']:,} -> "
        f"{after['rows_with_any_violation']:,}"
    )
    print(
        "Rows dropped for empty allowed sets: "
        f"{report['rows_dropped_empty_allowable_set']:,}"
    )
    print(f"Report: {report['output_report']}")


def main() -> None:
    args = parse_args()
    document = load_constraint_document(args.constraints)
    csv_files = resolve_csv_files(args.data)
    reference = (
        pd.read_csv(args.reference_data)
        if args.reference_data is not None
        else None
    )
    destinations = output_paths(
        args.data,
        csv_files,
        args.output,
        args.output_report,
    )
    for source, csv_output, report_output in destinations:
        report = fix_csv(
            source,
            csv_output,
            report_output,
            args.constraints,
            document,
            reference_data=reference,
            seed=args.seed,
        )
        print_report(report)


if __name__ == "__main__":
    main()
