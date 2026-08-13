"""Model-selectable tools and host state for categorical discovery."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from .config import DEFAULT_MAX_CONFIGURATION_SUMMARIES
from .graph import (
    Adjacency,
    add_dependency,
    direction_result,
    ensure_nodes,
    topological_order,
)
from .materialization import materialize_majority_value_table
from .models import (
    AnalyzeDependencyArguments,
    CategoricalConstraintProposal,
    DependentFrequenciesArguments,
    InspectColumnsArguments,
    SubmitConstraintArguments,
    scalar_key,
)
from .verifier import (
    CategoricalConstraintVerifier,
    constraint_fingerprint,
    constraint_signature,
    json_scalar,
    typed_tuple,
)


INSPECT_TOOL_NAME = "inspect_categorical_columns"
ANALYZE_TOOL_NAME = "analyze_dependency"
FREQUENCIES_TOOL_NAME = "get_dependent_frequencies"
SUBMIT_TOOL_NAME = "submit_categorical_constraint"


@dataclass
class EvidenceRecord:
    id: str
    kind: str
    signature: tuple[tuple[str, ...], str] | None
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class CategoricalToolContext:
    data: pd.DataFrame
    descriptions: dict[str, str]
    verifier: CategoricalConstraintVerifier
    max_configuration_summaries: int = DEFAULT_MAX_CONFIGURATION_SUMMARIES
    max_constraints: int = 30
    dependency_graph: Adjacency = field(default_factory=dict)
    evidence: dict[str, EvidenceRecord] = field(default_factory=dict)
    accepted: dict[str, dict[str, Any]] = field(default_factory=dict)
    _evidence_counter: int = 0

    def __post_init__(self) -> None:
        ensure_nodes(self.dependency_graph, self.descriptions)

    @property
    def categorical_columns(self) -> set[str]:
        return set(self.descriptions)

    def add_evidence(
        self,
        kind: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        signature: tuple[tuple[str, ...], str] | None = None,
    ) -> str:
        self._evidence_counter += 1
        evidence_id = f"evidence_{self._evidence_counter:04d}"
        self.evidence[evidence_id] = EvidenceRecord(
            id=evidence_id,
            kind=kind,
            signature=signature,
            arguments=arguments,
            result=result,
        )
        return evidence_id


def _tool_schema(
    name: str,
    description: str,
    arguments: type[BaseModel],
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": arguments.model_json_schema(),
        "strict": False,
    }


def tool_schemas() -> list[dict[str, Any]]:
    return [
        _tool_schema(
            INSPECT_TOOL_NAME,
            "Inspect metadata, cardinality, frequencies, representative values, "
            "and key-like warnings for selected categorical columns. An empty "
            "column list inspects every available categorical column.",
            InspectColumnsArguments,
        ),
        _tool_schema(
            ANALYZE_TOOL_NAME,
            "Analyze whether the chosen categorical determinant columns constrain "
            "one categorical dependent column on the complete table. Returns "
            "majority violation rate, entropy, proper-subset comparisons, and "
            "compact configuration summaries. Directions that would create a dependency "
            "cycle are rejected without evidence. Call this before submitting a "
            "constraint with the same signature.",
            AnalyzeDependencyArguments,
        ),
        _tool_schema(
            FREQUENCIES_TOOL_NAME,
            "Return dependent-value frequencies for one Cartesian, set-valued "
            "determinant configuration on the complete table.",
            DependentFrequenciesArguments,
        ),
        _tool_schema(
            SUBMIT_TOOL_NAME,
            "Validate and evaluate one evidence-backed categorical dependency "
            "constraint. Accepted constraints are stored by the host. A prior "
            "analyze_dependency evidence ID for the identical determinant and "
            "dependent signature is required. Verified directions that would create a "
            "dependency cycle are rejected.",
            SubmitConstraintArguments,
        ),
    ]


def _validate_signature(
    determinants: list[str], dependent: str, context: CategoricalToolContext
) -> tuple[tuple[str, ...], str]:
    if len(determinants) != len(set(determinants)):
        raise ValueError("determinants must be unique")
    if dependent in determinants:
        raise ValueError("dependent cannot also be a determinant")
    if len(determinants) > context.verifier.max_determinants:
        raise ValueError(
            f"at most {context.verifier.max_determinants} determinants are allowed"
        )
    unknown = (set(determinants) | {dependent}) - context.categorical_columns
    if unknown:
        raise ValueError(f"unknown or non-categorical columns: {sorted(unknown)}")
    return tuple(sorted(determinants)), dependent


def _frequency_records(values: dict[str, tuple[Any, int]]) -> list[dict[str, Any]]:
    total = sum(count for _, count in values.values())
    ordered = sorted(
        values.values(),
        key=lambda item: (-item[1], scalar_key(json_scalar(item[0]))),
    )
    return [
        {
            "value": json_scalar(value),
            "count": count,
            "frequency": float(count / total) if total else 0.0,
        }
        for value, count in ordered
    ]


def _dependency_stats(
    data: pd.DataFrame,
    determinants: list[str],
    dependent: str,
    *,
    include_configurations: bool,
    max_configurations: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], dict[str, tuple[Any, int]]] = {}
    raw_keys: dict[tuple[str, ...], tuple[Any, ...]] = {}
    columns = [*determinants, dependent]
    for row in data[columns].itertuples(index=False, name=None):
        determinant_raw = tuple(json_scalar(value) for value in row[:-1])
        key = typed_tuple(list(determinant_raw))
        raw_keys.setdefault(key, determinant_raw)
        dependent_value = json_scalar(row[-1])
        dependent_key = scalar_key(dependent_value)
        counts = groups.setdefault(key, {})
        previous = counts.get(dependent_key)
        counts[dependent_key] = (
            dependent_value,
            1 if previous is None else previous[1] + 1,
        )

    rows = len(data)
    majority_violations = 0
    conflict_count = 0
    entropy_weighted = 0.0
    summaries: list[dict[str, Any]] = []
    for key, counts in groups.items():
        group_size = sum(count for _, count in counts.values())
        majority = max(count for _, count in counts.values())
        majority_violations += group_size - majority
        if len(counts) > 1:
            conflict_count += 1
        entropy = -sum(
            (count / group_size) * math.log2(count / group_size)
            for _, count in counts.values()
        )
        entropy_weighted += group_size * entropy
        if include_configurations:
            summaries.append(
                {
                    "determinant_values": [
                        [json_scalar(value)] for value in raw_keys[key]
                    ],
                    "support_count": group_size,
                    "support": float(group_size / rows),
                    "dependent_frequencies": _frequency_records(counts),
                    "conflicting": len(counts) > 1,
                }
            )

    summaries.sort(
        key=lambda item: (
            not item["conflicting"],
            -item["support_count"],
            json.dumps(item["determinant_values"], ensure_ascii=False),
        )
    )
    result = {
        "rows": rows,
        "support": 1.0,
        "support_count": rows,
        "determinant_configurations": len(groups),
        "conflicting_configurations": conflict_count,
        "majority_violations": majority_violations,
        "majority_violation_rate": (
            float(majority_violations / rows) if rows else 0.0
        ),
        "conditional_entropy": float(entropy_weighted / rows) if rows else 0.0,
        "key_like_warning": bool(groups and len(groups) / rows >= 0.95),
    }
    if include_configurations:
        result["configuration_summaries"] = summaries[:max_configurations]
        result["configuration_summaries_truncated"] = (
            len(summaries) > max_configurations
        )
    return result


def inspect_columns(
    arguments: InspectColumnsArguments, context: CategoricalToolContext
) -> dict[str, Any]:
    columns = arguments.columns or list(context.descriptions)
    unknown = set(columns) - context.categorical_columns
    if unknown:
        raise ValueError(f"unknown or non-categorical columns: {sorted(unknown)}")
    profiles = []
    rows = len(context.data)
    for column in columns:
        counts = context.data[column].value_counts(dropna=False)
        top_values = [
            {
                "value": json_scalar(value),
                "count": int(count),
                "frequency": float(count / rows),
            }
            for value, count in counts.head(20).items()
        ]
        profiles.append(
            {
                "name": column,
                "description": context.descriptions[column],
                "distinct_values": int(context.data[column].nunique()),
                "uniqueness_ratio": float(context.data[column].nunique() / rows),
                "key_like_warning": context.data[column].nunique() / rows >= 0.95,
                "top_values": top_values,
                "top_values_truncated": len(counts) > len(top_values),
            }
        )
    result = {"rows": rows, "columns": profiles}
    evidence_id = context.add_evidence(
        INSPECT_TOOL_NAME, arguments.model_dump(), result
    )
    return {"evidence_id": evidence_id, **result}


def analyze_dependency(
    arguments: AnalyzeDependencyArguments, context: CategoricalToolContext
) -> dict[str, Any]:
    signature = _validate_signature(
        arguments.determinants, arguments.dependent, context
    )
    direction = direction_result(
        context.dependency_graph,
        arguments.determinants,
        arguments.dependent,
    )
    if direction["status"] == "rejected_direction":
        return direction
    result = _dependency_stats(
        context.data,
        arguments.determinants,
        arguments.dependent,
        include_configurations=True,
        max_configurations=context.max_configuration_summaries,
    )
    subset_results = []
    if len(arguments.determinants) > 1:
        for size in range(1, len(arguments.determinants)):
            for subset in itertools.combinations(arguments.determinants, size):
                subset_result = _dependency_stats(
                    context.data,
                    list(subset),
                    arguments.dependent,
                    include_configurations=False,
                    max_configurations=0,
                )
                subset_results.append(
                    {
                        "determinants": list(subset),
                        "majority_violation_rate": subset_result[
                            "majority_violation_rate"
                        ],
                        "conditional_entropy": subset_result[
                            "conditional_entropy"
                        ],
                        "determinant_configurations": subset_result[
                            "determinant_configurations"
                        ],
                        "key_like_warning": subset_result["key_like_warning"],
                    }
                )
    result["proper_subset_results"] = subset_results
    evidence_id = context.add_evidence(
        ANALYZE_TOOL_NAME,
        arguments.model_dump(),
        result,
        signature=signature,
    )
    return {"evidence_id": evidence_id, **direction, **result}


def dependent_frequencies(
    arguments: DependentFrequenciesArguments,
    context: CategoricalToolContext,
) -> dict[str, Any]:
    signature = _validate_signature(
        arguments.determinants, arguments.dependent, context
    )
    mask = pd.Series(True, index=context.data.index)
    for column, values in zip(
        arguments.determinants,
        arguments.determinant_values,
        strict=True,
    ):
        domain = {
            scalar_key(json_scalar(value))
            for value in context.data[column].drop_duplicates().tolist()
        }
        invalid = [value for value in values if scalar_key(value) not in domain]
        if invalid:
            raise ValueError(f"unknown values for {column!r}: {invalid}")
        value_keys = {scalar_key(value) for value in values}
        mask &= context.data[column].map(
            lambda value: scalar_key(json_scalar(value)) in value_keys
        )
    selected = context.data.loc[mask, arguments.dependent]
    counts: dict[str, tuple[Any, int]] = {}
    for raw_value, count in selected.value_counts(dropna=False).items():
        value = json_scalar(raw_value)
        counts[scalar_key(value)] = (value, int(count))
    support_count = int(mask.sum())
    result = {
        "support_count": support_count,
        "support": float(support_count / len(context.data)),
        "dependent_frequencies": _frequency_records(counts),
    }
    evidence_id = context.add_evidence(
        FREQUENCIES_TOOL_NAME,
        arguments.model_dump(),
        result,
        signature=signature,
    )
    return {"evidence_id": evidence_id, **result}


def submit_constraint(
    arguments: SubmitConstraintArguments,
    context: CategoricalToolContext,
) -> dict[str, Any]:
    proposal = arguments.constraint
    signature = constraint_signature(proposal)
    cited = [context.evidence.get(evidence_id) for evidence_id in arguments.evidence_ids]
    if any(record is None for record in cited):
        unknown = [
            evidence_id
            for evidence_id, record in zip(arguments.evidence_ids, cited, strict=True)
            if record is None
        ]
        return {
            "status": "rejected",
            "reason": "unknown_evidence_ids",
            "unknown_evidence_ids": unknown,
        }
    if not any(
        record.kind == ANALYZE_TOOL_NAME and record.signature == signature
        for record in cited
        if record is not None
    ):
        return {
            "status": "rejected",
            "reason": "missing_dependency_analysis",
            "required_signature": {
                "determinants": list(signature[0]),
                "dependent": signature[1],
            },
        }

    direction = direction_result(
        context.dependency_graph,
        proposal.determinants,
        proposal.dependent,
    )
    if direction["status"] == "rejected_direction":
        return {**direction, "materialized_value_table": False}

    materialized = False
    if not proposal.value_table:
        proposal = materialize_majority_value_table(proposal, context)
        materialized = True
    result = context.verifier.verify(proposal)
    if result["status"] != "accepted":
        return {**result, "materialized_value_table": materialized}
    fingerprint = constraint_fingerprint(proposal)
    prior = context.accepted.get(fingerprint)
    if prior is not None:
        return {
            **result,
            "status": "duplicate",
            "duplicate_of": prior["constraint"]["id"],
            "materialized_value_table": materialized,
        }
    if len(context.accepted) >= context.max_constraints:
        return {
            **result,
            "status": "rejected",
            "reason": "maximum_constraints_reached",
            "materialized_value_table": materialized,
        }
    context.accepted[fingerprint] = {
        "constraint": proposal.model_dump(),
        "verification": result,
        "evidence_ids": list(arguments.evidence_ids),
    }
    add_dependency(
        context.dependency_graph,
        proposal.determinants,
        proposal.dependent,
    )
    return {
        **result,
        "materialized_value_table": materialized,
        "added_edges": direction["proposed_edges"],
        "topological_order": topological_order(context.dependency_graph),
    }


ARGUMENT_MODELS: dict[str, type[BaseModel]] = {
    INSPECT_TOOL_NAME: InspectColumnsArguments,
    ANALYZE_TOOL_NAME: AnalyzeDependencyArguments,
    FREQUENCIES_TOOL_NAME: DependentFrequenciesArguments,
    SUBMIT_TOOL_NAME: SubmitConstraintArguments,
}


def execute_tool(
    name: str,
    raw_arguments: str | None,
    context: CategoricalToolContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and execute one model-selected tool call."""
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        parsed = {"_raw": raw_arguments}
    model = ARGUMENT_MODELS.get(name)
    if model is None:
        return parsed, {"status": "error", "error": f"unknown tool: {name}"}
    try:
        arguments = model.model_validate_json(raw_arguments or "{}")
        if name == INSPECT_TOOL_NAME:
            result = inspect_columns(arguments, context)
        elif name == ANALYZE_TOOL_NAME:
            result = analyze_dependency(arguments, context)
        elif name == FREQUENCIES_TOOL_NAME:
            result = dependent_frequencies(arguments, context)
        else:
            result = submit_constraint(arguments, context)
        return parsed, result
    except (ValidationError, ValueError) as exc:
        return parsed, {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
