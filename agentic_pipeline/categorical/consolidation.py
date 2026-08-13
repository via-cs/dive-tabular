"""Deterministic deduplication, subsumption, and compatible-table merging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .models import CategoricalConstraintProposal, ValueTableRow
from .graph import add_dependency, cycle_details, topological_order
from .verifier import (
    CategoricalConstraintVerifier,
    atomic_mapping,
    constraint_fingerprint,
    constraint_signature,
)


AtomicMap = dict[tuple[str, ...], frozenset[str]]


def _signature_mapping(constraint: CategoricalConstraintProposal) -> AtomicMap:
    """Return the atomic map with determinants in sorted signature order."""
    original = atomic_mapping(constraint)
    order = sorted(
        range(len(constraint.determinants)),
        key=lambda index: constraint.determinants[index],
    )
    return {
        tuple(key[index] for index in order): allowed
        for key, allowed in original.items()
    }


def strictly_subsumes(strong: AtomicMap, weak: AtomicMap) -> bool:
    """Return whether enforcing ``strong`` makes ``weak`` redundant."""
    if not set(weak).issubset(strong):
        return False
    if any(not strong[key].issubset(allowed) for key, allowed in weak.items()):
        return False
    return strong != weak


def compatible(left: AtomicMap, right: AtomicMap) -> bool:
    overlap = set(left) & set(right)
    return all(left[key] == right[key] for key in overlap)


def _decode(value: str) -> Any:
    return json.loads(value)


@dataclass
class _Box:
    dimensions: tuple[frozenset[str], ...]
    allowed: frozenset[str]


def _compress_mapping(mapping: AtomicMap) -> list[ValueTableRow]:
    """Merge exact tuples into rectangular set-valued rows without weakening."""
    boxes = [
        _Box(tuple(frozenset({value}) for value in key), allowed)
        for key, allowed in mapping.items()
    ]
    changed = True
    while changed:
        changed = False
        boxes.sort(
            key=lambda box: (
                tuple(tuple(sorted(group)) for group in box.dimensions),
                tuple(sorted(box.allowed)),
            )
        )
        for left_index, left in enumerate(boxes):
            for right_index in range(left_index + 1, len(boxes)):
                right = boxes[right_index]
                if left.allowed != right.allowed:
                    continue
                differences = [
                    index
                    for index, (left_group, right_group) in enumerate(
                        zip(left.dimensions, right.dimensions, strict=True)
                    )
                    if left_group != right_group
                ]
                if len(differences) != 1:
                    continue
                dimension = differences[0]
                combined = list(left.dimensions)
                combined[dimension] = (
                    left.dimensions[dimension] | right.dimensions[dimension]
                )
                boxes[left_index] = _Box(tuple(combined), left.allowed)
                del boxes[right_index]
                changed = True
                break
            if changed:
                break

    rows = [
        ValueTableRow(
            determinant_values=[
                [_decode(value) for value in sorted(group)]
                for group in box.dimensions
            ],
            dependent_values=[_decode(value) for value in sorted(box.allowed)],
        )
        for box in boxes
    ]
    rows.sort(
        key=lambda row: json.dumps(
            row.model_dump(), ensure_ascii=False, sort_keys=True
        )
    )
    return rows


def _record_with_mapping(
    source: dict[str, Any],
    signature: tuple[tuple[str, ...], str],
    mapping: AtomicMap,
    verifier: CategoricalConstraintVerifier,
    merged_from: list[str],
) -> dict[str, Any]:
    original = CategoricalConstraintProposal.model_validate(source["constraint"])
    proposal = original.model_copy(
        update={
            "determinants": list(signature[0]),
            "dependent": signature[1],
            "value_table": _compress_mapping(mapping),
        }
    )
    verification = verifier.verify(proposal)
    return {
        "constraint": proposal.model_dump(),
        "verification": verification,
        "evidence_ids": list(source.get("evidence_ids", [])),
        "merged_from": merged_from,
    }


def consolidate_records(
    records: list[dict[str, Any]],
    verifier: CategoricalConstraintVerifier,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Consolidate accepted records while preserving first-accepted DAG order."""
    input_count = len(records)
    dependency_graph: dict[str, set[str]] = {}
    acyclic_records: list[dict[str, Any]] = []
    cycle_rejected: list[dict[str, Any]] = []
    for record in records:
        proposal = CategoricalConstraintProposal.model_validate(record["constraint"])
        cycles = cycle_details(
            dependency_graph, proposal.determinants, proposal.dependent
        )
        if cycles:
            cycle_rejected.append(
                {
                    "dropped": proposal.id,
                    "reason": "would_create_cycle",
                    "blocking_paths": cycles,
                }
            )
            continue
        add_dependency(
            dependency_graph, proposal.determinants, proposal.dependent
        )
        acyclic_records.append(record)
    records = acyclic_records

    exact_seen: dict[str, dict[str, Any]] = {}
    exact_duplicates: list[dict[str, str]] = []
    for record in records:
        proposal = CategoricalConstraintProposal.model_validate(record["constraint"])
        fingerprint = constraint_fingerprint(proposal)
        if fingerprint in exact_seen:
            exact_duplicates.append(
                {
                    "dropped": proposal.id,
                    "kept": exact_seen[fingerprint]["constraint"]["id"],
                }
            )
            continue
        exact_seen[fingerprint] = record

    grouped: dict[
        tuple[tuple[str, ...], str], list[tuple[dict[str, Any], AtomicMap]]
    ] = {}
    for record in exact_seen.values():
        proposal = CategoricalConstraintProposal.model_validate(record["constraint"])
        grouped.setdefault(constraint_signature(proposal), []).append(
            (record, _signature_mapping(proposal))
        )

    subsumed: list[dict[str, str]] = []
    unresolved_conflicts: list[dict[str, Any]] = []
    consolidated: list[dict[str, Any]] = []
    for signature, candidates in grouped.items():
        survivors: list[tuple[dict[str, Any], AtomicMap]] = []
        for index, (record, mapping) in enumerate(candidates):
            stronger = next(
                (
                    other_record
                    for other_index, (other_record, other_mapping) in enumerate(
                        candidates
                    )
                    if other_index != index
                    and strictly_subsumes(other_mapping, mapping)
                ),
                None,
            )
            if stronger is not None:
                subsumed.append(
                    {
                        "dropped": record["constraint"]["id"],
                        "kept": stronger["constraint"]["id"],
                    }
                )
            else:
                survivors.append((record, mapping))

        components: list[tuple[dict[str, Any], AtomicMap, list[str]]] = []
        for record, mapping in survivors:
            placed = False
            for component_index, (base, combined, ids) in enumerate(components):
                if compatible(combined, mapping):
                    components[component_index] = (
                        base,
                        {**combined, **mapping},
                        [*ids, record["constraint"]["id"]],
                    )
                    placed = True
                    break
            if not placed:
                components.append(
                    (record, dict(mapping), [record["constraint"]["id"]])
                )

        if len(components) > 1:
            unresolved_conflicts.append(
                {
                    "determinants": list(signature[0]),
                    "dependent": signature[1],
                    "constraint_groups": [ids for _, _, ids in components],
                }
            )
        for base, mapping, ids in components:
            consolidated_record = _record_with_mapping(
                base, signature, mapping, verifier, ids
            )
            if consolidated_record["verification"]["status"] != "accepted":
                # This should not occur for compatible accepted tables, but never
                # publish a consolidation result that fails authoritative checks.
                for original, _ in survivors:
                    if original["constraint"]["id"] in ids:
                        consolidated.append(original)
                continue
            consolidated.append(consolidated_record)

    used_ids: set[str] = set()
    for record in consolidated:
        proposal = CategoricalConstraintProposal.model_validate(record["constraint"])
        if proposal.id in used_ids:
            proposal = proposal.model_copy(
                update={"id": f"{proposal.id}_{constraint_fingerprint(proposal)[:8]}"}
            )
            record["constraint"] = proposal.model_dump()
            record["verification"] = verifier.verify(proposal)
        used_ids.add(proposal.id)

    consolidated.sort(key=lambda record: record["constraint"]["id"])
    return consolidated, {
        "input_constraints": input_count,
        "cycle_rejected": cycle_rejected,
        "topological_order": topological_order(dependency_graph),
        "published_constraints": len(consolidated),
        "exact_duplicates": exact_duplicates,
        "strictly_subsumed": subsumed,
        "unresolved_conflicts": unresolved_conflicts,
    }
