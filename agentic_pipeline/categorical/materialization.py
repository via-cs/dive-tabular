"""Host-side construction of exact categorical correspondence tables."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from .models import CategoricalConstraintProposal, ValueTableRow, scalar_key
from .verifier import json_scalar, typed_tuple


class _DataContext(Protocol):
    data: pd.DataFrame


def materialize_majority_value_table(
    proposal: CategoricalConstraintProposal,
    context: _DataContext,
) -> CategoricalConstraintProposal:
    """Build singleton exact configurations with one majority dependent value."""
    columns = [*proposal.determinants, proposal.dependent]
    grouped: dict[tuple[str, ...], dict[str, tuple[Any, int]]] = {}
    raw_keys: dict[tuple[str, ...], tuple[Any, ...]] = {}
    for row in context.data[columns].itertuples(index=False, name=None):
        determinant_raw = tuple(json_scalar(value) for value in row[:-1])
        key = typed_tuple(list(determinant_raw))
        raw_keys.setdefault(key, determinant_raw)
        dependent_value = json_scalar(row[-1])
        dependent_key = scalar_key(dependent_value)
        counts = grouped.setdefault(key, {})
        previous = counts.get(dependent_key)
        counts[dependent_key] = (
            dependent_value,
            1 if previous is None else previous[1] + 1,
        )

    rows: list[ValueTableRow] = []
    for key in sorted(grouped):
        majority_value, _ = min(
            grouped[key].values(),
            key=lambda item: (-item[1], scalar_key(json_scalar(item[0]))),
        )
        rows.append(
            ValueTableRow(
                determinant_values=[
                    [json_scalar(value)] for value in raw_keys[key]
                ],
                dependent_values=[json_scalar(majority_value)],
            )
        )
    return CategoricalConstraintProposal.model_validate(
        {
            **proposal.model_dump(),
            "value_table": [row.model_dump() for row in rows],
        }
    )
