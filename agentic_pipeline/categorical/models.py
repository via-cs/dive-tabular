"""Typed model inputs and outputs for categorical dependency discovery."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JsonScalar = str | int | float | bool


def scalar_key(value: JsonScalar) -> str:
    """Return a stable key, normalizing integral JSON floats to integers."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _validate_values(values: list[JsonScalar], label: str) -> list[JsonScalar]:
    if not values:
        raise ValueError(f"{label} must not be empty")
    keys: set[str] = set()
    for value in values:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} values must be finite")
        key = scalar_key(value)
        if key in keys:
            raise ValueError(f"{label} contains duplicate value {value!r}")
        keys.add(key)
    return values


class ValueTableRow(BaseModel):
    """One Cartesian determinant configuration and its admissible values."""

    model_config = ConfigDict(extra="forbid")

    determinant_values: list[list[JsonScalar]] = Field(min_length=1, max_length=10)
    dependent_values: list[JsonScalar] = Field(min_length=1)

    @field_validator("determinant_values")
    @classmethod
    def valid_determinant_values(
        cls, groups: list[list[JsonScalar]]
    ) -> list[list[JsonScalar]]:
        for index, values in enumerate(groups):
            _validate_values(values, f"determinant_values[{index}]")
        return groups

    @field_validator("dependent_values")
    @classmethod
    def valid_dependent_values(
        cls, values: list[JsonScalar]
    ) -> list[JsonScalar]:
        return _validate_values(values, "dependent_values")


class CategoricalConstraintProposal(BaseModel):
    """One unified FD, conditional FD, or admissible categorical mapping."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    determinants: list[str] = Field(min_length=1, max_length=10)
    dependent: str = Field(min_length=1)
    value_table: list[ValueTableRow] = Field(default_factory=list, max_length=100_000)

    @field_validator("determinants")
    @classmethod
    def valid_determinants(cls, columns: list[str]) -> list[str]:
        if any(not column for column in columns):
            raise ValueError("determinant names must be non-empty")
        if len(columns) != len(set(columns)):
            raise ValueError("determinants must be unique")
        return columns

    @model_validator(mode="after")
    def valid_shape(self) -> "CategoricalConstraintProposal":
        if self.dependent in self.determinants:
            raise ValueError("dependent cannot also be a determinant")
        expected = len(self.determinants)
        for index, row in enumerate(self.value_table):
            if len(row.determinant_values) != expected:
                raise ValueError(
                    f"value_table[{index}] has {len(row.determinant_values)} "
                    f"determinant groups; expected {expected}"
                )
        return self


class InspectColumnsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[str] = Field(default_factory=list, max_length=30)


class AnalyzeDependencyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    determinants: list[str] = Field(min_length=1, max_length=10)
    dependent: str = Field(min_length=1)


class DependentFrequenciesArguments(AnalyzeDependencyArguments):
    determinant_values: list[list[JsonScalar]] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def aligned_values(self) -> "DependentFrequenciesArguments":
        if len(self.determinant_values) != len(self.determinants):
            raise ValueError(
                "determinant_values must contain one value list per determinant"
            )
        for index, values in enumerate(self.determinant_values):
            _validate_values(values, f"determinant_values[{index}]")
        return self


class SubmitConstraintArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraint: CategoricalConstraintProposal
    evidence_ids: list[str] = Field(min_length=1, max_length=100)


class RejectedHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ProposerCompletion(BaseModel):
    """Structured response emitted when the model needs no more tools."""

    model_config = ConfigDict(extra="forbid")
    rejected_hypotheses: list[RejectedHypothesis]


def model_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
