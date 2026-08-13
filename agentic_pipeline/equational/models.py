"""Typed records exchanged by the equational-discovery agent and verifier."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EquationalConstraint(BaseModel):
    """One LLM-proposed row-wise equation and its executable check."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable snake_case identifier for the constraint.",
    )
    description: str = Field(
        min_length=1,
        description="Concise statement of the relationship being checked.",
    )
    rationale: str = Field(
        min_length=1,
        description="Why the relationship is semantically plausible.",
    )
    columns: list[str] = Field(
        min_length=2,
        description="All and only the numerical columns referenced by check_code.",
    )
    check_code: str = Field(
        min_length=1,
        description=(
            "Python source defining check(df), which returns an index-aligned "
            "Boolean pandas Series."
        ),
    )

    @field_validator("columns")
    @classmethod
    def unique_columns(cls, columns: list[str]) -> list[str]:
        if len(columns) != len(set(columns)):
            raise ValueError("columns must not contain duplicates")
        return columns


class ConstraintBatch(BaseModel):
    """Arguments accepted by the full-dataset verification tool."""

    model_config = ConfigDict(extra="forbid")

    constraints: list[EquationalConstraint] = Field(min_length=1, max_length=20)


class RejectedHypothesis(BaseModel):
    """A candidate the model deliberately abandons after verification."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    reason: str = Field(min_length=1)


class DiscoveryOutput(BaseModel):
    """A phase-completion response; verified constraints live in the ledger."""

    model_config = ConfigDict(extra="forbid")

    rejected_hypotheses: list[RejectedHypothesis]


class FixProposal(BaseModel):
    """One model proposal for repairing a single target column."""

    model_config = ConfigDict(extra="forbid")

    code: str | None = Field(
        description=(
            "Python source defining fix(df), or null when the target cannot be "
            "reconstructed from the other involved columns."
        )
    )
    rationale: str = Field(
        min_length=1,
        description="Why the proposed reconstruction is semantically valid.",
    )


class FixCodeEntry(BaseModel):
    """Published repair path for one column of a constraint."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1)
    code: str | None
