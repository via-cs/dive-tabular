"""Typed records exchanged by the linear proposer and verifier."""

from __future__ import annotations

import math
from pydantic import BaseModel, ConfigDict, Field, field_validator


class LinearConstraintProposal(BaseModel):
    """One LLM-proposed canonical linear inequality."""

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
        description="Why the inequality is a universal semantic relationship.",
    )
    coefficients: dict[str, float] = Field(
        min_length=2,
        max_length=30,
        description=(
            "Nonzero coefficients in the canonical inequality "
            "sum(a_j * x_j) >= rhs."
        ),
    )
    rhs: float = Field(
        description="Finite right-hand side of sum(a_j * x_j) >= rhs."
    )

    @field_validator("coefficients")
    @classmethod
    def valid_coefficients(
        cls, coefficients: dict[str, float]
    ) -> dict[str, float]:
        for column, value in coefficients.items():
            if not column:
                raise ValueError("coefficient column names must be non-empty")
            if not math.isfinite(value):
                raise ValueError(
                    f"coefficient for {column!r} must be finite"
                )
            if value == 0:
                raise ValueError(
                    f"coefficient for {column!r} must be nonzero"
                )
        return coefficients

    @field_validator("rhs")
    @classmethod
    def finite_rhs(cls, rhs: float) -> float:
        if not math.isfinite(rhs):
            raise ValueError("rhs must be finite")
        return rhs


class ConstraintBatch(BaseModel):
    """Arguments accepted by the full-dataset verification tool."""

    model_config = ConfigDict(extra="forbid")

    constraints: list[LinearConstraintProposal] = Field(
        min_length=1, max_length=20
    )


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
    """Phase completion; verified constraints are stored in the host ledger."""

    model_config = ConfigDict(extra="forbid")

    rejected_hypotheses: list[RejectedHypothesis]

