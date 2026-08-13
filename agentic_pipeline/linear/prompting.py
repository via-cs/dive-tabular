"""Render editable prompt templates for linear-constraint agent roles."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd


TEMPLATE_DIR = Path(__file__).with_name("prompt_templates")


def load_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8").strip()


def render_template(name: str, **values: object) -> str:
    return Template(load_template(name)).substitute(
        {key: str(value) for key, value in values.items()}
    ).strip()


COMMON_CONSTRAINT_RULES = load_template("common_constraint_rules.md")
DISCOVERY_SYSTEM_PROMPT = render_template(
    "discovery_system.md", common_rules=COMMON_CONSTRAINT_RULES
)
REFINEMENT_SYSTEM_PROMPT = render_template(
    "refinement_system.md", common_rules=COMMON_CONSTRAINT_RULES
)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def build_discovery_prompt(
    dataset_description: str,
    numerical_descriptions: dict[str, str],
    sample: pd.DataFrame,
    max_constraints: int,
    phase: int,
    max_phases: int,
    accepted_summaries: list[dict[str, object]],
    rejected_summaries: list[dict[str, object]],
) -> str:
    columns = [
        {"name": name, "description": description}
        for name, description in numerical_descriptions.items()
    ]
    return render_template(
        "discovery_user.md",
        phase=phase,
        max_phases=max_phases,
        max_constraints=max_constraints,
        dataset_description=dataset_description,
        columns_json=_json(columns),
        sample_rows=len(sample),
        sample_csv=sample.to_csv(index=False).strip(),
        accepted_json=_json(accepted_summaries),
        rejected_json=_json(rejected_summaries),
    )


def build_refinement_prompt(
    dataset_description: str,
    involved_descriptions: dict[str, str],
    sample: pd.DataFrame,
    candidate_history: list[dict[str, object]],
    phase: int,
    refinement_round: int,
    max_refinement_rounds: int,
) -> str:
    columns = [
        {"name": name, "description": description}
        for name, description in involved_descriptions.items()
    ]
    return render_template(
        "refinement_user.md",
        dataset_description=dataset_description,
        columns_json=_json(columns),
        sample_rows=len(sample),
        sample_csv=sample.to_csv(index=False).strip(),
        candidate_history_json=_json(candidate_history),
        phase=phase,
        refinement_round=refinement_round,
        max_refinement_rounds=max_refinement_rounds,
    )

