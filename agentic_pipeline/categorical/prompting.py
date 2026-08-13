"""Prompt rendering for the categorical dependency proposer."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

import pandas as pd


TEMPLATE_DIR = Path(__file__).with_name("prompt_templates")


def _load(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = _load("system.md")


def build_user_prompt(
    dataset_description: str,
    categorical_descriptions: dict[str, str],
    sample: pd.DataFrame,
    run: int,
    runs: int,
    max_constraints: int,
    violation_threshold: float,
) -> str:
    columns = [
        {"name": name, "description": description}
        for name, description in categorical_descriptions.items()
    ]
    return Template(_load("user.md")).substitute(
        dataset_description=dataset_description,
        columns_json=json.dumps(columns, indent=2, ensure_ascii=False),
        sample_rows=len(sample),
        sample_csv=sample.to_csv(index=False).strip(),
        run=run,
        runs=runs,
        max_constraints=max_constraints,
        violation_threshold=violation_threshold,
    ).strip()
