"""Build a low-cardinality categorical view of the Social Anxiety Dataset.

The positional source accepts either a directory containing ``data.csv`` or an
exact CSV file. ``--data-file`` can explicitly override either form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from agentic_pipeline.categorical.models import CategoricalConstraintProposal
from evaluation.metrics.categorical_constraint import evaluate_dataframe


DATASET_URL = "https://www.kaggle.com/datasets/natezhang123/social-anxiety-dataset"
DATASET_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_SOURCE = DATASET_DIRECTORY / "raw"
DEFAULT_OUTPUT = DATASET_DIRECTORY

OCCUPATION_GROUPS = {
    "Scientist": "Higher caffeine floor",
    "Doctor": "Higher caffeine floor",
    "Engineer": "Higher caffeine floor",
    "Lawyer": "Higher caffeine floor",
    "Student": "Moderate caffeine floor",
    "Nurse": "Moderate caffeine floor",
    "Freelancer": "Moderate caffeine floor",
    "Chef": "Moderate caffeine floor",
    "Artist": "No annotated caffeine floor",
    "Athlete": "No annotated caffeine floor",
    "Musician": "No annotated caffeine floor",
    "Other": "No annotated caffeine floor",
    "Teacher": "No annotated caffeine floor",
}


def _cut(
    series: pd.Series,
    bins: list[float],
    labels: list[str],
) -> pd.Series:
    result = pd.cut(
        pd.to_numeric(series, errors="raise"),
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    if result.isna().any():
        bad = series[result.isna()].drop_duplicates().tolist()
        raise ValueError(f"Values fall outside configured discretization: {bad}")
    return result.astype(str)


DISCRETIZERS: dict[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
    "Age": (
        "Age Band",
        lambda values: _cut(
            values,
            [float("-inf"), 19, 49, float("inf")],
            ["Under 20", "20-49", "50 or older"],
        ),
    ),
    "Sleep Hours": (
        "Sleep Band",
        lambda values: _cut(
            values,
            [float("-inf"), 5.9, 9, float("inf")],
            ["Under 6 hours", "6-9 hours", "Over 9 hours"],
        ),
    ),
    "Physical Activity (hrs/week)": (
        "Physical Activity Band",
        lambda values: _cut(
            values,
            [float("-inf"), 4, float("inf")],
            ["0-4 hours", "Over 4 hours"],
        ),
    ),
    "Caffeine Intake (mg/day)": (
        "Caffeine Band",
        lambda values: _cut(
            values,
            [float("-inf"), 99, 249, 293, float("inf")],
            ["0-99 mg", "100-249 mg", "250-293 mg", "294 mg or more"],
        ),
    ),
    "Alcohol Consumption (drinks/week)": (
        "Alcohol Band",
        lambda values: _cut(
            values,
            [float("-inf"), 4, float("inf")],
            ["0-4 drinks", "5 or more drinks"],
        ),
    ),
    "Stress Level (1-10)": (
        "Stress Band",
        lambda values: _cut(
            values,
            [float("-inf"), 3, 6, 8, float("inf")],
            ["Low (1-3)", "Moderate (4-6)", "High (7-8)", "Very high (9-10)"],
        ),
    ),
    "Heart Rate (bpm)": (
        "Heart Rate Band",
        lambda values: _cut(
            values,
            [float("-inf"), 84, float("inf")],
            ["Below 85 bpm", "85 bpm or higher"],
        ),
    ),
    "Breathing Rate (breaths/min)": (
        "Breathing Rate Band",
        lambda values: _cut(
            values,
            [float("-inf"), 19, float("inf")],
            ["Below 20 breaths/min", "20 breaths/min or higher"],
        ),
    ),
    "Sweating Level (1-5)": (
        "Sweating Band",
        lambda values: _cut(
            values,
            [float("-inf"), 2, 3, float("inf")],
            ["Low (1-2)", "Moderate (3)", "High (4-5)"],
        ),
    ),
    "Therapy Sessions (per month)": (
        "Therapy Band",
        lambda values: _cut(
            values,
            [float("-inf"), 2, float("inf")],
            ["0-2 sessions", "3 or more sessions"],
        ),
    ),
    "Diet Quality (1-10)": (
        "Diet Quality Band",
        lambda values: _cut(
            values,
            [float("-inf"), 4, 7, float("inf")],
            ["Low (1-4)", "Moderate (5-7)", "High (8-10)"],
        ),
    ),
    "Anxiety Level (1-10)": (
        "Anxiety Band",
        lambda values: _cut(
            values,
            [float("-inf"), 3, 6, 8, float("inf")],
            ["Low (1-3)", "Moderate (4-6)", "High (7-8)", "Very high (9-10)"],
        ),
    ),
}


PASSTHROUGH_COLUMNS = (
    "Gender",
    "Occupation",
    "Smoking",
    "Family History of Anxiety",
    "Dizziness",
    "Medication",
    "Recent Major Life Event",
)


DESCRIPTIONS = {
    "Age Band": "Age grouped around the expert very-high-anxiety interval of 20 through 49.",
    "Gender": "Self-reported gender category.",
    "Occupation": "Reported occupation; thirteen categories occur in the dataset.",
    "Occupation Group": "Occupation grouping induced by the two annotated caffeine-floor rules.",
    "Sleep Band": "Daily sleep duration grouped as under 6, 6-9, or over 9 hours.",
    "Physical Activity Band": "Weekly activity grouped at the annotated four-hour threshold.",
    "Caffeine Band": "Daily caffeine grouped at 100, 250, and 294 mg thresholds used by expert rules.",
    "Alcohol Band": "Weekly alcohol consumption grouped at the annotated five-drink threshold.",
    "Smoking": "Whether the participant reports smoking.",
    "Family History of Anxiety": "Whether the participant reports a family history of anxiety.",
    "Stress Band": "Stress score grouped into low, moderate, high, and very-high bands.",
    "Heart Rate Band": "Heart rate grouped at the annotated 85 bpm threshold.",
    "Breathing Rate Band": "Breathing rate grouped at the annotated 20 breaths/min threshold.",
    "Sweating Band": "Sweating score grouped into low, moderate, and high bands.",
    "Dizziness": "Whether dizziness is reported.",
    "Medication": "Whether anxiety-related medication is reported.",
    "Therapy Band": "Monthly therapy sessions grouped at the annotated three-session threshold.",
    "Recent Major Life Event": "Whether a recent major life event is reported.",
    "Diet Quality Band": "Diet quality grouped into low, moderate, and high bands.",
    "Anxiety Band": "Anxiety score grouped into low, moderate, high, and very-high bands.",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Anxiety dataset directory or exact source CSV.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Exact source CSV overriding the positional source.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_data(source: Path, data_file: Path | None = None) -> Path:
    path = (data_file if data_file is not None else source).expanduser().resolve()
    resolved = path / "data.csv" if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(f"Anxiety CSV not found: {resolved}")
    return resolved


def build_frame(source: pd.DataFrame) -> pd.DataFrame:
    required = [*DISCRETIZERS, *PASSTHROUGH_COLUMNS]
    missing = [column for column in required if column not in source]
    if missing:
        raise ValueError(f"Source anxiety CSV is missing columns: {missing}")

    output: dict[str, pd.Series] = {}
    for source_column in source.columns:
        if source_column in DISCRETIZERS:
            target, transform = DISCRETIZERS[source_column]
            output[target] = transform(source[source_column])
        elif source_column in PASSTHROUGH_COLUMNS:
            output[source_column] = source[source_column].astype(str)
        if source_column == "Occupation":
            unknown = sorted(set(source[source_column]) - set(OCCUPATION_GROUPS))
            if unknown:
                raise ValueError(f"Unassigned occupations: {unknown}")
            output["Occupation Group"] = source[source_column].map(OCCUPATION_GROUPS)
    frame = pd.DataFrame(output)
    if list(frame) != list(DESCRIPTIONS):
        raise AssertionError(
            f"Unexpected output column order: {list(frame)}; expected {list(DESCRIPTIONS)}"
        )
    if frame.isna().any().any():
        raise ValueError("Categorical anxiety view contains missing values")
    return frame


def _constraint(
    constraint_id: str,
    description: str,
    rationale: str,
    determinants: list[str],
    dependent: str,
    rows: list[tuple[list[list[str]], list[str]]],
) -> dict[str, Any]:
    proposal = CategoricalConstraintProposal(
        id=constraint_id,
        description=description,
        rationale=rationale,
        determinants=determinants,
        dependent=dependent,
        value_table=[
            {
                "determinant_values": determinant_values,
                "dependent_values": dependent_values,
            }
            for determinant_values, dependent_values in rows
        ],
    )
    return proposal.model_dump()


def build_constraint_document(frame: pd.DataFrame) -> dict[str, Any]:
    very_high = [["Very high (9-10)"]]
    expert_rules = [
        (
            "very_high_anxiety_to_caffeine_band",
            "Very-high anxiety requires at least 294 mg/day caffeine in the annotated data.",
            "Caffeine Band",
            ["294 mg or more"],
        ),
        (
            "very_high_anxiety_to_breathing_rate_band",
            "Very-high anxiety requires breathing rate of at least 20 breaths/min.",
            "Breathing Rate Band",
            ["20 breaths/min or higher"],
        ),
        (
            "very_high_anxiety_to_heart_rate_band",
            "Very-high anxiety requires heart rate of at least 85 bpm.",
            "Heart Rate Band",
            ["85 bpm or higher"],
        ),
        (
            "very_high_anxiety_to_alcohol_band",
            "Very-high anxiety requires at least five drinks/week in the annotated data.",
            "Alcohol Band",
            ["5 or more drinks"],
        ),
        (
            "very_high_anxiety_to_diet_quality_band",
            "Very-high anxiety permits only the low diet-quality band.",
            "Diet Quality Band",
            ["Low (1-4)"],
        ),
        (
            "very_high_anxiety_to_therapy_band",
            "Very-high anxiety requires at least three therapy sessions/month.",
            "Therapy Band",
            ["3 or more sessions"],
        ),
        (
            "very_high_anxiety_to_age_band",
            "Very-high anxiety is restricted to ages 20 through 49 in the annotated data.",
            "Age Band",
            ["20-49"],
        ),
        (
            "very_high_anxiety_to_physical_activity_band",
            "Very-high anxiety permits at most four hours of weekly physical activity.",
            "Physical Activity Band",
            ["0-4 hours"],
        ),
    ]
    constraints = [
        _constraint(
            constraint_id,
            description,
            "This is the categorical form of an existing expert annotation in dataset/anxiety/constraints.json.",
            ["Anxiety Band"],
            dependent,
            [(very_high, allowed)],
        )
        for constraint_id, description, dependent, allowed in expert_rules
    ]

    validated_discoveries = [
        (
            "very_high_anxiety_to_sleep_band",
            "Very-high anxiety excludes the over-nine-hours sleep band.",
            "Sleep Band",
            ["Under 6 hours", "6-9 hours"],
        ),
        (
            "very_high_anxiety_to_stress_band",
            "Very-high anxiety permits only high or very-high stress.",
            "Stress Band",
            ["High (7-8)", "Very high (9-10)"],
        ),
        (
            "very_high_anxiety_to_sweating_band",
            "Very-high anxiety permits only moderate or high sweating.",
            "Sweating Band",
            ["Moderate (3)", "High (4-5)"],
        ),
    ]
    constraints.extend(
        _constraint(
            constraint_id,
            description,
            "This pipeline-discovered relationship was independently validated "
            "with zero violations on the held-out audit split before being added "
            "to ground truth.",
            ["Anxiety Band"],
            dependent,
            [(very_high, allowed)],
        )
        for constraint_id, description, dependent, allowed in validated_discoveries
    )

    occupation_rows = [
        ([[occupation]], [group])
        for occupation, group in sorted(OCCUPATION_GROUPS.items())
    ]
    constraints.append(
        _constraint(
            "occupation_to_occupation_group",
            "Occupation determines its expert-rule caffeine-floor group.",
            "This derived label makes the two occupation lists explicit without a large value table.",
            ["Occupation"],
            "Occupation Group",
            occupation_rows,
        )
    )
    constraints.append(
        _constraint(
            "occupation_group_to_caffeine_band",
            "Annotated occupation groups restrict the admissible caffeine bands.",
            "This combines the two occupation/caffeine expert annotations in one set-valued categorical mapping.",
            ["Occupation Group"],
            "Caffeine Band",
            [
                (
                    [["Higher caffeine floor"]],
                    ["250-293 mg", "294 mg or more"],
                ),
                (
                    [["Moderate caffeine floor"]],
                    ["100-249 mg", "250-293 mg", "294 mg or more"],
                ),
            ],
        )
    )
    document = {"dsl_version": "2.0", "constraints": constraints}
    report = evaluate_dataframe(frame, document)
    if report["overall"]["total_constraint_violations"]:
        raise AssertionError(
            "Discretized anxiety data violates its expert ground truth: "
            f"{report['overall']['total_constraint_violations']} violations"
        )
    return document


def build_info(frame: pd.DataFrame) -> dict[str, Any]:
    columns = list(frame)
    return {
        "name": "anxiety-categorical",
        "source": DATASET_URL,
        "license": "CC0: Public Domain",
        "task": "categorical_dependency_detection",
        "target": "Anxiety Band",
        "features_by_order": columns,
        "shape": [int(len(frame)), int(len(columns))],
        "col_types": {
            column: {
                "type": "cat",
                "unique_values": int(frame[column].nunique(dropna=False)),
                "missing_values": int(frame[column].isna().sum()),
            }
            for column in columns
        },
    }


def build_meta() -> dict[str, Any]:
    return {
        "dataset_description": (
            "A low-cardinality categorical view of an anxiety and lifestyle survey. "
            "Continuous measurements and ordinal scores are grouped at interpretable "
            "or expert-annotated thresholds. Existing expert rules relate the very-high "
            "anxiety band and occupation groups to admissible physiological, lifestyle, "
            "demographic, and treatment bands."
        ),
        "column_descriptions": DESCRIPTIONS,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ensure_writable(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing files: {existing}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = resolve_data(args.source, args.data_file)
    output_dir = args.output_dir.expanduser().resolve()
    constraints_dir = output_dir / "constraints_expert"
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / "data.csv",
        output_dir / "meta.json",
        output_dir / "info.json",
        constraints_dir / "categorical_dependency_constraint.json",
    )
    ensure_writable(paths, args.overwrite)

    frame = build_frame(pd.read_csv(source_path))
    document = build_constraint_document(frame)
    frame.to_csv(paths[0], index=False)
    write_json(paths[1], build_meta())
    write_json(paths[2], build_info(frame))
    write_json(paths[3], document)
    cardinalities = {column: int(frame[column].nunique()) for column in frame}
    print(
        f"Wrote {len(frame):,} rows, {len(frame.columns)} columns, and "
        f"{len(document['constraints'])} constraints to {output_dir}; "
        f"maximum cardinality={max(cardinalities.values())}"
    )


if __name__ == "__main__":
    main()
