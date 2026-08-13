"""Gather the dataset-column end-to-end results table.

The script first refreshes each completed dataset's
``selected_evaluation_summary.json``, then reads those summaries and the
split-local proposed constraint files. It writes:

* a compact CSV containing the paper table cells;
* a Markdown rendering with bold postprocessed constraint endpoints; and
* a JSON detail file retaining full-precision means, standard deviations,
  split counts, metric paths, and per-split constraint counts.

The input may be an experiments root, one dataset experiment directory, or an
exact selected-summary JSON file. Exact summary, dataset-directory, and
constraint-file paths can also be supplied as overrides.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from . import summarize_experiment_results as experiment_summary
else:
    import summarize_experiment_results as experiment_summary


DEFAULT_SUMMARY_NAME = "selected_evaluation_summary.json"
DEFAULT_CSV_NAME = "main_table_results.csv"
DEFAULT_MARKDOWN_NAME = "main_table_results.md"
DEFAULT_JSON_NAME = "main_table_results.json"
ARROW = " → "
DASH = "—"
RANGE_DASH = "–"

CONSTRAINT_FILENAMES = {
    "ld": "categorical_dependency_constraint.json",
    "eq": "equational_constraint.json",
    "lin": "linear_constraint.json",
}

DISPLAY_NAMES = {
    "adult": "Adult",
    "flights": "Flights",
    "heloc": "HELOC",
    "nba": "NBA",
    "news": "News",
    "steel": "Steel",
    "taxi": "Taxi",
    "url": "URL",
}


@dataclass(frozen=True)
class DatasetInput:
    dataset_id: str
    display_name: str
    summary_path: Path
    experiment_dir: Path


@dataclass(frozen=True)
class RowSpec:
    label: str
    metric_path: str | None
    value_kind: str
    family: str | None = None
    delta_only: bool = False


ROW_SPECS = (
    RowSpec("Constraints (LD/Eq/Lin)", None, "counts"),
    RowSpec(
        "LD CVR ↓",
        "constraint.categorical_dependency_cvr",
        "percent",
        family="ld",
    ),
    RowSpec(
        "LD SCVC ↓",
        "constraint.categorical_dependency_scvc",
        "percent",
        family="ld",
    ),
    RowSpec(
        "Eq. CVR ↓",
        "constraint.equational_cvr",
        "percent",
        family="eq",
    ),
    RowSpec(
        "Eq. SCVC ↓",
        "constraint.equational_scvc",
        "percent",
        family="eq",
    ),
    RowSpec(
        "Eq. R² ↑",
        "constraint.average_r2_consistency",
        "r2",
        family="eq",
    ),
    RowSpec(
        "Linear CVR ↓",
        "constraint.linear_cvr",
        "percent",
        family="lin",
    ),
    RowSpec(
        "Linear SCVC ↓",
        "constraint.linear_scvc",
        "percent",
        family="lin",
    ),
    RowSpec(
        "LFD ↓",
        "constraint.linear_feasibility_distance",
        "score",
        family="lin",
    ),
    RowSpec("Utility ↑", None, "score"),
    RowSpec(
        "Shapes ↑",
        "quality.properties[0].Score",
        "score",
    ),
    RowSpec(
        "Δ Pair Trends ↑",
        "quality.properties[1].Score",
        "delta",
        delta_only=True,
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Experiments root, one dataset experiment directory, or an exact "
            f"{DEFAULT_SUMMARY_NAME} file."
        ),
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "Exact selected-summary JSON; repeat to replace directory "
            "discovery. The dataset id defaults to the parent directory name."
        ),
    )
    parser.add_argument(
        "--dataset-summary",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Add or replace one dataset with an exact selected-summary JSON.",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help=(
            "Override the experiment directory used to find split constraint "
            "files for one dataset."
        ),
    )
    parser.add_argument(
        "--constraint-file",
        action="append",
        default=[],
        metavar="DATASET:SPLIT:FAMILY=PATH",
        help=(
            "Override one exact constraint JSON. FAMILY is ld, eq, or lin; "
            "repeat as needed."
        ),
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help=(
            "Reuse existing selected summaries instead of refreshing each "
            "discovered dataset experiment first."
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_named_path(value: str, option: str) -> tuple[str, Path]:
    name, separator, path_text = value.partition("=")
    if not separator or not name.strip() or not path_text.strip():
        raise ValueError(f"{option} must have the form DATASET=PATH: {value!r}")
    return name.strip().lower(), Path(path_text).expanduser().resolve()


def parse_constraint_override(
    value: str,
) -> tuple[tuple[str, str, str], Path]:
    key_text, separator, path_text = value.partition("=")
    components = key_text.split(":")
    if (
        not separator
        or len(components) != 3
        or any(not component for component in components)
        or not path_text.strip()
    ):
        raise ValueError(
            "--constraint-file must have the form "
            f"DATASET:SPLIT:FAMILY=PATH: {value!r}"
        )
    dataset, split, family = components
    family = family.lower()
    if family not in CONSTRAINT_FILENAMES:
        raise ValueError(
            f"Unknown constraint family {family!r}; expected ld, eq, or lin"
        )
    return (
        dataset.lower(),
        split,
        family,
    ), Path(path_text).expanduser().resolve()


def discover_summary_paths(input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    direct = path / DEFAULT_SUMMARY_NAME
    if direct.is_file():
        return [direct]
    discovered = sorted(path.glob(f"*/{DEFAULT_SUMMARY_NAME}"))
    if not discovered:
        raise FileNotFoundError(
            f"No {DEFAULT_SUMMARY_NAME} files found under {path}"
        )
    return discovered


def discover_experiment_dirs(input_path: Path) -> list[Path]:
    """Find dataset experiment directories that can be summarized."""
    path = input_path.expanduser().resolve()
    if path.is_file():
        return []
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    if (path / "splits").is_dir():
        return [path]
    return sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and (child / "splits").is_dir()
    )


def refresh_selected_summaries(input_path: Path) -> list[Path]:
    """Create the selected JSON and CSV summaries for every experiment."""
    outputs = []
    for experiment_dir in discover_experiment_dirs(input_path):
        summary_args = argparse.Namespace(
            experiment_dir=experiment_dir,
            split_dir=[],
            evaluation_file=[],
            real_metrics_cache=[],
            output_json=None,
            output_csv=None,
            no_csv=False,
        )
        summary = experiment_summary.build_summary(summary_args)
        json_path = experiment_dir / experiment_summary.DEFAULT_JSON_NAME
        csv_path = experiment_dir / experiment_summary.DEFAULT_CSV_NAME
        experiment_summary.write_json(json_path, summary)
        experiment_summary.write_csv(
            csv_path, experiment_summary.csv_rows(summary)
        )
        outputs.append(json_path)
        print(f"Summarized {experiment_dir.name}: {json_path}")
    return outputs


def dataset_input(
    dataset_id: str,
    summary_path: Path,
    experiment_dir_override: Path | None,
) -> DatasetInput:
    summary_path = summary_path.expanduser().resolve()
    summary = read_json(summary_path)
    configured_dir = summary.get("experiment_dir")
    if experiment_dir_override is not None:
        experiment_dir = experiment_dir_override
    elif isinstance(configured_dir, str) and configured_dir:
        experiment_dir = Path(configured_dir).expanduser().resolve()
    else:
        experiment_dir = summary_path.parent
    return DatasetInput(
        dataset_id=dataset_id,
        display_name=DISPLAY_NAMES.get(dataset_id, dataset_id.replace("_", " ").title()),
        summary_path=summary_path,
        experiment_dir=experiment_dir,
    )


def resolve_inputs(args: argparse.Namespace) -> list[DatasetInput]:
    directory_overrides = dict(
        parse_named_path(value, "--dataset-dir") for value in args.dataset_dir
    )
    if args.summary_file:
        summary_paths = [path.expanduser().resolve() for path in args.summary_file]
    else:
        summary_paths = discover_summary_paths(args.input)

    paths_by_dataset = {path.parent.name.lower(): path for path in summary_paths}
    for value in args.dataset_summary:
        name, path = parse_named_path(value, "--dataset-summary")
        paths_by_dataset[name] = path
    if len(paths_by_dataset) != len(summary_paths) + len(args.dataset_summary):
        # Replacing an inferred path via --dataset-summary is valid. Duplicate
        # explicit names, however, are almost always an invocation mistake.
        explicit_names = [
            parse_named_path(value, "--dataset-summary")[0]
            for value in args.dataset_summary
        ]
        if len(explicit_names) != len(set(explicit_names)):
            raise ValueError("Duplicate --dataset-summary dataset ids")

    unknown_directories = sorted(set(directory_overrides) - set(paths_by_dataset))
    if unknown_directories:
        raise ValueError(
            "--dataset-dir has no matching selected summary for: "
            f"{unknown_directories}"
        )
    return [
        dataset_input(
            name,
            path,
            directory_overrides.get(name),
        )
        for name, path in sorted(paths_by_dataset.items())
    ]


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def constraint_count(path: Path) -> int:
    document: Any
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Constraint file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if isinstance(document, list):
        return len(document)
    if isinstance(document, dict) and isinstance(document.get("constraints"), list):
        return len(document["constraints"])
    raise ValueError(f"Cannot find a constraint list in {path}")


def compact_range(values: Sequence[int]) -> str:
    minimum = min(values)
    maximum = max(values)
    return str(minimum) if minimum == maximum else f"{minimum}{RANGE_DASH}{maximum}"


def constraint_counts(
    dataset: DatasetInput,
    summary: dict[str, Any],
    overrides: dict[tuple[str, str, str], Path],
) -> tuple[dict[str, list[int]], str]:
    splits = summary.get("splits")
    if not isinstance(splits, list) or not splits or not all(
        isinstance(split, str) and split for split in splits
    ):
        raise ValueError(f"Summary has no valid split list: {dataset.summary_path}")
    result = {family: [] for family in CONSTRAINT_FILENAMES}
    for split in splits:
        for family, filename in CONSTRAINT_FILENAMES.items():
            path = overrides.get(
                (dataset.dataset_id, split, family),
                dataset.experiment_dir / "splits" / split / "constraints" / filename,
            )
            result[family].append(constraint_count(path))
    compact = "/".join(compact_range(result[family]) for family in ("ld", "eq", "lin"))
    return result, compact


def utility_metric_path(summary: dict[str, Any], dataset: DatasetInput) -> str:
    selection = summary.get("utility_model_selection")
    if not isinstance(selection, dict):
        raise ValueError(f"Missing utility_model_selection in {dataset.summary_path}")
    model = selection.get("selected_model")
    score = selection.get("score")
    if not isinstance(model, str) or not isinstance(score, str):
        raise ValueError(f"Invalid utility model selection in {dataset.summary_path}")
    path = f"utility.models.{model}.tstr.{score}"
    if path not in summary.get("metric_paths", []):
        raise ValueError(f"Selected utility metric {path!r} is absent from the summary")
    return path


def metric_detail(summary: dict[str, Any], path: str) -> dict[str, Any]:
    across = summary.get("across_generators", {})
    try:
        raw = across["raw"]["metrics"][path]
        constrained = across["constrained"]["metrics"][path]
        comparison = across["comparison"][path]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Summary is missing metric {path!r}") from exc
    return {
        "metric_path": path,
        "raw_mean": finite_number(raw.get("mean")),
        "raw_std": finite_number(raw.get("std")),
        "raw_n_splits": raw.get("n"),
        "postprocessed_mean": finite_number(constrained.get("mean")),
        "postprocessed_std": finite_number(constrained.get("std")),
        "postprocessed_n_splits": constrained.get("n"),
        "delta": finite_number(comparison.get("constrained_minus_raw")),
    }


def format_number(value: float, kind: str) -> str:
    if kind == "percent":
        return f"{100.0 * value:.1f}"
    if kind == "r2":
        return f"{value:.2f}"
    if kind == "score":
        return f"{value:.3f}"
    if kind == "delta":
        if value != 0.0 and round(value, 3) == 0.0:
            return f"{value:+.4f}"
        return f"{value:+.3f}"
    raise ValueError(f"Unknown value kind: {kind}")


def pair_cell(detail: dict[str, Any], kind: str, markdown: bool) -> str:
    raw = detail["raw_mean"]
    postprocessed = detail["postprocessed_mean"]
    if raw is None or postprocessed is None:
        return DASH
    raw_text = format_number(raw, kind)
    post_text = format_number(postprocessed, kind)
    if markdown:
        post_text = f"**{post_text}**"
    return f"{raw_text}{ARROW}{post_text}"


def delta_cell(detail: dict[str, Any]) -> str:
    delta = detail["delta"]
    return DASH if delta is None else format_number(delta, "delta")


def gather(
    datasets: Sequence[DatasetInput],
    constraint_overrides: dict[tuple[str, str, str], Path],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    plain_rows = [{"Metric": spec.label} for spec in ROW_SPECS]
    markdown_rows = [{"Metric": spec.label} for spec in ROW_SPECS]

    for dataset in datasets:
        summary = read_json(dataset.summary_path)
        counts, compact_counts = constraint_counts(
            dataset, summary, constraint_overrides
        )
        applicable = {
            family: any(value > 0 for value in family_counts)
            for family, family_counts in counts.items()
        }
        utility_path = utility_metric_path(summary, dataset)
        dataset_details: dict[str, Any] = {
            "display_name": dataset.display_name,
            "summary_path": str(dataset.summary_path),
            "experiment_dir": str(dataset.experiment_dir),
            "splits": summary["splits"],
            "generator_count": summary.get("across_generators", {}).get(
                "generator_count"
            ),
            "constraint_counts": counts,
            "constraint_count_cell": compact_counts,
            "families_applicable": applicable,
            "utility_model_selection": summary.get("utility_model_selection"),
            "metrics": {},
        }

        for index, spec in enumerate(ROW_SPECS):
            if spec.value_kind == "counts":
                plain = markdown = compact_counts
            else:
                metric_path = utility_path if spec.label == "Utility ↑" else spec.metric_path
                assert metric_path is not None
                detail = metric_detail(summary, metric_path)
                dataset_details["metrics"][spec.label] = detail
                if spec.family is not None and not applicable[spec.family]:
                    plain = markdown = DASH
                elif spec.delta_only:
                    plain = markdown = delta_cell(detail)
                else:
                    plain = pair_cell(detail, spec.value_kind, markdown=False)
                    markdown = pair_cell(detail, spec.value_kind, markdown=True)
            plain_rows[index][dataset.display_name] = plain
            markdown_rows[index][dataset.display_name] = markdown
        details[dataset.dataset_id] = dataset_details

    return {
        "schema_version": 1,
        "aggregation": (
            "means from selected_evaluation_summary.json: samples within "
            "generator and split, equal-weight generators within split, then "
            "mean across splits"
        ),
        "datasets": [dataset.dataset_id for dataset in datasets],
        "columns": [dataset.display_name for dataset in datasets],
        "rows": plain_rows,
        "markdown_rows": markdown_rows,
        "details": details,
    }


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", *columns])
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(columns: Sequence[str], rows: Sequence[dict[str, str]]) -> str:
    header = "| " + " | ".join(["Metric", *columns]) + " |"
    alignment = "|---|" + "|".join("---:" for _ in columns) + "|"
    body = [
        "| " + " | ".join([row["Metric"], *(row[column] for column in columns)]) + " |"
        for row in rows
    ]
    return "\n".join([header, alignment, *body]) + "\n"


def default_output_parent(input_path: Path) -> Path:
    resolved = input_path.expanduser().resolve()
    return resolved.parent if resolved.is_file() else resolved


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if not args.no_summarize and not args.summary_file:
            refresh_selected_summaries(args.input)
        datasets = resolve_inputs(args)
        constraint_overrides = dict(
            parse_constraint_override(value) for value in args.constraint_file
        )
        unknown_datasets = sorted(
            {key[0] for key in constraint_overrides}
            - {dataset.dataset_id for dataset in datasets}
        )
        if unknown_datasets:
            raise ValueError(
                "--constraint-file has no matching selected summary for: "
                f"{unknown_datasets}"
            )
        result = gather(datasets, constraint_overrides)
        parent = default_output_parent(args.input)
        csv_path = (args.output_csv or parent / DEFAULT_CSV_NAME).resolve()
        markdown_path = (
            args.output_markdown or parent / DEFAULT_MARKDOWN_NAME
        ).resolve()
        json_path = (args.output_json or parent / DEFAULT_JSON_NAME).resolve()
        write_csv(csv_path, result["columns"], result["rows"])
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            markdown_table(result["columns"], result["markdown_rows"]),
            encoding="utf-8",
        )
        json_output = {key: value for key, value in result.items() if key != "markdown_rows"}
        write_json(json_path, json_output)
        print(markdown_table(result["columns"], result["markdown_rows"]), end="")
        print(f"\nWrote CSV to {csv_path}")
        print(f"Wrote Markdown to {markdown_path}")
        print(f"Wrote JSON details to {json_path}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
