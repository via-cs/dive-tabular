"""Summarize selected raw versus constrained metrics across experiment splits.

For every variant, the script first averages repeated synthetic samples within
each generator and split, then averages the generators with equal weight. It
reports the mean and sample standard deviation across those generator-averaged
split means. The utility model is selected globally by the
highest mean TRTR R2 (regression) or ROC AUC (binary classification) read from
each split's ``real_metrics_cache.json``; only that model's TSTR score is
reported for synthetic data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_JSON_NAME = "selected_evaluation_summary.json"
DEFAULT_CSV_NAME = "selected_evaluation_summary.csv"
VARIANTS = ("raw", "constrained")
BASE_METRICS = (
    "constraint.average_r2_consistency",
    "constraint.categorical_dependency_cvr",
    "constraint.categorical_dependency_scvc",
    "constraint.equational_cvr",
    "constraint.equational_scvc",
    "constraint.linear_cvr",
    "constraint.linear_scvc",
    "constraint.linear_feasibility_distance",
    "quality.properties[0].Score",
    "quality.properties[1].Score",
)


@dataclass(frozen=True)
class EvaluationInput:
    split_id: str
    split_dir: Path
    generator: str
    variant: str
    path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Experiment root containing splits/split_*/generators/.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Exact split directory; repeat to replace experiment-root split "
            "discovery."
        ),
    )
    parser.add_argument(
        "--evaluation-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "Exact raw/constrained evaluation.json; repeat to replace all "
            "evaluation-file discovery. The path must be under "
            "generators/<generator>/<raw|constrained>/."
        ),
    )
    parser.add_argument(
        "--real-metrics-cache",
        type=Path,
        action="append",
        default=[],
        help=(
            "Exact real_metrics_cache.json; repeat once per selected split, in "
            "sorted split order, to replace split-local cache lookup."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=f"JSON output file (default: experiment_dir/{DEFAULT_JSON_NAME}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=f"CSV output file (default: experiment_dir/{DEFAULT_CSV_NAME}).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the companion flat CSV summary.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def split_id(path: Path) -> str:
    for candidate in (path, *path.parents):
        if candidate.name.startswith("split_"):
            return candidate.name
    raise ValueError(f"Could not infer split directory from path: {path}")


def infer_evaluation_input(path: Path) -> EvaluationInput:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {resolved}")
    variant_dir = resolved.parent
    generator_dir = variant_dir.parent
    generators_dir = generator_dir.parent
    if variant_dir.name not in VARIANTS or generators_dir.name != "generators":
        raise ValueError(
            "Evaluation files must be under "
            f"generators/<generator>/<raw|constrained>/: {resolved}"
        )
    inferred_split_id = split_id(resolved)
    inferred_split_dir = next(
        parent for parent in resolved.parents if parent.name == inferred_split_id
    )
    return EvaluationInput(
        split_id=inferred_split_id,
        split_dir=inferred_split_dir,
        generator=generator_dir.name,
        variant=variant_dir.name,
        path=resolved,
    )


def discover_split_dirs(experiment_dir: Path, explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        paths = [path.expanduser().resolve() for path in explicit]
    else:
        root = experiment_dir.expanduser().resolve()
        paths = sorted(path for path in (root / "splits").glob("split_*") if path.is_dir())
    if not paths:
        raise FileNotFoundError("No split directories found")
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Split directory not found: {path}")
    identifiers = [path.name for path in paths]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate split identifiers: {identifiers}")
    return sorted(paths, key=lambda path: path.name)


def discover_evaluations(
    split_dirs: Sequence[Path], explicit: Sequence[Path]
) -> list[EvaluationInput]:
    if explicit:
        inputs = [infer_evaluation_input(path) for path in explicit]
    else:
        inputs = []
        for directory in split_dirs:
            for path in sorted((directory / "generators").glob("*/raw/evaluation.json")):
                inputs.append(infer_evaluation_input(path))
            for path in sorted(
                (directory / "generators").glob("*/constrained/evaluation.json")
            ):
                inputs.append(infer_evaluation_input(path))
    if not inputs:
        raise FileNotFoundError("No raw or constrained evaluation files found")

    selected_split_ids = {path.name for path in split_dirs}
    unexpected = sorted({item.split_id for item in inputs} - selected_split_ids)
    if unexpected:
        raise ValueError(
            "Evaluation files refer to splits not selected by --split-dir: "
            f"{unexpected}"
        )
    keys = [(item.split_id, item.generator, item.variant) for item in inputs]
    if len(keys) != len(set(keys)):
        raise ValueError("Multiple evaluation files supplied for one split/generator/variant")

    generators = sorted({item.generator for item in inputs})
    missing = []
    for split in sorted(selected_split_ids):
        for generator in generators:
            for variant in VARIANTS:
                if (split, generator, variant) not in set(keys):
                    missing.append(f"{split}/{generator}/{variant}")
    if missing:
        raise ValueError(
            "Every selected generator needs raw and constrained evaluations in "
            f"every selected split; missing: {missing}"
        )
    return sorted(inputs, key=lambda item: (item.split_id, item.generator, item.variant))


def resolve_cache_paths(
    split_dirs: Sequence[Path], explicit: Sequence[Path]
) -> dict[str, Path]:
    if explicit and len(explicit) != len(split_dirs):
        raise ValueError(
            "--real-metrics-cache must be repeated once per selected split "
            f"({len(split_dirs)} expected, {len(explicit)} supplied)"
        )
    paths = (
        [path.expanduser().resolve() for path in explicit]
        if explicit
        else [directory / "real_metrics_cache.json" for directory in split_dirs]
    )
    result = {}
    for directory, path in zip(split_dirs, paths, strict=True):
        if not path.is_file():
            raise FileNotFoundError(f"Real metrics cache not found: {path}")
        result[directory.name] = path
    return result


def cache_utility(path: Path) -> dict[str, Any]:
    document = read_json(path)
    try:
        utility = document["real_metrics"]["utility"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing real_metrics.utility in cache: {path}") from exc
    if not isinstance(utility, dict) or not isinstance(utility.get("models"), dict):
        raise ValueError(f"Invalid cached utility structure: {path}")
    return utility


def score_name_for_task(task: str) -> str:
    if task == "regression":
        return "r2"
    if task == "binary_classification":
        return "roc_auc"
    if task == "multiclass_classification":
        raise ValueError(
            "Multiclass utility does not currently emit ROC AUC; cannot apply "
            "the requested ROC-AUC model-selection rule"
        )
    raise ValueError(f"Unsupported utility task in cache: {task!r}")


def summarize(values: Iterable[float | None]) -> dict[str, Any]:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
    }


def select_utility_model(
    caches: dict[str, Path],
) -> tuple[str, str, str, dict[str, Any]]:
    utilities = {identifier: cache_utility(path) for identifier, path in caches.items()}
    tasks = {utility.get("task") for utility in utilities.values()}
    if len(tasks) != 1:
        raise ValueError(f"Real metric caches disagree on utility task: {sorted(tasks)}")
    task = str(next(iter(tasks)))
    score_name = score_name_for_task(task)
    model_sets = [set(utility["models"]) for utility in utilities.values()]
    shared_models = set.intersection(*model_sets) if model_sets else set()
    if not shared_models:
        raise ValueError("No utility model appears in every real metric cache")

    candidates = {}
    for model in sorted(shared_models):
        split_scores = {}
        for identifier, utility in utilities.items():
            value = finite_number(
                utility["models"].get(model, {}).get("trtr", {}).get(score_name)
            )
            if value is None:
                raise ValueError(
                    f"Missing finite TRTR {score_name} for {model!r} in "
                    f"{caches[identifier]}"
                )
            split_scores[identifier] = value
        candidates[model] = {
            "split_scores": split_scores,
            **summarize(split_scores.values()),
        }
    selected = min(
        candidates,
        key=lambda model: (-candidates[model]["mean"], model),
    )
    return task, score_name, selected, candidates


def tokenize_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for component in path.split("."):
        name = component.split("[", 1)[0]
        if name:
            tokens.append(name)
        remainder = component[len(name) :]
        while remainder:
            if not remainder.startswith("[") or "]" not in remainder:
                raise ValueError(f"Invalid metric path: {path}")
            index_text, remainder = remainder[1:].split("]", 1)
            tokens.append(int(index_text))
    return tokens


def metric_value(document: Any, path: str) -> float | None:
    current = document
    try:
        for token in tokenize_path(path):
            current = current[token]
    except (KeyError, IndexError, TypeError):
        return None
    return finite_number(current)


def property_labels(evaluations: Sequence[EvaluationInput]) -> dict[str, str | None]:
    for item in evaluations:
        per_file = read_json(item.path).get("per_file")
        if not isinstance(per_file, list) or not per_file:
            continue
        properties = per_file[0].get("quality", {}).get("properties", [])
        labels = {}
        for index in (0, 1):
            label = None
            if index < len(properties) and isinstance(properties[index], dict):
                value = properties[index].get("Property")
                label = str(value) if value is not None else None
            labels[f"quality.properties[{index}].Score"] = label
        return labels
    return {path: None for path in BASE_METRICS if path.startswith("quality.")}


def evaluation_records(
    inputs: Sequence[EvaluationInput], metric_paths: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, float | None]]]:
    file_records = []
    split_values: dict[
        tuple[str, str, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for item in inputs:
        document = read_json(item.path)
        per_file = document.get("per_file")
        if not isinstance(per_file, list) or not per_file:
            raise ValueError(f"Evaluation has no nonempty per_file list: {item.path}")
        for sample_index, result in enumerate(per_file):
            if not isinstance(result, dict):
                raise ValueError(f"Invalid per_file entry in {item.path}")
            values = {path: metric_value(result, path) for path in metric_paths}
            file_records.append(
                {
                    "split": item.split_id,
                    "generator": item.generator,
                    "variant": item.variant,
                    "sample_index": sample_index,
                    "evaluation_path": str(item.path),
                    "synthetic_path": result.get("path"),
                    "metrics": values,
                }
            )
            key = (item.split_id, item.generator, item.variant)
            for path, value in values.items():
                if value is not None:
                    split_values[key][path].append(value)

    split_means = {}
    for item in inputs:
        key = (item.split_id, item.generator, item.variant)
        if key in split_means:
            continue
        split_means[key] = {
            path: (
                statistics.fmean(split_values[key].get(path, []))
                if split_values[key].get(path)
                else None
            )
            for path in metric_paths
        }
    return file_records, split_means


def aggregate_variants(
    inputs: Sequence[EvaluationInput],
    metric_paths: Sequence[str],
    split_means: dict[tuple[str, str, str], dict[str, float | None]],
) -> dict[str, Any]:
    split_ids = sorted({item.split_id for item in inputs})
    generators = sorted({item.generator for item in inputs})
    output = {}
    for generator in generators:
        variants = {}
        for variant in VARIANTS:
            metric_summaries = {}
            for path in metric_paths:
                values = {
                    split: split_means[(split, generator, variant)][path]
                    for split in split_ids
                }
                metric_summaries[path] = {
                    "split_means": values,
                    **summarize(values.values()),
                }
            variants[variant] = {"splits": len(split_ids), "metrics": metric_summaries}
        comparisons = {}
        for path in metric_paths:
            raw = variants["raw"]["metrics"][path]
            constrained = variants["constrained"]["metrics"][path]
            delta = (
                constrained["mean"] - raw["mean"]
                if constrained["mean"] is not None and raw["mean"] is not None
                else None
            )
            comparisons[path] = {"constrained_minus_raw": delta}
        output[generator] = {**variants, "comparison": comparisons}
    return output


def aggregate_across_generators(
    inputs: Sequence[EvaluationInput],
    metric_paths: Sequence[str],
    split_means: dict[tuple[str, str, str], dict[str, float | None]],
) -> dict[str, Any]:
    """Average generators equally inside each split, then summarize splits."""
    split_ids = sorted({item.split_id for item in inputs})
    generators = sorted({item.generator for item in inputs})
    variants = {}
    for variant in VARIANTS:
        metric_summaries = {}
        for path in metric_paths:
            generator_averages = {}
            generator_values = {}
            for split in split_ids:
                values = {
                    generator: split_means[(split, generator, variant)][path]
                    for generator in generators
                }
                generator_values[split] = values
                available = [value for value in values.values() if value is not None]
                if available and len(available) != len(generators):
                    missing = [
                        generator
                        for generator, value in values.items()
                        if value is None
                    ]
                    raise ValueError(
                        f"Metric {path!r} is unavailable for only some "
                        f"generators in {split}/{variant}; missing={missing}"
                    )
                generator_averages[split] = (
                    statistics.fmean(available) if available else None
                )
            metric_summaries[path] = {
                "per_split_generator_means": generator_values,
                "split_means_across_generators": generator_averages,
                **summarize(generator_averages.values()),
            }
        variants[variant] = {
            "splits": len(split_ids),
            "generators": len(generators),
            "metrics": metric_summaries,
        }
    comparison = {}
    for path in metric_paths:
        raw = variants["raw"]["metrics"][path]
        constrained = variants["constrained"]["metrics"][path]
        comparison[path] = {
            "constrained_minus_raw": (
                constrained["mean"] - raw["mean"]
                if constrained["mean"] is not None and raw["mean"] is not None
                else None
            )
        }
    return {
        "generators": generators,
        "generator_count": len(generators),
        **variants,
        "comparison": comparison,
    }


def csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    labels = summary["metric_labels"]
    result = summary["across_generators"]
    for metric in summary["metric_paths"]:
        raw = result["raw"]["metrics"][metric]
        constrained = result["constrained"]["metrics"][metric]
        rows.append(
            {
                "scope": "average_across_generators",
                "generators": ",".join(result["generators"]),
                "generator_count": result["generator_count"],
                "metric": metric,
                "label": labels.get(metric),
                "raw_mean": raw["mean"],
                "raw_std": raw["std"],
                "constrained_mean": constrained["mean"],
                "constrained_std": constrained["std"],
                "constrained_minus_raw": result["comparison"][metric][
                    "constrained_minus_raw"
                ],
                "n_splits_raw": raw["n"],
                "n_splits_constrained": constrained["n"],
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def print_summary(summary: dict[str, Any]) -> None:
    selection = summary["utility_model_selection"]
    selected = selection["selected_model"]
    selected_summary = selection["candidates"][selected]
    print(
        f"Selected utility model: {selected} by mean TRTR "
        f"{selection['score']}={selected_summary['mean']:.6f} "
        f"± {selected_summary['std']:.6f} across {selected_summary['n']} splits"
    )
    result = summary["across_generators"]
    print(
        f"\n[average across {result['generator_count']} generators: "
        f"{', '.join(result['generators'])}]"
    )
    print("metric\traw mean ± std\tconstrained mean ± std\tdelta")
    for metric in summary["metric_paths"]:
        raw = result["raw"]["metrics"][metric]
        constrained = result["constrained"]["metrics"][metric]
        delta = result["comparison"][metric]["constrained_minus_raw"]
        print(
            f"{metric}\t{format_number(raw['mean'])} ± {format_number(raw['std'])}"
            f"\t{format_number(constrained['mean'])} ± {format_number(constrained['std'])}"
            f"\t{format_number(delta)}"
        )
def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = args.experiment_dir.expanduser().resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    split_dirs = discover_split_dirs(experiment_dir, args.split_dir)
    evaluations = discover_evaluations(split_dirs, args.evaluation_file)
    caches = resolve_cache_paths(split_dirs, args.real_metrics_cache)
    task, score_name, selected_model, candidates = select_utility_model(caches)
    utility_path = f"utility.models.{selected_model}.tstr.{score_name}"
    metric_paths = (*BASE_METRICS, utility_path)
    file_records, split_means = evaluation_records(evaluations, metric_paths)
    labels = property_labels(evaluations)
    labels["constraint.linear_feasibility_distance"] = (
        "Linear Feasibility Distance (LFD)"
    )
    labels[utility_path] = f"TSTR {score_name} ({selected_model})"
    summary = {
        "schema_version": 1,
        "experiment_dir": str(experiment_dir),
        "aggregation": (
            "within each split: mean over samples per generator, then equal-weight "
            "mean across generators; final mean and sample standard deviation "
            "(ddof=1) across split means; zero for one split"
        ),
        "splits": [directory.name for directory in split_dirs],
        "metric_paths": list(metric_paths),
        "metric_labels": labels,
        "utility_model_selection": {
            "task": task,
            "score": score_name,
            "rule": "highest mean TRTR score across selected splits",
            "selected_model": selected_model,
            "candidates": candidates,
            "cache_paths": {key: str(value) for key, value in caches.items()},
        },
        "across_generators": aggregate_across_generators(
            evaluations, metric_paths, split_means
        ),
        "per_generator": aggregate_variants(evaluations, metric_paths, split_means),
        "file_records": file_records,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        summary = build_summary(args)
        output_json = (
            args.output_json.expanduser().resolve()
            if args.output_json is not None
            else args.experiment_dir.expanduser().resolve() / DEFAULT_JSON_NAME
        )
        write_json(output_json, summary)
        if not args.no_csv:
            output_csv = (
                args.output_csv.expanduser().resolve()
                if args.output_csv is not None
                else args.experiment_dir.expanduser().resolve() / DEFAULT_CSV_NAME
            )
            write_csv(output_csv, csv_rows(summary))
        print_summary(summary)
        print(f"\nWrote JSON summary to {output_json}")
        if not args.no_csv:
            print(f"Wrote CSV summary to {output_csv}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
