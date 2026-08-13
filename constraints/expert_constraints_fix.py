"""Apply every available expert-constraint repair in a single pass.

The constraints directory is searched for
``categorical_dependency_constraint.json``, ``equational_constraint.json``, and
``linear_constraint.json``. Each present family is applied once in that order.
A missing file or an empty constraint list skips its stage; at least one
supported file must exist.

The source experiment must contain ``train.csv``, ``test.csv``,
``metadata.json``, and a ``synthetic`` directory. Repaired CSVs retain their
source names under the target experiment's ``synthetic`` directory. The three
experiment artifacts are copied to the target root. Exact input paths may be
overridden independently.

Example:
    uv run python constraints/expert_constraints_fix.py \
        experiments/flights/tvae/unconstrained \
        experiments/flights/tvae/expert_fix \
        dataset/flights/constraints_expert
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from constraints.categorical.fix import fix_dataframe as fix_categorical
from constraints.equational.fix import (
    DEFAULT_EQUATIONAL_STRATEGY,
    EQUATIONAL_STRATEGIES,
    equational_fix,
)
from constraints.equational.scoring import load_constraints, load_csv
from constraints.linear.projector import project_dataframe
from constraints.linear.schema import LinearConstraint
from constraints.linear.schema import load_constraints as load_linear_constraints
from evaluation.metrics.categorical_constraint import (
    load_constraint_document,
    resolve_csv_files,
)


CATEGORICAL_CONSTRAINT_NAME = "categorical_dependency_constraint.json"
EQUATIONAL_CONSTRAINT_NAME = "equational_constraint.json"
LINEAR_CONSTRAINT_NAME = "linear_constraint.json"
FIXED_SUFFIX = "_expert_constraints_fixed"
REPORT_SUFFIX = "_expert_constraints_fix_report"
SKIPPED_STAGE = {"skipped": True, "reason": "constraint file missing"}
EMPTY_STAGE = {"skipped": True, "reason": "constraint list empty"}
SYNTHETIC_DIRECTORY_NAME = "synthetic"
FIX_REPORT_DIRECTORY_NAME = "fix_report"
EXPERIMENT_ARTIFACT_NAMES = ("train.csv", "test.csv", "metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_experiment",
        type=Path,
        help=(
            "Unconstrained experiment directory containing train.csv, test.csv, "
            "metadata.json, and synthetic/."
        ),
    )
    parser.add_argument(
        "target_experiment",
        type=Path,
        help="Target fixed experiment directory.",
    )
    parser.add_argument(
        "constraints_expert",
        type=Path,
        help=(
            "Directory searched for categorical, equational, and linear "
            "constraint JSON files. Missing files skip that repair stage."
        ),
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=None,
        help=(
            "Synthetic CSV or directory override. Defaults to "
            "<source_experiment>/synthetic."
        ),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help="Training CSV override. Defaults to <source_experiment>/train.csv.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Test CSV override. Defaults to <source_experiment>/test.csv.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Metadata JSON override. Defaults to "
            "<source_experiment>/metadata.json."
        ),
    )
    parser.add_argument(
        "--output-synthetic",
        "--output",
        dest="output_synthetic",
        type=Path,
        default=None,
        help=(
            "Output CSV path for one synthetic input or output directory. "
            "Defaults to <target_experiment>/synthetic."
        ),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help=(
            "Output JSON path for one synthetic input or report directory. "
            "Defaults to <target_experiment>/fix_report."
        ),
    )
    parser.add_argument(
        "--invalid-row-policy",
        choices=("error", "drop"),
        default="error",
        help=(
            "How to handle rows that remain invalid after equational or linear "
            "repair. Use 'drop' to remove mathematically unsatisfiable or "
            "numerically residual rows."
        ),
    )
    parser.add_argument(
        "--max-drop-fraction",
        type=float,
        default=0.05,
        help=(
            "Maximum cumulative fraction of input rows that --invalid-row-policy "
            "drop may remove (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--equational-strategy",
        choices=EQUATIONAL_STRATEGIES,
        default=DEFAULT_EQUATIONAL_STRATEGY,
        help=(
            "Equational repair planner (default: static-global). Use "
            "dynamic-greedy for the legacy stepwise strategy."
        ),
    )
    parser.add_argument(
        "--linear-scale-mode",
        choices=("none", "std"),
        default="std",
        help="Linear projection objective scaling (default: std).",
    )
    parser.add_argument(
        "--linear-solver",
        default="OSQP",
        help="CVXPY solver used for linear projection (default: OSQP).",
    )
    parser.add_argument("--linear-batch-size", type=int, default=1000)
    parser.add_argument("--linear-tolerance", type=float, default=1e-7)
    return parser.parse_args()


def _require_file(path: Path, label: str, suffix: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if path.suffix.lower() != suffix:
        raise ValueError(f"{label} must be a {suffix} file: {path}")
    return path


def experiment_input_paths(
    source_experiment: Path,
    synthetic: Path | None = None,
    train: Path | None = None,
    test: Path | None = None,
    metadata: Path | None = None,
) -> dict[str, Path]:
    """Resolve default experiment inputs while honoring exact-path overrides."""
    if not source_experiment.is_dir():
        raise FileNotFoundError(
            f"Source experiment directory not found: {source_experiment}"
        )

    synthetic_path = synthetic or source_experiment / SYNTHETIC_DIRECTORY_NAME
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic data path not found: {synthetic_path}")
    train_path = _require_file(
        train or source_experiment / "train.csv",
        "Training",
        ".csv",
    )
    test_path = _require_file(
        test or source_experiment / "test.csv",
        "Test",
        ".csv",
    )
    metadata_path = _require_file(
        metadata or source_experiment / "metadata.json",
        "Metadata",
        ".json",
    )
    return {
        "synthetic": synthetic_path,
        "train": train_path,
        "test": test_path,
        "metadata": metadata_path,
    }


def constraint_paths(
    constraints_expert: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    if not constraints_expert.is_dir():
        raise FileNotFoundError(
            f"Expert constraints directory not found: {constraints_expert}"
        )
    categorical = constraints_expert / CATEGORICAL_CONSTRAINT_NAME
    equational = constraints_expert / EQUATIONAL_CONSTRAINT_NAME
    linear = constraints_expert / LINEAR_CONSTRAINT_NAME
    categorical_path = categorical if categorical.is_file() else None
    equational_path = equational if equational.is_file() else None
    linear_path = linear if linear.is_file() else None
    if all(
        path is None
        for path in (categorical_path, equational_path, linear_path)
    ):
        raise FileNotFoundError(
            f"Expert constraints directory {constraints_expert} contains none of "
            f"{CATEGORICAL_CONSTRAINT_NAME}, {EQUATIONAL_CONSTRAINT_NAME}, or "
            f"{LINEAR_CONSTRAINT_NAME}"
        )
    return categorical_path, equational_path, linear_path


def _load_linear_constraints_for_repair(
    path: Path,
) -> list[LinearConstraint]:
    """Load linear constraints, treating an explicit empty list as no stage."""
    document = json.loads(path.read_text(encoding="utf-8"))
    constraint_list = (
        document.get("constraints") if isinstance(document, dict) else document
    )
    if constraint_list == []:
        return []
    return load_linear_constraints(path)


def _single_output_path(
    source: Path,
    requested: Path | None,
    suffix: str,
    extension: str,
) -> Path:
    default = source.with_name(f"{source.stem}{suffix}{extension}")
    if requested is None:
        return default
    if requested.suffix.lower() == extension:
        return requested
    return requested / default.name


def output_paths(
    input_path: Path,
    csv_files: list[Path],
    output: Path | None,
    output_report: Path | None,
) -> list[tuple[Path, Path, Path]]:
    if input_path.is_file():
        source = csv_files[0]
        return [
            (
                source,
                _single_output_path(source, output, FIXED_SUFFIX, ".csv"),
                _single_output_path(
                    source, output_report, REPORT_SUFFIX, ".json"
                ),
            )
        ]

    if output is not None and output.suffix.lower() == ".csv":
        raise ValueError("--output must be a directory for directory input.")
    if output_report is not None and output_report.suffix.lower() == ".json":
        raise ValueError(
            "--output-report must be a directory for directory input."
        )

    csv_root = output or input_path.with_name(f"{input_path.name}{FIXED_SUFFIX}")
    report_root = output_report or input_path.with_name(
        f"{input_path.name}{REPORT_SUFFIX}"
    )
    return [
        (
            source,
            csv_root / source.relative_to(input_path),
            report_root / source.relative_to(input_path).with_suffix(".json"),
        )
        for source in csv_files
    ]


def experiment_output_paths(
    input_path: Path,
    csv_files: list[Path],
    output_synthetic: Path,
    output_report: Path,
) -> list[tuple[Path, Path, Path]]:
    """Route repaired files into a target experiment without renaming them."""
    if input_path.is_file():
        source = csv_files[0]
        output_csv = (
            output_synthetic
            if output_synthetic.suffix.lower() == ".csv"
            else output_synthetic / source.name
        )
        report = (
            output_report
            if output_report.suffix.lower() == ".json"
            else output_report / source.with_suffix(".json").name
        )
        return [(source, output_csv, report)]

    return output_paths(
        input_path,
        csv_files,
        output_synthetic,
        output_report,
    )


def repair_dataframe(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    categorical_document: dict[str, Any] | None,
    equational_constraints: list[dict[str, Any]] | None,
    linear_constraints: list[LinearConstraint] | None = None,
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
    equational_strategy: str = DEFAULT_EQUATIONAL_STRATEGY,
    linear_scale_mode: str = "std",
    linear_solver: str = "OSQP",
    linear_batch_size: int = 1000,
    linear_tolerance: float = 1e-7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = synthetic
    stage_order: list[str] = []
    if invalid_row_policy not in {"error", "drop"}:
        raise ValueError(
            "invalid_row_policy must be either 'error' or 'drop'; "
            f"got {invalid_row_policy!r}."
        )
    if not 0.0 <= max_drop_fraction <= 1.0:
        raise ValueError(
            "max_drop_fraction must be between 0 and 1 inclusive; "
            f"got {max_drop_fraction}."
        )

    stages: dict[str, Any] = {}
    policy_rows_dropped = 0

    if categorical_document is not None:
        current, categorical_report = fix_categorical(
            current,
            categorical_document,
            reference_data=train,
        )
        stage_order.append("categorical_dependency")
        stages["categorical_dependency"] = categorical_report
    else:
        stages["categorical_dependency"] = dict(SKIPPED_STAGE)

    if equational_constraints:
        current, equational_report = equational_fix(
            current,
            train,
            equational_constraints,
            invalid_row_policy=invalid_row_policy,
            max_drop_fraction=max_drop_fraction,
            strategy=equational_strategy,
        )
        policy_rows_dropped = equational_report["rows_dropped"]
        stage_order.append("greedy_equational")
        stages["greedy_equational"] = equational_report
    else:
        stages["greedy_equational"] = dict(
            SKIPPED_STAGE if equational_constraints is None else EMPTY_STAGE
        )

    if linear_constraints:
        equational_columns = {
            column
            for constraint in equational_constraints or []
            for column in constraint["columns"]
        }
        protected_linear_columns = sorted(
            equational_columns.intersection(
                column
                for constraint in linear_constraints
                for column in constraint.mutable_columns
            )
        )
        protected_linear_constraints = [
            replace(
                constraint,
                mutable_columns=tuple(
                    column
                    for column in constraint.mutable_columns
                    if column not in equational_columns
                ),
            )
            for constraint in linear_constraints
        ]
        linear_input_rows = len(current)
        if invalid_row_policy == "drop" and linear_input_rows:
            remaining_drop_rows = (
                max_drop_fraction * len(synthetic) - policy_rows_dropped
            )
            linear_max_drop_fraction = max(
                0.0,
                min(1.0, remaining_drop_rows / linear_input_rows),
            )
        else:
            linear_max_drop_fraction = max_drop_fraction
        linear_result = project_dataframe(
            current,
            protected_linear_constraints,
            reference_data=train,
            scale_mode=linear_scale_mode,
            solver=linear_solver,
            batch_size=linear_batch_size,
            tolerance=linear_tolerance,
            invalid_row_policy=invalid_row_policy,
            max_drop_fraction=linear_max_drop_fraction,
        )
        linear_result.report["equationally_protected_columns"] = (
            protected_linear_columns
        )
        linear_result.report["mutable_columns_by_constraint"] = {
            constraint.id: list(constraint.mutable_columns)
            for constraint in protected_linear_constraints
        }
        linear_result.report["configured_max_drop_fraction"] = (
            max_drop_fraction
        )
        linear_result.report["rows_dropped_before_linear"] = (
            policy_rows_dropped
        )
        cumulative_rows_dropped = (
            policy_rows_dropped + linear_result.report["rows_dropped"]
        )
        linear_result.report["cumulative_rows_dropped_by_policy"] = (
            cumulative_rows_dropped
        )
        linear_result.report["cumulative_row_drop_fraction"] = (
            cumulative_rows_dropped / len(synthetic) if len(synthetic) else 0.0
        )
        current = linear_result.data
        stage_order.append("linear")
        stages["linear"] = linear_result.report
    else:
        stages["linear"] = dict(
            SKIPPED_STAGE if linear_constraints is None else EMPTY_STAGE
        )

    return current, {
        "algorithm": "single-pass expert constraint repair",
        "stage_order": stage_order,
        "stages": stages,
    }


def _refuse_protected_output(output: Path, protected: list[Path]) -> None:
    resolved_output = output.resolve()
    for path in protected:
        if resolved_output == path.resolve():
            raise ValueError(f"Refusing to overwrite input file: {path}")


def repair_csv(
    source: Path,
    output_csv: Path,
    output_report: Path,
    train_path: Path,
    train: pd.DataFrame,
    constraints_expert: Path,
    categorical_path: Path | None,
    categorical_document: dict[str, Any] | None,
    equational_path: Path | None,
    equational_constraints: list[dict[str, Any]] | None,
    linear_path: Path | None = None,
    linear_constraints: list[LinearConstraint] | None = None,
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
    equational_strategy: str = DEFAULT_EQUATIONAL_STRATEGY,
    linear_scale_mode: str = "std",
    linear_solver: str = "OSQP",
    linear_batch_size: int = 1000,
    linear_tolerance: float = 1e-7,
) -> dict[str, Any]:
    protected_csv = [source, train_path]
    protected_report = [source, train_path]
    if categorical_path is not None:
        protected_report.append(categorical_path)
    if equational_path is not None:
        protected_report.append(equational_path)
    if linear_path is not None:
        protected_report.append(linear_path)

    _refuse_protected_output(output_csv, protected_csv)
    _refuse_protected_output(output_report, protected_report)

    synthetic = load_csv(source, "Synthetic")
    repaired, report = repair_dataframe(
        synthetic,
        train,
        categorical_document,
        equational_constraints,
        linear_constraints,
        invalid_row_policy=invalid_row_policy,
        max_drop_fraction=max_drop_fraction,
        equational_strategy=equational_strategy,
        linear_scale_mode=linear_scale_mode,
        linear_solver=linear_solver,
        linear_batch_size=linear_batch_size,
        linear_tolerance=linear_tolerance,
    )
    report.update(
        {
            "source_csv": str(source),
            "output_csv": str(output_csv),
            "output_report": str(output_report),
            "training_csv": str(train_path),
            "constraints_expert_directory": str(constraints_expert),
            "equational_strategy": equational_strategy,
            "categorical_constraints": (
                str(categorical_path) if categorical_path is not None else None
            ),
            "equational_constraints": (
                str(equational_path) if equational_path is not None else None
            ),
            "linear_constraints": (
                str(linear_path) if linear_path is not None else None
            ),
            "rows": int(len(synthetic)),
            "input_rows": int(len(synthetic)),
            "output_rows": int(len(repaired)),
            "rows_dropped": int(len(synthetic) - len(repaired)),
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(output_csv, index=False)
    output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def print_report(report: dict[str, Any], index: int, total: int) -> None:
    categorical = report["stages"]["categorical_dependency"]
    equational = report["stages"]["greedy_equational"]
    linear = report["stages"]["linear"]
    print(f"Processed {index}/{total}: {report['source_csv']}")
    if categorical.get("skipped"):
        print("  Categorical repair: skipped (file missing)")
    else:
        print(
            "  Categorical rows with violations: "
            f"{categorical['before']['rows_with_any_violation']:,} -> "
            f"{categorical['after']['rows_with_any_violation']:,}"
        )
    if equational.get("skipped"):
        print(
            f"  Equational repair: skipped ({equational['reason']})"
        )
    else:
        print(
            "  Equational constraints resolved: "
            f"{equational['constraints_resolved']:,} "
            f"(strategy={equational['strategy']})"
        )
        planning = equational.get("planning")
        if planning is not None and planning["optimal_threshold"] is not None:
            print(
                "  Predicted KS threshold: "
                f"{planning['optimal_threshold']:.12g}; "
                "predicted total delta: "
                f"{planning['predicted_total_delta_ks_complement']:.12g}"
            )
        print(
            "  Rows: "
            f"{equational['input_rows']:,} -> {equational['output_rows']:,} "
            f"(dropped {equational['rows_dropped']:,})"
        )
    if linear.get("skipped"):
        print(f"  Linear repair: skipped ({linear['reason']})")
    else:
        print(
            "  Linear CVR: "
            f"{linear['before']['cvr']:.3%} -> {linear['after']['cvr']:.3%}"
        )
        print(
            "  Linear rows: "
            f"{linear['input_rows']:,} -> {linear['output_rows']:,} "
            f"(dropped {linear['rows_dropped']:,})"
        )

    print(f"  Output: {report['output_csv']}")
    print(f"  Report: {report['output_report']}")


def copy_experiment_artifacts(
    target_experiment: Path,
    inputs: dict[str, Path],
) -> dict[str, str]:
    """Copy train, test, and metadata inputs to canonical target filenames."""
    target_experiment.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in EXPERIMENT_ARTIFACT_NAMES:
        key = Path(name).stem
        source = inputs[key]
        destination = target_experiment / name
        _refuse_protected_output(destination, [source])
        shutil.copy2(source, destination)
        copied[name] = str(destination)
    return copied


def repair_experiment(
    source_experiment: Path,
    target_experiment: Path,
    constraints_expert: Path,
    synthetic: Path | None = None,
    train: Path | None = None,
    test: Path | None = None,
    metadata: Path | None = None,
    output_synthetic: Path | None = None,
    output_report: Path | None = None,
    invalid_row_policy: str = "error",
    max_drop_fraction: float = 0.05,
    equational_strategy: str = DEFAULT_EQUATIONAL_STRATEGY,
    linear_scale_mode: str = "std",
    linear_solver: str = "OSQP",
    linear_batch_size: int = 1000,
    linear_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Repair all synthetic CSVs and construct the target experiment."""
    if source_experiment.resolve() == target_experiment.resolve():
        raise ValueError("Source and target experiment directories must differ.")

    inputs = experiment_input_paths(
        source_experiment,
        synthetic=synthetic,
        train=train,
        test=test,
        metadata=metadata,
    )
    categorical_path, equational_path, linear_path = constraint_paths(
        constraints_expert
    )
    categorical_document = (
        load_constraint_document(categorical_path)
        if categorical_path is not None
        else None
    )
    equational_constraints = (
        load_constraints(equational_path)
        if equational_path is not None
        else None
    )
    linear_constraints = (
        _load_linear_constraints_for_repair(linear_path)
        if linear_path is not None
        else None
    )
    train_data = load_csv(inputs["train"], "Training")
    csv_files = resolve_csv_files(inputs["synthetic"])
    synthetic_output = output_synthetic or (
        target_experiment / SYNTHETIC_DIRECTORY_NAME
    )
    report_output = output_report or (
        target_experiment / FIX_REPORT_DIRECTORY_NAME
    )
    destinations = experiment_output_paths(
        inputs["synthetic"],
        csv_files,
        synthetic_output,
        report_output,
    )

    protected = list(inputs.values()) + [
        target_experiment / name for name in EXPERIMENT_ARTIFACT_NAMES
    ]
    for _, output_csv, report_path in destinations:
        _refuse_protected_output(output_csv, protected)
        _refuse_protected_output(report_path, protected)

    reports: list[dict[str, Any]] = []
    for index, (source, output_csv, report_path) in enumerate(
        destinations,
        start=1,
    ):
        report = repair_csv(
            source,
            output_csv,
            report_path,
            inputs["train"],
            train_data,
            constraints_expert,
            categorical_path,
            categorical_document,
            equational_path,
            equational_constraints,
            linear_path,
            linear_constraints,
            invalid_row_policy=invalid_row_policy,
            max_drop_fraction=max_drop_fraction,
            equational_strategy=equational_strategy,
            linear_scale_mode=linear_scale_mode,
            linear_solver=linear_solver,
            linear_batch_size=linear_batch_size,
            linear_tolerance=linear_tolerance,
        )
        reports.append(report)
        print_report(report, index, len(destinations))

    copied_artifacts = copy_experiment_artifacts(target_experiment, inputs)
    return {
        "source_experiment": str(source_experiment),
        "target_experiment": str(target_experiment),
        "constraints_expert": str(constraints_expert),
        "inputs": {key: str(path) for key, path in inputs.items()},
        "output_synthetic": str(synthetic_output),
        "output_report": str(report_output),
        "copied_artifacts": copied_artifacts,
        "reports": reports,
    }


def main() -> None:
    args = parse_args()
    result = repair_experiment(
        source_experiment=args.source_experiment,
        target_experiment=args.target_experiment,
        constraints_expert=args.constraints_expert,
        synthetic=args.synthetic,
        train=args.train,
        test=args.test,
        metadata=args.metadata,
        output_synthetic=args.output_synthetic,
        output_report=args.output_report,
        invalid_row_policy=args.invalid_row_policy,
        max_drop_fraction=args.max_drop_fraction,
        equational_strategy=args.equational_strategy,
        linear_scale_mode=args.linear_scale_mode,
        linear_solver=args.linear_solver,
        linear_batch_size=args.linear_batch_size,
        linear_tolerance=args.linear_tolerance,
    )
    print(
        "Copied experiment artifacts: "
        + ", ".join(result["copied_artifacts"].values())
    )


if __name__ == "__main__":
    main()
