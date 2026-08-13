"""Run reproducible multi-split tabular synthesis experiments.

The manager freezes train/test rows once, discovers constraints from training
rows only, trains configured generators, repairs their samples, evaluates raw
and repaired data against the proposed constraints, and aggregates every file.
Run every command from the repository root through ``uv run python``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_NAME = "experiment_config.json"
DEFAULT_SPLIT_SEEDS = [42, 43, 44]
DEFAULT_TRAINING_SEEDS = [2000, 2001, 2002]
DEFAULT_SAMPLE_SEEDS = [1000, 1001, 1002]
MULTIRUN_SYNTHETIC_UTILITY_REGIMES = ("tstr",)
EQUATIONAL_REPAIR_STRATEGIES = {"static-global", "dynamic-greedy"}
GENERATOR_SCRIPTS = {
    "ctgan": "run_ctgan.py",
    "tvae": "run_tvae.py",
    "gaussian_copula": "run_GaussianCopula.py",
    "tabddpm": "run_TabDDPM.py",
}
GENERATOR_CHECKPOINTS = {
    "ctgan": ["ctgan.pkl"],
    "tvae": ["tvae.pkl"],
    "tabddpm": ["tabddpm.pt", "tabddpm_preprocessor.pkl"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically write stable, readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(value: str | Path | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def default_config(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    return {
        "schema_version": 1,
        "dataset": {
            "name": dataset_dir.name,
            "directory": portable_path(dataset_dir),
            "data_file": portable_path(dataset_dir / "data.csv"),
            "info_file": portable_path(dataset_dir / "info.json"),
            "utility_feature_file": portable_path(
                dataset_dir / "utility_feature.json"
            ),
            "meta_file": portable_path(dataset_dir / "meta.json"),
        },
        "splitting": {
            "test_size": 0.3,
            "seeds": list(DEFAULT_SPLIT_SEEDS),
            "stratify_classification": True,
        },
        "agentic_pipelines": {
            "categorical": {
                "enabled": True,
                "model": "gpt-5.6-luna",
                "provider": "openai",
                "arguments": {},
            },
            "equational": {
                "enabled": True,
                "model": "gpt-5.6-luna",
                "provider": "openai",
                "generate_fixes": True,
                "arguments": {},
            },
            "linear": {
                "enabled": True,
                "model": "gpt-5.6-luna",
                "provider": "openai",
                "arguments": {},
            },
        },
        "generators": {
            "ctgan": {
                "enabled": True,
                "model": "CTGAN",
                "training_arguments": {"epochs": 300, "device": "auto"},
                "sampling_arguments": {},
            },
            "tvae": {
                "enabled": True,
                "model": "TVAE",
                "training_arguments": {"epochs": 300, "device": "auto"},
                "sampling_arguments": {},
            },
            "gaussian_copula": {
                "enabled": True,
                "model": "GaussianCopulaSynthesizer",
                "training_arguments": {"default_distribution": "beta"},
                "sampling_arguments": {},
            },
            "tabddpm": {
                "enabled": True,
                "model": "TabDDPM",
                "training_arguments": {"steps": 30000, "device": "auto"},
                "sampling_arguments": {"checkpoint_variant": "ema"},
            },
        },
        "runs": {
            "training_seeds": list(DEFAULT_TRAINING_SEEDS),
            "sample_seeds": list(DEFAULT_SAMPLE_SEEDS),
            "samples_per_generator": 3,
        },
        "postprocessing": {
            "equational_strategy": "static-global",
            "invalid_row_policy": "drop",
            "max_drop_fraction": 0.05,
            "linear_scale_mode": "std",
            "linear_solver": "OSQP",
            "linear_batch_size": 1000,
            "linear_tolerance": 1e-7,
        },
        "evaluation": {
            "metrics": ["quality", "utility", "constraint"],
            "constraint_source": "proposed",
        },
    }


def dataset_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset = config["dataset"]
    directory = resolve_path(dataset.get("directory"))
    if directory is None:
        raise ValueError("dataset.directory is required")
    paths = {
        "directory": directory,
        "data": resolve_path(dataset.get("data_file"), directory / "data.csv"),
        "info": resolve_path(dataset.get("info_file"), directory / "info.json"),
        "utility": resolve_path(
            dataset.get("utility_feature_file"), directory / "utility_feature.json"
        ),
        "meta": resolve_path(dataset.get("meta_file"), directory / "meta.json"),
    }
    return {name: path for name, path in paths.items() if path is not None}


def agentic_pipeline_configs(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return all three family configs, upgrading older two-family configs."""
    configured = config.get("agentic_pipelines", {})
    result = {
        family: dict(settings)
        for family, settings in configured.items()
        if family in {"categorical", "equational", "linear"}
        and isinstance(settings, dict)
    }
    if "categorical" not in result:
        fallback = next(
            (
                result[family]
                for family in ("equational", "linear")
                if result.get(family, {}).get("enabled", False)
            ),
            {},
        )
        result["categorical"] = {
            "enabled": True,
            "model": fallback.get("model", "gpt-5.6-luna"),
            "provider": fallback.get("provider", "openai"),
            "arguments": {},
        }
    return result


def validate_config(config: dict[str, Any], *, require_meta: bool = False) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("experiment_config.json schema_version must be 1")
    split = config.get("splitting", {})
    test_size = split.get("test_size")
    if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError("splitting.test_size must be between 0 and 1")
    seeds = split.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(
        isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("splitting.seeds must be a non-empty integer list")
    if len(seeds) != len(set(seeds)):
        raise ValueError("splitting.seeds must not contain duplicates")

    runs = config.get("runs", {})
    training_seeds = runs.get("training_seeds")
    sample_seeds = runs.get("sample_seeds")
    sample_count = runs.get("samples_per_generator")
    if not isinstance(training_seeds, list) or len(training_seeds) != len(seeds):
        raise ValueError(
            "runs.training_seeds must contain one seed for every split"
        )
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("runs.samples_per_generator must be positive")
    if not isinstance(sample_seeds, list) or len(sample_seeds) != sample_count:
        raise ValueError(
            "runs.sample_seeds must contain samples_per_generator seeds"
        )
    if sample_seeds != list(range(sample_seeds[0], sample_seeds[0] + sample_count)):
        raise ValueError(
            "runs.sample_seeds must be consecutive because generator runners "
            "use base_seed + file_index"
        )

    pipelines = agentic_pipeline_configs(config)
    for family in ("categorical", "equational", "linear"):
        model = pipelines.get(family, {})
        if not model.get("enabled", False):
            raise ValueError(
                f"agentic_pipelines.{family}.enabled must be true because "
                "multi-run discovery generates all three families"
            )
        if not model.get("model"):
            raise ValueError(f"agentic_pipelines.{family}.model is required")
    if not pipelines["equational"].get("generate_fixes", True):
        raise ValueError(
            "Equational fix generation must be enabled for postprocessing"
        )
    invalid_row_policy = config.get("postprocessing", {}).get(
        "invalid_row_policy", "drop"
    )
    if invalid_row_policy not in {"error", "drop"}:
        raise ValueError(
            "postprocessing.invalid_row_policy must be either 'error' or 'drop'"
        )
    equational_strategy = config.get("postprocessing", {}).get(
        "equational_strategy", "static-global"
    )
    if equational_strategy not in EQUATIONAL_REPAIR_STRATEGIES:
        raise ValueError(
            "postprocessing.equational_strategy must be either "
            "'static-global' or 'dynamic-greedy'"
        )

    generators = config.get("generators", {})
    unknown = sorted(set(generators) - set(GENERATOR_SCRIPTS))
    if unknown:
        raise ValueError(f"Unsupported generators: {unknown}")
    for name, model in generators.items():
        if model.get("enabled", False) and not model.get("model"):
            raise ValueError(f"generators.{name}.model is required")

    paths = dataset_paths(config)
    required = ["data", "info", "utility"] + (["meta"] if require_meta else [])
    for name in required:
        if not paths[name].is_file():
            raise FileNotFoundError(f"Dataset {name} file not found: {paths[name]}")


def split_name(index: int, seed: int) -> str:
    return f"split_{index:02d}_seed_{seed}"


def expected_split_dirs(experiment_dir: Path, config: dict[str, Any]) -> list[Path]:
    return [
        experiment_dir / "splits" / split_name(index, seed)
        for index, seed in enumerate(config["splitting"]["seeds"])
    ]


def _classification_target(
    target: pd.Series,
    info: dict[str, Any],
    target_name: str,
) -> bool:
    declared = str(info.get("task") or info.get("task_type") or "").lower()
    if "regression" in declared:
        return False
    if "classification" in declared or declared in {"binary", "multiclass"}:
        return True
    col_types = info.get("col_types")
    if isinstance(col_types, dict):
        target_info = col_types.get(target_name)
        if isinstance(target_info, dict) and target_info.get("type") == "cat":
            return True
    if not pd.api.types.is_numeric_dtype(target):
        return True
    return 1 < target.nunique(dropna=False) <= 20


def _stratification(
    data: pd.DataFrame,
    target_name: str,
    test_size: float,
    enabled: bool,
    classification: bool,
) -> tuple[pd.Series | None, str | None]:
    if not enabled:
        return None, "disabled_by_config"
    if not classification:
        return None, "regression_target"
    counts = data[target_name].value_counts(dropna=False)
    test_rows = int(round(len(data) * test_size))
    train_rows = len(data) - test_rows
    if len(counts) < 2:
        return None, "target_has_one_class"
    if counts.min() < 2:
        return None, "class_with_fewer_than_two_rows"
    if len(counts) > min(test_rows, train_rows):
        return None, "too_many_classes_for_split_sizes"
    return data[target_name], None


def _split_fingerprint(config: dict[str, Any], source_hash: str) -> str:
    return sha256_json(
        {
            "source_sha256": source_hash,
            "test_size": config["splitting"]["test_size"],
            "seeds": config["splitting"]["seeds"],
            "stratify_classification": config["splitting"].get(
                "stratify_classification", True
            ),
        }
    )


def prepare_splits(
    experiment_dir: Path,
    config: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    validate_config(config)
    paths = dataset_paths(config)
    data = pd.read_csv(paths["data"])
    if data.empty:
        raise ValueError("Dataset must contain at least one row")
    missing = data.columns[data.isna().any()].tolist()
    if missing:
        raise ValueError(f"Dataset contains missing values in columns: {missing}")
    info = read_json(paths["info"])
    utility = read_json(paths["utility"])
    target_name = utility.get("target_column") or utility.get("target")
    if not isinstance(target_name, str) or target_name not in data.columns:
        raise ValueError("utility_feature.json must name a target present in data.csv")

    source_hash = sha256_file(paths["data"])
    fingerprint = _split_fingerprint(config, source_hash)
    test_size = float(config["splitting"]["test_size"])
    classification = _classification_target(data[target_name], info, target_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / CONFIG_NAME, config)

    for index, seed in enumerate(config["splitting"]["seeds"]):
        split_dir = experiment_dir / "splits" / split_name(index, seed)
        train_path = split_dir / "train.csv"
        test_path = split_dir / "test.csv"
        manifest_path = split_dir / "split_manifest.json"
        if manifest_path.is_file() and train_path.is_file() and test_path.is_file():
            manifest = read_json(manifest_path)
            current = (
                manifest.get("split_fingerprint") == fingerprint
                and manifest.get("split_seed") == seed
                and manifest.get("train", {}).get("sha256")
                == sha256_file(train_path)
                and manifest.get("test", {}).get("sha256")
                == sha256_file(test_path)
            )
            if current and not force:
                print(f"[prepare] current: {split_dir}", flush=True)
                continue
            if not force:
                raise RuntimeError(
                    f"Existing split differs from its configuration: {split_dir}. "
                    "Use --force to replace this exact split."
                )

        stratify, fallback_reason = _stratification(
            data,
            target_name,
            test_size,
            bool(config["splitting"].get("stratify_classification", True)),
            classification,
        )
        train, test = train_test_split(
            data,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        train.reset_index(drop=True).to_csv(train_path, index=False)
        test.reset_index(drop=True).to_csv(test_path, index=False)
        manifest = {
            "schema_version": 1,
            "dataset": config["dataset"]["name"],
            "split_index": index,
            "split_seed": seed,
            "test_size": test_size,
            "task_type": "classification" if classification else "regression",
            "stratified": stratify is not None,
            "stratification_fallback_reason": fallback_reason,
            "split_fingerprint": fingerprint,
            "source": {
                "path": portable_path(paths["data"]),
                "sha256": source_hash,
                "rows": len(data),
            },
            "train": {
                "path": "train.csv",
                "sha256": sha256_file(train_path),
                "rows": len(train),
            },
            "test": {
                "path": "test.csv",
                "sha256": sha256_file(test_path),
                "rows": len(test),
            },
            "created_at": utc_now(),
        }
        write_json(manifest_path, manifest)
        print(f"[prepare] wrote: {split_dir}", flush=True)


def verify_split(split_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(split_dir / "split_manifest.json")
    paths = dataset_paths(config)
    source_hash = sha256_file(paths["data"])
    if manifest.get("split_fingerprint") != _split_fingerprint(config, source_hash):
        raise RuntimeError(f"Split configuration or source changed: {split_dir}")
    for name in ("train", "test"):
        path = split_dir / f"{name}.csv"
        if not path.is_file() or sha256_file(path) != manifest[name]["sha256"]:
            raise RuntimeError(f"Frozen {name} split hash mismatch: {path}")
    return manifest


def _option_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def append_options(
    command: list[str],
    options: dict[str, Any] | None,
    *,
    reserved: Iterable[str] = (),
) -> None:
    reserved_set = set(reserved)
    for name, value in (options or {}).items():
        if name in reserved_set:
            raise ValueError(f"Configuration option {name!r} is managed by multirun")
        option = _option_name(name)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif isinstance(value, list):
            command.extend([option, ",".join(str(item) for item in value)])
        else:
            command.extend([option, str(value)])


def _command_current(
    manifest_path: Path,
    fingerprint: str,
    outputs: list[Path],
) -> bool:
    if not manifest_path.is_file() or not all(path.exists() for path in outputs):
        return False
    manifest = read_json(manifest_path)
    return (
        manifest.get("status") == "complete"
        and manifest.get("fingerprint") == fingerprint
    )


def run_command(
    command: list[str],
    manifest_path: Path,
    *,
    fingerprint_payload: dict[str, Any],
    outputs: list[Path],
    force: bool,
    dry_run: bool = False,
) -> bool:
    fingerprint = sha256_json(fingerprint_payload)
    if not force and _command_current(manifest_path, fingerprint, outputs):
        print(f"[resume] current: {manifest_path}", flush=True)
        return False
    record = {
        "command": command,
        "cwd": str(REPO_ROOT),
        "fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_payload,
        "python": sys.version,
        "started_at": utc_now(),
        "status": "dry_run" if dry_run else "running",
    }
    write_json(manifest_path, record)
    print("[command] " + " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except BaseException as exc:
        record["status"] = "failed"
        record["finished_at"] = utc_now()
        record["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, record)
        raise
    record["status"] = "dry_run" if dry_run else "complete"
    record["finished_at"] = utc_now()
    record["outputs"] = [str(path) for path in outputs]
    write_json(manifest_path, record)
    return True


def _all_constraint_command(
    pipeline_configs: dict[str, dict[str, Any]],
    meta_path: Path,
    train_path: Path,
    output_dir: Path,
    split_seed: int,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "agentic_pipeline.generate_constraints",
        str(meta_path),
        str(train_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(split_seed),
    ]
    for family in ("categorical", "equational", "linear"):
        settings = pipeline_configs[family]
        command.extend(
            [
                f"--{family}-model",
                str(settings["model"]),
                f"--{family}-provider",
                str(settings.get("provider", "openai")),
                f"--{family}-arguments",
                json.dumps(
                    settings.get("arguments", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    if dry_run:
        command.append("--dry-run")
    return command


def discover_constraints(
    experiment_dir: Path,
    config: dict[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    validate_config(config, require_meta=True)
    dataset = dataset_paths(config)
    pipeline_configs = agentic_pipeline_configs(config)
    for split_dir in expected_split_dirs(experiment_dir, config):
        split = verify_split(split_dir, config)
        constraints_dir = split_dir / "constraints"
        inputs_dir = constraints_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        local_meta = inputs_dir / "meta.json"
        local_info = inputs_dir / "info.json"
        shutil.copy2(dataset["meta"], local_meta)
        shutil.copy2(dataset["info"], local_info)

        command = _all_constraint_command(
            pipeline_configs,
            local_meta,
            split_dir / "train.csv",
            constraints_dir,
            split["split_seed"],
            dry_run,
        )
        outputs = [
            constraints_dir / filename
            for filename in (
                "categorical_dependency_constraint.json",
                "equational_constraint.json",
                "linear_constraint.json",
            )
        ]
        run_command(
            command,
            constraints_dir / "all_constraints_command.json",
            fingerprint_payload={
                "pipeline_configs": pipeline_configs,
                "family_order": ["categorical", "equational", "linear"],
                "linear_filter": (
                    "drop_if_columns_subset_of_equational_column_union"
                ),
                "split_train_sha256": split["train"]["sha256"],
                "meta_sha256": sha256_file(local_meta),
                "info_sha256": sha256_file(local_info),
                "dry_run": dry_run,
            },
            outputs=[] if dry_run else outputs,
            force=force,
            dry_run=dry_run,
        )


def _common_generator_training_command(
    script: str,
    dataset: dict[str, Path],
    split_dir: Path,
    raw_dir: Path,
    training_seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / script),
        "train",
        "--data-dir",
        str(dataset["directory"]),
        "--data-file",
        str(dataset["data"]),
        "--train-file",
        str(split_dir / "train.csv"),
        "--test-file",
        str(split_dir / "test.csv"),
        "--info-file",
        str(dataset["info"]),
        "--utility-feature-file",
        str(dataset["utility"]),
        "--output-dir",
        str(raw_dir),
        "--seed",
        str(training_seed),
    ]


def _write_generator_manifest(
    raw_dir: Path,
    generator: str,
    model_config: dict[str, Any],
    split: dict[str, Any],
    training_seed: int,
    sample_seeds: list[int],
) -> None:
    samples = []
    for index, seed in enumerate(sample_seeds):
        path = raw_dir / "synthetic" / f"synthetic_{index}.csv"
        samples.append(
            {
                "index": index,
                "seed": seed,
                "path": portable_path(path),
                "rows": len(pd.read_csv(path)),
                "sha256": sha256_file(path),
            }
        )
    write_json(
        raw_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "generator": generator,
            "model_config": model_config,
            "split_index": split["split_index"],
            "split_seed": split["split_seed"],
            "training_seed": training_seed,
            "sample_seeds": sample_seeds,
            "train_sha256": split["train"]["sha256"],
            "test_sha256": split["test"]["sha256"],
            "synthetic": samples,
            "completed_at": utc_now(),
        },
    )


def generate_synthetic(
    experiment_dir: Path,
    config: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    validate_config(config)
    dataset = dataset_paths(config)
    sample_seeds = config["runs"]["sample_seeds"]
    sample_count = config["runs"]["samples_per_generator"]
    for split_position, split_dir in enumerate(expected_split_dirs(experiment_dir, config)):
        split = verify_split(split_dir, config)
        training_seed = config["runs"]["training_seeds"][split_position]
        for generator, model_config in config["generators"].items():
            if not model_config.get("enabled", False):
                continue
            script = GENERATOR_SCRIPTS[generator]
            raw_dir = split_dir / "generators" / generator / "raw"
            expected_samples = [
                raw_dir / "synthetic" / f"synthetic_{index}.csv"
                for index in range(sample_count)
            ]
            reserved = {
                "data_dir", "data_file", "train_file", "test_file", "info_file",
                "utility_feature_file", "output_dir", "seed", "num_files",
                "sample_seed", "experiment_dir",
            }
            if generator == "gaussian_copula":
                command = [
                    sys.executable,
                    str(REPO_ROOT / script),
                    "--data-dir",
                    str(dataset["directory"]),
                    "--data-file",
                    str(dataset["data"]),
                    "--train-file",
                    str(split_dir / "train.csv"),
                    "--test-file",
                    str(split_dir / "test.csv"),
                    "--info-file",
                    str(dataset["info"]),
                    "--utility-feature-file",
                    str(dataset["utility"]),
                    "--output-dir",
                    str(raw_dir),
                    "--seed",
                    str(training_seed),
                    "--sample-seed",
                    str(sample_seeds[0]),
                    "--num-files",
                    str(sample_count),
                ]
                append_options(command, model_config.get("training_arguments"), reserved=reserved)
                append_options(command, model_config.get("sampling_arguments"), reserved=reserved)
                run_command(
                    command,
                    raw_dir / "run_command.json",
                    fingerprint_payload={
                        "generator": generator,
                        "model_config": model_config,
                        "train_sha256": split["train"]["sha256"],
                        "test_sha256": split["test"]["sha256"],
                        "training_seed": training_seed,
                        "sample_seeds": sample_seeds,
                    },
                    outputs=expected_samples + [raw_dir / "metadata.json"],
                    force=force,
                )
            else:
                training_command = _common_generator_training_command(
                    script, dataset, split_dir, raw_dir, training_seed
                )
                append_options(
                    training_command,
                    model_config.get("training_arguments"),
                    reserved=reserved,
                )
                training_outputs = [
                    raw_dir / name for name in GENERATOR_CHECKPOINTS[generator]
                ] + [raw_dir / "metadata.json"]
                run_command(
                    training_command,
                    raw_dir / "training_command.json",
                    fingerprint_payload={
                        "generator": generator,
                        "training_arguments": model_config.get("training_arguments", {}),
                        "train_sha256": split["train"]["sha256"],
                        "test_sha256": split["test"]["sha256"],
                        "training_seed": training_seed,
                    },
                    outputs=training_outputs,
                    force=force,
                )
                sampling_command = [
                    sys.executable,
                    str(REPO_ROOT / script),
                    "sample",
                    "--experiment-dir",
                    str(raw_dir),
                    "--num-files",
                    str(sample_count),
                    "--seed",
                    str(sample_seeds[0]),
                ]
                append_options(
                    sampling_command,
                    model_config.get("sampling_arguments"),
                    reserved=reserved,
                )
                run_command(
                    sampling_command,
                    raw_dir / "sampling_command.json",
                    fingerprint_payload={
                        "generator": generator,
                        "sampling_arguments": model_config.get("sampling_arguments", {}),
                        "training_fingerprint": read_json(
                            raw_dir / "training_command.json"
                        )["fingerprint"],
                        "sample_seeds": sample_seeds,
                        "sample_rows": split["train"]["rows"],
                    },
                    outputs=expected_samples,
                    force=force,
                )
            _write_generator_manifest(
                raw_dir,
                generator,
                model_config,
                split,
                training_seed,
                sample_seeds,
            )


def _postprocessing_reports_complete(
    constrained_dir: Path,
    sample_count: int,
) -> bool:
    report_dir = constrained_dir / "fix_report"
    return all(
        (report_dir / f"synthetic_{index}.json").is_file()
        for index in range(sample_count)
    )


def postprocess_synthetic(
    experiment_dir: Path,
    config: dict[str, Any],
    *,
    force: bool = False,
    continue_completed: bool = False,
) -> None:
    validate_config(config)
    sample_count = config["runs"]["samples_per_generator"]
    for split_dir in expected_split_dirs(experiment_dir, config):
        split = verify_split(split_dir, config)
        constraints_dir = split_dir / "constraints"
        constraint_files = [
            constraints_dir / "categorical_dependency_constraint.json",
            constraints_dir / "equational_constraint.json",
            constraints_dir / "linear_constraint.json",
        ]
        constraint_hashes: dict[str, str] | None = None
        for generator, model_config in config["generators"].items():
            if not model_config.get("enabled", False):
                continue
            generator_dir = split_dir / "generators" / generator
            raw_dir = generator_dir / "raw"
            constrained_dir = generator_dir / "constrained"
            if continue_completed and _postprocessing_reports_complete(
                constrained_dir, sample_count
            ):
                print(
                    f"[continue] fix reports complete: {constrained_dir}",
                    flush=True,
                )
                continue
            if constraint_hashes is None:
                for path in constraint_files:
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Proposed constraint file not found: {path}"
                        )
                constraint_hashes = {
                    path.name: sha256_file(path) for path in constraint_files
                }
            expected = [
                constrained_dir / "synthetic" / f"synthetic_{index}.csv"
                for index in range(sample_count)
            ]
            command = [
                sys.executable,
                "-m",
                "constraints.expert_constraints_fix",
                str(raw_dir),
                str(constrained_dir),
                str(constraints_dir),
            ]
            append_options(command, config.get("postprocessing"))
            run_command(
                command,
                constrained_dir / "postprocessing_command.json",
                fingerprint_payload={
                    "generator_manifest_sha256": sha256_file(
                        raw_dir / "run_manifest.json"
                    ),
                    "constraint_hashes": constraint_hashes,
                    "postprocessing": config.get("postprocessing", {}),
                },
                outputs=expected + [constrained_dir / "metadata.json"],
                force=force,
            )


def _evaluation_command(
    experiment_path: Path,
    constraints_dir: Path,
    real_metrics_cache: Path,
    config: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "evaluation.evaluate_experiment",
        str(experiment_path),
        "--output",
        str(experiment_path / "evaluation.json"),
        "--constraint-details-output",
        str(experiment_path / "constraint_evaluation_details.json"),
        "--real-metrics-cache",
        str(real_metrics_cache),
        "--metrics",
        *[str(metric) for metric in config["evaluation"]["metrics"]],
    ]
    if "utility" in config["evaluation"]["metrics"]:
        command.extend(
            ["--utility-regimes", *MULTIRUN_SYNTHETIC_UTILITY_REGIMES]
        )
    if "constraint" in config["evaluation"]["metrics"]:
        command.extend(["--constraints-expert", str(constraints_dir)])
    return command


def _evaluation_complete(
    experiment_path: Path,
    config: dict[str, Any],
    sample_count: int,
) -> bool:
    evaluation_path = experiment_path / "evaluation.json"
    try:
        evaluation = read_json(evaluation_path)
    except (OSError, ValueError):
        return False

    expected_metrics = {str(metric) for metric in config["evaluation"]["metrics"]}
    if "utility" in expected_metrics:
        expected_regimes = ["trtr", *MULTIRUN_SYNTHETIC_UTILITY_REGIMES]
        if evaluation.get("paths", {}).get("utility_regimes") != expected_regimes:
            return False

    per_file = evaluation.get("per_file")
    if not isinstance(per_file, list) or len(per_file) != sample_count:
        return False
    if not all(
        isinstance(result, dict)
        and isinstance(result.get("path"), str)
        and expected_metrics.issubset(result)
        for result in per_file
    ):
        return False

    expected_names = {
        f"synthetic_{index}.csv" for index in range(sample_count)
    }
    reported_names = {Path(result["path"]).name for result in per_file}
    if reported_names != expected_names:
        return False

    if "constraint" in expected_metrics:
        details_path = experiment_path / "constraint_evaluation_details.json"
        try:
            read_json(details_path)
        except (OSError, ValueError):
            return False
    return True


def flatten_numbers(value: Any, prefix: str = "") -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"path", "details_path"}:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten_numbers(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            output.update(flatten_numbers(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)
    elif value is None and prefix.endswith("linear_feasibility_distance"):
        # Preserve an explicitly unavailable LFD so datasets without linear
        # constraints report N/A at every aggregation level.
        output[prefix] = None
    return output


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for metric, value in flatten_numbers(record["metrics"]).items():
            values.setdefault(metric, [])
            if value is not None:
                values[metric].append(value)
    return {metric: summarize_values(metric_values) for metric, metric_values in sorted(values.items())}


def aggregate_evaluations(
    experiment_dir: Path,
    config: dict[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    sample_seeds = config["runs"]["sample_seeds"]
    for split_dir in expected_split_dirs(experiment_dir, config):
        split = read_json(split_dir / "split_manifest.json")
        for generator, model_config in config["generators"].items():
            if not model_config.get("enabled", False):
                continue
            for variant in ("raw", "constrained"):
                evaluation = read_json(
                    split_dir / "generators" / generator / variant / "evaluation.json"
                )
                per_file = evaluation.get("per_file")
                if not isinstance(per_file, list) or len(per_file) != len(sample_seeds):
                    raise ValueError(
                        f"Evaluation does not contain one per_file result per seed: "
                        f"{split_dir}/{generator}/{variant}"
                    )
                for index, result in enumerate(per_file):
                    metrics = {
                        key: value
                        for key, value in result.items()
                        if key != "path"
                    }
                    records.append(
                        {
                            "split_index": split["split_index"],
                            "split_seed": split["split_seed"],
                            "generator": generator,
                            "variant": variant,
                            "sample_index": index,
                            "sample_seed": sample_seeds[index],
                            "path": result["path"],
                            "metrics": metrics,
                        }
                    )

    within_split = []
    split_mean_records: list[dict[str, Any]] = []
    for key in sorted(
        {(r["split_index"], r["generator"], r["variant"]) for r in records}
    ):
        split_index, generator, variant = key
        group = [
            record
            for record in records
            if (
                record["split_index"], record["generator"], record["variant"]
            ) == key
        ]
        summary = summarize_records(group)
        within_split.append(
            {
                "split_index": split_index,
                "split_seed": group[0]["split_seed"],
                "generator": generator,
                "variant": variant,
                "samples": len(group),
                "metrics": summary,
            }
        )
        split_mean_records.append(
            {
                "generator": generator,
                "variant": variant,
                "metrics": {
                    metric: metric_summary["mean"]
                    for metric, metric_summary in summary.items()
                },
            }
        )

    across_splits = []
    overall = []
    for generator, variant in sorted(
        {(r["generator"], r["variant"]) for r in records}
    ):
        split_group = [
            record
            for record in split_mean_records
            if record["generator"] == generator and record["variant"] == variant
        ]
        across_splits.append(
            {
                "generator": generator,
                "variant": variant,
                "splits": len(split_group),
                "metrics": summarize_records(split_group),
            }
        )
        all_samples = [
            record
            for record in records
            if record["generator"] == generator and record["variant"] == variant
        ]
        overall.append(
            {
                "generator": generator,
                "variant": variant,
                "samples": len(all_samples),
                "metrics": summarize_records(all_samples),
            }
        )
    output = {
        "schema_version": 1,
        "constraint_source": config["evaluation"].get("constraint_source"),
        "standard_deviation": "sample (ddof=1; zero for n=1)",
        "records": records,
        "within_split": within_split,
        "across_splits": across_splits,
        "overall": overall,
        "created_at": utc_now(),
    }
    write_json(
        experiment_dir / "evaluation_summary.json"
        if output_path is None
        else output_path,
        output,
    )
    return output


def evaluate_all(
    experiment_dir: Path,
    config: dict[str, Any],
    *,
    force: bool = False,
    continue_completed: bool = False,
) -> None:
    validate_config(config)
    sample_count = config["runs"]["samples_per_generator"]
    for split_dir in expected_split_dirs(experiment_dir, config):
        split = verify_split(split_dir, config)
        constraints_dir = split_dir / "constraints"
        constraint_names = (
            "categorical_dependency_constraint.json",
            "equational_constraint.json",
            "linear_constraint.json",
        )
        constraint_hashes: dict[str, str] | None = None
        real_metrics_cache = split_dir / "real_metrics_cache.json"
        for generator, model_config in config["generators"].items():
            if not model_config.get("enabled", False):
                continue
            for variant in ("raw", "constrained"):
                experiment_path = split_dir / "generators" / generator / variant
                if continue_completed and _evaluation_complete(
                    experiment_path, config, sample_count
                ):
                    print(
                        f"[continue] evaluation complete: {experiment_path}",
                        flush=True,
                    )
                    continue
                if constraint_hashes is None:
                    constraint_hashes = {
                        name: sha256_file(constraints_dir / name)
                        for name in constraint_names
                    }
                command = _evaluation_command(
                    experiment_path,
                    constraints_dir,
                    real_metrics_cache,
                    config,
                )
                run_command(
                    command,
                    experiment_path / "evaluation_command.json",
                    fingerprint_payload={
                        "variant": variant,
                        "evaluation": config["evaluation"],
                        "synthetic_utility_regimes": (
                            list(MULTIRUN_SYNTHETIC_UTILITY_REGIMES)
                            if "utility" in config["evaluation"]["metrics"]
                            else []
                        ),
                        "constraint_hashes": constraint_hashes,
                        "real_metrics_cache": portable_path(
                            real_metrics_cache
                        ),
                        "synthetic_hashes": [
                            sha256_file(path)
                            for path in sorted(
                                (experiment_path / "synthetic").glob("*.csv")
                            )
                        ],
                        "train_sha256": split["train"]["sha256"],
                        "test_sha256": split["test"]["sha256"],
                    },
                    outputs=[experiment_path / "evaluation.json"],
                    force=force,
                )
    aggregate_evaluations(experiment_dir, config)


def load_experiment_config(experiment_dir: Path) -> dict[str, Any]:
    return read_json(experiment_dir / CONFIG_NAME)


def initialize_config(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    experiment_dir = args.output_dir.resolve()
    config_path = experiment_dir / CONFIG_NAME
    if args.config is not None:
        config = read_json(args.config)
    elif config_path.is_file():
        config = read_json(config_path)
    else:
        if args.dataset_dir is None:
            raise ValueError(
                "--dataset-dir is required when creating a new experiment"
            )
        config = default_config(args.dataset_dir)

    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.resolve()
        config.setdefault("dataset", {})["name"] = dataset_dir.name
        config["dataset"]["directory"] = portable_path(dataset_dir)
    dataset_dir = resolve_path(config["dataset"]["directory"])
    assert dataset_dir is not None
    for argument, key, default_name in (
        (args.data_file, "data_file", "data.csv"),
        (args.info_file, "info_file", "info.json"),
        (
            args.utility_feature_file,
            "utility_feature_file",
            "utility_feature.json",
        ),
        (args.meta_file, "meta_file", "meta.json"),
    ):
        if argument is not None:
            config["dataset"][key] = portable_path(argument)
        elif key not in config["dataset"]:
            config["dataset"][key] = portable_path(dataset_dir / default_name)
    if args.test_size is not None:
        config.setdefault("splitting", {})["test_size"] = args.test_size
    if args.split_seeds is not None:
        config.setdefault("splitting", {})["seeds"] = args.split_seeds
        if len(config["runs"]["training_seeds"]) != len(args.split_seeds):
            config["runs"]["training_seeds"] = [
                2000 + index for index in range(len(args.split_seeds))
            ]
    return experiment_dir, config


def _add_existing_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    allow_continue: bool = False,
) -> None:
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment directory containing experiment_config.json.",
    )
    rerun_group = parser.add_mutually_exclusive_group()
    rerun_group.add_argument(
        "--force",
        action="store_true",
        help="Rerun this stage even when matching completed manifests exist.",
    )
    if allow_continue:
        rerun_group.add_argument(
            "--continue",
            dest="continue_completed",
            action="store_true",
            help=(
                "Skip units with complete output artifacts, even when their "
                "command manifest or configuration fingerprint differs."
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Create frozen train/test splits."
    )
    prepare.add_argument("--dataset-dir", type=Path, default=None)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=None)
    prepare.add_argument("--data-file", type=Path, default=None)
    prepare.add_argument("--info-file", type=Path, default=None)
    prepare.add_argument("--utility-feature-file", type=Path, default=None)
    prepare.add_argument("--meta-file", type=Path, default=None)
    prepare.add_argument("--test-size", type=float, default=None)
    prepare.add_argument("--split-seeds", type=int, nargs="+", default=None)
    prepare.add_argument("--force", action="store_true")

    discover = subparsers.add_parser(
        "discover", help="Discover split constraints."
    )
    _add_existing_experiment_args(discover)
    discover.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write agent request previews without API calls.",
    )

    for name, help_text in (
        ("generate", "Train generators and create synthetic samples."),
        ("postprocess", "Apply split-specific proposed constraints."),
        ("evaluate", "Evaluate and aggregate raw and constrained samples."),
        ("all", "Run every stage from an existing prepared configuration."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _add_existing_experiment_args(
            command,
            allow_continue=name in {"postprocess", "evaluate"},
        )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        experiment_dir, config = initialize_config(args)
        prepare_splits(experiment_dir, config, force=args.force)
        return

    experiment_dir = args.experiment_dir.resolve()
    config = load_experiment_config(experiment_dir)
    if args.command == "discover":
        discover_constraints(
            experiment_dir, config, force=args.force, dry_run=args.dry_run
        )
    elif args.command == "generate":
        generate_synthetic(experiment_dir, config, force=args.force)
    elif args.command == "postprocess":
        postprocess_synthetic(
            experiment_dir,
            config,
            force=args.force,
            continue_completed=args.continue_completed,
        )
    elif args.command == "evaluate":
        evaluate_all(
            experiment_dir,
            config,
            force=args.force,
            continue_completed=args.continue_completed,
        )
    elif args.command == "all":
        prepare_splits(experiment_dir, config, force=args.force)
        discover_constraints(experiment_dir, config, force=args.force)
        generate_synthetic(experiment_dir, config, force=args.force)
        postprocess_synthetic(experiment_dir, config, force=args.force)
        evaluate_all(experiment_dir, config, force=args.force)


if __name__ == "__main__":
    main()
