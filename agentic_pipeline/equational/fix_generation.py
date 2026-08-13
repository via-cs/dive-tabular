"""Generate and verify source-only repair code for equational constraints.

Both positional inputs accept either an explicit file or a directory containing
the expected file. By default, the constraints artifact is replaced atomically
after every target path has been processed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from agentic_pipeline.model_backends import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    SUPPORTED_PROVIDERS,
    ModelBackend,
    create_backend,
    ensure_backend,
)

from .config import DEFAULT_MODEL
from .dataset_io import (
    load_optional_column_descriptions,
    resolve_input,
    resolve_output,
    write_json_atomic,
)
from .fix_verifier import FixVerifier
from .models import EquationalConstraint, FixCodeEntry, FixProposal
from .prompting import (
    FIX_SYSTEM_PROMPT,
    build_fix_prompt,
    build_fix_refinement_prompt,
)
from .verifier import validate_check_code


CONSTRAINT_FILENAME = "equational_constraints.json"
DATA_FILENAME = "data.csv"
REPORT_FILENAME = "fix_generation_report.json"
CORE_CONSTRAINT_FIELDS = {
    "id",
    "description",
    "rationale",
    "columns",
    "check_code",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "constraints",
        type=Path,
        help=(
            "Path to equational_constraints.json or a directory containing it."
        ),
    )
    parser.add_argument(
        "data",
        type=Path,
        help="Path to data.csv or a directory containing data.csv.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help=(
            "Optional meta.json file/directory for column descriptions; by "
            "default a meta.json beside data.csv is used when present."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON file or directory (default: atomically replace the "
            "input constraints file)."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Report JSON file or directory (default: fix_generation_report.json "
            "beside the output)."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider", choices=SUPPORTED_PROVIDERS, default="openai"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--sample-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-refinements", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-counterexamples", type=int, default=20)
    parser.add_argument("--violation-threshold", type=float, default=0.005)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write prompt previews without a model call.",
    )
    return parser.parse_args()


def _usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)


def _response_items(response: Any) -> list[Any]:
    return list(getattr(response, "output", []) or [])


def _fix_request(
    client: ModelBackend | Any, model: str, input_items: list[Any]
) -> Any:
    return ensure_backend(client).create(
        model=model,
        instructions=FIX_SYSTEM_PROMPT,
        input_items=list(input_items),
        output_schema=FixProposal.model_json_schema(),
        output_name="equational_fix_proposal",
    )


def load_constraint_artifact(
    constraints_path: Path,
) -> tuple[list[dict[str, Any]], list[EquationalConstraint]]:
    """Load the JSON artifact while preserving non-core fields for output."""
    payload = json.loads(constraints_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("equational_constraints.json must contain a JSON list")
    raw_constraints: list[dict[str, Any]] = []
    constraints: list[EquationalConstraint] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"constraint at position {position} must be an object")
        missing = CORE_CONSTRAINT_FIELDS - set(item)
        if missing:
            raise ValueError(
                f"constraint at position {position} is missing fields: {sorted(missing)}"
            )
        constraint = EquationalConstraint.model_validate(
            {name: item[name] for name in CORE_CONSTRAINT_FIELDS}
        )
        if constraint.id in seen_ids:
            raise ValueError(f"duplicate constraint ID: {constraint.id}")
        seen_ids.add(constraint.id)
        raw_constraints.append(dict(item))
        constraints.append(constraint)
    return raw_constraints, constraints


def validate_fix_inputs(
    data: pd.DataFrame, constraints: list[EquationalConstraint]
) -> None:
    if data.empty:
        raise ValueError("data.csv must contain at least one row")
    for constraint in constraints:
        missing = set(constraint.columns) - set(data.columns)
        if missing:
            raise ValueError(
                f"constraint {constraint.id} uses missing columns: {sorted(missing)}"
            )
        non_numeric = [
            name
            for name in constraint.columns
            if not pd.api.types.is_numeric_dtype(data[name])
        ]
        if non_numeric:
            raise ValueError(
                f"constraint {constraint.id} uses non-numeric columns: {non_numeric}"
            )
        involved = data[constraint.columns]
        if involved.isna().any().any():
            raise ValueError(
                f"constraint {constraint.id} has missing values in involved columns"
            )
        if not np.isfinite(involved.to_numpy(dtype=float)).all():
            raise ValueError(
                f"constraint {constraint.id} has non-finite involved values"
            )
        validate_check_code(constraint, set(data.columns))


def build_source_sample(
    data: pd.DataFrame,
    constraint: EquationalConstraint,
    target_column: str,
    sample_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, list[int]]:
    """Build a stable target-free sample for one independent repair path."""
    fingerprint_seed = int.from_bytes(
        f"{constraint.id}\0{target_column}".encode("utf-8"), "little"
    ) % (2**32)
    positions = np.random.default_rng(seed ^ fingerprint_seed).permutation(len(data))[
        :sample_rows
    ]
    source_columns = [
        column for column in constraint.columns if column != target_column
    ]
    return data.iloc[positions][source_columns].copy(), positions.tolist()


def _invalid_output_verification(exc: ValidationError) -> dict[str, Any]:
    return {
        "status": "invalid_model_output",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "counterexamples": [],
    }


def generate_fix_codes(
    client: ModelBackend | Any,
    model: str,
    raw_constraints: list[dict[str, Any]],
    constraints: list[EquationalConstraint],
    data: pd.DataFrame,
    column_descriptions: dict[str, str],
    sample_rows: int = 100,
    seed: int = 42,
    max_refinements: int = 3,
    violation_threshold: float = 0.005,
    max_counterexamples: int = 20,
    timeout_seconds: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate every `(constraint, target)` path and return artifact plus trace."""
    validate_fix_inputs(data, constraints)
    verifier = FixVerifier(
        data=data,
        violation_threshold=violation_threshold,
        max_counterexamples=max_counterexamples,
        timeout_seconds=timeout_seconds,
        sample_seed=seed,
    )
    output: list[dict[str, Any]] = []
    target_reports: list[dict[str, Any]] = []

    for raw_constraint, constraint in zip(raw_constraints, constraints, strict=True):
        entries: list[FixCodeEntry] = []
        for target_column in constraint.columns:
            source_sample, _ = build_source_sample(
                data,
                constraint,
                target_column,
                sample_rows,
                seed,
            )
            input_items: list[Any] = [
                {
                    "role": "user",
                    "content": build_fix_prompt(
                        constraint=constraint.model_dump(),
                        target_column=target_column,
                        column_descriptions=column_descriptions,
                        source_sample=source_sample,
                    ),
                }
            ]
            attempts: list[dict[str, Any]] = []
            accepted_code: str | None = None
            terminal_status = "refinements_exhausted"
            final_verification: dict[str, Any] | None = None

            for attempt_index in range(max_refinements + 1):
                response = _fix_request(client, model, input_items)
                response_items = _response_items(response)
                input_items.extend(response_items)
                attempt: dict[str, Any] = {
                    "attempt": attempt_index + 1,
                    "refinement_round": attempt_index,
                    "response_id": getattr(response, "id", None),
                    "usage": _usage_dict(response),
                }
                try:
                    proposal = FixProposal.model_validate_json(
                        response.output_text or ""
                    )
                    attempt["proposal"] = proposal.model_dump()
                    if proposal.code is None:
                        terminal_status = "model_declined"
                        attempts.append(attempt)
                        break
                    final_verification = verifier.verify(
                        constraint, target_column, proposal.code
                    )
                except ValidationError as exc:
                    final_verification = _invalid_output_verification(exc)
                attempt["verification"] = final_verification
                attempts.append(attempt)

                if final_verification["status"] == "accepted":
                    accepted_code = proposal.code
                    terminal_status = "accepted"
                    break
                if attempt_index < max_refinements:
                    input_items.append(
                        {
                            "role": "user",
                            "content": build_fix_refinement_prompt(
                                constraint_id=constraint.id,
                                target_column=target_column,
                                refinement_round=attempt_index + 1,
                                max_refinements=max_refinements,
                                verification=final_verification,
                            ),
                        }
                    )

            entries.append(FixCodeEntry(column=target_column, code=accepted_code))
            target_reports.append(
                {
                    "constraint_id": constraint.id,
                    "target_column": target_column,
                    "source_columns": list(source_sample.columns),
                    "status": terminal_status,
                    "final_verification": final_verification,
                    "attempts": attempts,
                }
            )

        published = dict(raw_constraint)
        published["fix_code"] = [entry.model_dump() for entry in entries]
        output.append(published)

    report = {
        "model": model,
        "rows": len(data),
        "constraints": len(constraints),
        "target_paths": len(target_reports),
        "sample_rows": sample_rows,
        "seed": seed,
        "max_refinements": max_refinements,
        "violation_threshold": violation_threshold,
        "max_counterexamples": max_counterexamples,
        "timeout_seconds": timeout_seconds,
        "accepted_paths": sum(
            target_report["status"] == "accepted"
            for target_report in target_reports
        ),
        "unavailable_paths": sum(
            target_report["status"] != "accepted"
            for target_report in target_reports
        ),
        "targets": target_reports,
    }
    return output, report


def build_fix_previews(
    constraints: list[EquationalConstraint],
    data: pd.DataFrame,
    column_descriptions: dict[str, str],
    sample_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    validate_fix_inputs(data, constraints)
    previews = []
    for constraint in constraints:
        for target_column in constraint.columns:
            sample, _ = build_source_sample(
                data, constraint, target_column, sample_rows, seed
            )
            previews.append(
                {
                    "constraint_id": constraint.id,
                    "target_column": target_column,
                    "source_columns": list(sample.columns),
                    "user_prompt": build_fix_prompt(
                        constraint.model_dump(),
                        target_column,
                        column_descriptions,
                        sample,
                    ),
                }
            )
    return previews


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be positive")
    if args.sample_rows < 1:
        raise ValueError("--sample-rows must be positive")
    if args.max_refinements < 0:
        raise ValueError("--max-refinements cannot be negative")
    if args.max_counterexamples < 0:
        raise ValueError("--max-counterexamples cannot be negative")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if not 0 <= args.violation_threshold <= 1:
        raise ValueError("--violation-threshold must be between 0 and 1")


def main() -> None:
    args = parse_args()
    _validate_cli_args(args)
    constraints_path = resolve_input(args.constraints, CONSTRAINT_FILENAME)
    data_path = resolve_input(args.data, DATA_FILENAME)
    meta_path = resolve_input(args.meta, "meta.json") if args.meta else None
    output_path = resolve_output(args.output, constraints_path, CONSTRAINT_FILENAME)
    report_path = resolve_output(
        args.report,
        output_path.with_name(REPORT_FILENAME),
        REPORT_FILENAME,
    )
    raw_constraints, constraints = load_constraint_artifact(constraints_path)
    data = pd.read_csv(data_path)
    descriptions = load_optional_column_descriptions(data_path, meta_path)

    metadata = {
        "provider": args.provider,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "constraints_path": str(constraints_path),
        "data_path": str(data_path),
        "meta_path": str(meta_path) if meta_path else None,
        "output_path": str(output_path),
        "report_path": str(report_path),
        "rows": len(data),
        "sample_rows": args.sample_rows,
        "seed": args.seed,
        "max_refinements": args.max_refinements,
        "violation_threshold": args.violation_threshold,
        "max_counterexamples": args.max_counterexamples,
        "timeout_seconds": args.timeout_seconds,
    }

    if args.dry_run:
        previews = build_fix_previews(
            constraints,
            data,
            descriptions,
            args.sample_rows,
            args.seed,
        )
        preview_path = report_path.with_name("fix_request_preview.json")
        write_json_atomic(
            preview_path,
            {
                **metadata,
                "system_prompt": FIX_SYSTEM_PROMPT,
                "output_schema": FixProposal.model_json_schema(),
                "previews": previews,
            },
        )
        print(f"Wrote fix-generation dry-run preview to {preview_path}")
        return

    load_dotenv()
    output, report = generate_fix_codes(
        client=create_backend(
            args.provider, max_output_tokens=args.max_output_tokens
        ),
        model=args.model,
        raw_constraints=raw_constraints,
        constraints=constraints,
        data=data,
        column_descriptions=descriptions,
        sample_rows=args.sample_rows,
        seed=args.seed,
        max_refinements=args.max_refinements,
        violation_threshold=args.violation_threshold,
        max_counterexamples=args.max_counterexamples,
        timeout_seconds=args.timeout_seconds,
    )
    write_json_atomic(output_path, output)
    write_json_atomic(report_path, {**metadata, **report})
    print(f"Wrote fix code for {report['target_paths']} target paths to {output_path}")
    print(f"Wrote fix-generation report to {report_path}")


if __name__ == "__main__":
    main()
