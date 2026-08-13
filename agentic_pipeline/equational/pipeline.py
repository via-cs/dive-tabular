"""Run multi-phase discovery of numerical equational constraints.

Both positional inputs accept either the expected file or a directory that
contains it. Discovery phases use disjoint row samples; failed candidates are
handled by a separate, bounded refinement prompt.
"""

from __future__ import annotations

import argparse
import hashlib
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
from .consolidation import consolidate_records
from .dataset_io import load_inputs, resolve_input, write_json, write_json_atomic
from .fix_generation import generate_fix_codes
from .models import (
    ConstraintBatch,
    DiscoveryOutput,
    EquationalConstraint,
    FixProposal,
)
from .prompting import (
    DISCOVERY_SYSTEM_PROMPT,
    FIX_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    build_discovery_prompt,
    build_refinement_prompt,
)
from .verifier import ConstraintVerifier, constraint_fingerprint

DEFAULT_OUTPUT_DIR = Path("agentic_constraints/flights")
VERIFY_TOOL_NAME = "verify_equational_constraints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meta",
        type=Path,
        help="Path to meta.json or a directory containing meta.json.",
    )
    parser.add_argument(
        "data",
        type=Path,
        help="Path to data.csv or a directory containing data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default="openai",
        help="Native model API provider (default: openai).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum Anthropic output tokens per model turn.",
    )
    parser.add_argument("--sample-rows", type=int, default=100)
    parser.add_argument(
        "--refinement-sample-rows",
        type=int,
        default=100,
        help="Random involved-column rows shown to each refiner.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-constraints", type=int, default=12)
    parser.add_argument("--max-discovery-phases", type=int, default=3)
    parser.add_argument("--max-refinement-rounds", type=int, default=3)
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help=(
            "Publish every verifier-accepted constraint without deterministic "
            "column-set deduplication or private-column optimization."
        ),
    )
    parser.add_argument(
        "--fix-model",
        default=None,
        help="Model used for repair generation (default: --model).",
    )
    parser.add_argument("--fix-sample-rows", type=int, default=100)
    parser.add_argument("--max-fix-refinements", type=int, default=3)
    parser.add_argument(
        "--fix-violation-threshold", type=float, default=0.005
    )
    parser.add_argument(
        "--skip-fix-generation",
        action="store_true",
        help="Publish check_code only without generating repair paths.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-counterexamples", type=int, default=20)
    parser.add_argument("--violation-threshold", type=float, default=0.0005)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write request previews without a model call.",
    )
    return parser.parse_args()


def build_discovery_samples(
    data: pd.DataFrame,
    numerical_columns: list[str],
    sample_rows: int,
    max_phases: int,
    seed: int,
) -> list[tuple[pd.DataFrame, list[int]]]:
    """Return up to max_phases reproducible, position-disjoint samples."""
    positions = np.random.default_rng(seed).permutation(len(data))
    samples: list[tuple[pd.DataFrame, list[int]]] = []
    for phase_index in range(max_phases):
        start = phase_index * sample_rows
        if start >= len(positions):
            break
        chosen = positions[start : start + sample_rows]
        samples.append(
            (data.iloc[chosen][numerical_columns].copy(), chosen.tolist())
        )
    return samples


def verification_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": VERIFY_TOOL_NAME,
        "description": (
            "Run proposed check(df) functions on the complete dataset. Returns "
            "execution errors or full-data violation rates and sampled "
            "violating rows. Discovery batches candidates; refinement submits "
            "exactly one candidate."
        ),
        "parameters": ConstraintBatch.model_json_schema(),
        "strict": True,
    }


def _response_items(response: Any) -> list[Any]:
    return list(getattr(response, "output", []) or [])


def _usage_dict(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)


def _parse_phase_output(response: Any) -> DiscoveryOutput:
    try:
        return DiscoveryOutput.model_validate_json(response.output_text or "")
    except ValidationError as exc:
        raise RuntimeError(
            f"model returned invalid phase-completion output: {exc}"
        ) from exc


def _response_request(
    client: ModelBackend | Any,
    model: str,
    instructions: str,
    input_items: list[Any],
) -> Any:
    return ensure_backend(client).create(
        model=model,
        instructions=instructions,
        input_items=list(input_items),
        tools=[verification_tool_schema()],
        output_schema=DiscoveryOutput.model_json_schema(),
        output_name="equational_phase_output",
    )


def _constraint_summary(record: dict[str, Any]) -> dict[str, Any]:
    constraint = record["constraint"]
    return {
        "id": constraint["id"],
        "description": constraint["description"],
        "rationale": constraint["rationale"],
        "columns": constraint["columns"],
        "column_signature": sorted(constraint["columns"]),
        "check_code": constraint["check_code"],
        "violation_rate": record["verification"].get("violation_rate"),
    }


def _rejection_summary(record: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "id": record["id"],
        "reason": record["reason"],
    }
    if record.get("description"):
        summary["description"] = record["description"]
    if record.get("columns"):
        summary["columns"] = record["columns"]
    return summary


def build_refinement_sample(
    verifier: ConstraintVerifier,
    constraint_id: str,
    columns: list[str],
    sample_rows: int,
) -> pd.DataFrame:
    """Return a stable random sample projected to valid involved columns."""
    valid_columns = [
        column for column in columns if column in verifier.data.columns
    ]
    if not valid_columns or sample_rows <= 0 or verifier.data.empty:
        return pd.DataFrame(columns=valid_columns)
    digest = hashlib.sha256(constraint_id.encode("utf-8")).digest()
    stable_seed = (
        int.from_bytes(digest[:8], "big") ^ verifier.sample_seed
    ) % (2**32)
    positions = np.random.default_rng(stable_seed).choice(
        len(verifier.data),
        size=min(sample_rows, len(verifier.data)),
        replace=False,
    )
    return verifier.data.iloc[positions][valid_columns].copy()


def _refinement_context(
    candidate: dict[str, Any],
    numerical_descriptions: dict[str, str],
    verifier: ConstraintVerifier,
    sample_rows: int,
) -> tuple[dict[str, str], pd.DataFrame]:
    columns = candidate["constraint"]["columns"]
    descriptions = {
        column: numerical_descriptions[column]
        for column in columns
        if column in numerical_descriptions
    }
    sample = build_refinement_sample(
        verifier=verifier,
        constraint_id=candidate["constraint"]["id"],
        columns=columns,
        sample_rows=sample_rows,
    )
    return descriptions, sample


def constraint_equation_family(
    constraint: EquationalConstraint,
) -> tuple[str, ...]:
    """Identify algebraic variants that involve the exact same columns."""
    return tuple(sorted(constraint.columns))


def _record_rejection(
    rejected: list[dict[str, Any]],
    candidate: dict[str, Any] | None,
    constraint_id: str,
    reason: str,
    phase: int,
) -> None:
    constraint = candidate["constraint"] if candidate else {}
    rejected.append(
        {
            "id": constraint_id,
            "description": constraint.get("description"),
            "columns": constraint.get("columns"),
            "reason": reason,
            "discovery_phase": phase,
            "last_verification": (
                candidate.get("verification") if candidate else None
            ),
        }
    )


def _run_verifier_calls(
    calls: list[Any],
    verifier: ConstraintVerifier,
    accepted: dict[str, dict[str, Any]],
    phase: int,
    refinement_round: int,
    eligible_ids: set[str] | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Return call traces, refinable failures, and terminal rejections."""
    call_traces: list[dict[str, Any]] = []
    failed: dict[str, dict[str, Any]] = {}
    terminal_rejections: list[dict[str, Any]] = []
    accepted_ids = {
        record["constraint"]["id"] for record in accepted.values()
    }

    for call in calls:
        try:
            raw_arguments = json.loads(call.arguments)
        except (TypeError, json.JSONDecodeError):
            raw_arguments = {"_raw": getattr(call, "arguments", None)}

        if call.name != VERIFY_TOOL_NAME:
            tool_result = {"error": f"unknown tool requested: {call.name}"}
        else:
            try:
                arguments = ConstraintBatch.model_validate_json(call.arguments)
                results: list[dict[str, Any]] = []
                seen_call_ids: set[str] = set()
                for constraint in arguments.constraints:
                    family = constraint_equation_family(constraint)
                    if eligible_ids is not None and constraint.id not in eligible_ids:
                        result = {
                            "id": constraint.id,
                            "fingerprint": constraint_fingerprint(constraint),
                            "status": "invalid_refinement",
                            "error_type": "IneligibleConstraintId",
                            "error_message": (
                                "refinement may only revise currently failed IDs"
                            ),
                        }
                    elif constraint.id in seen_call_ids:
                        result = {
                            "id": constraint.id,
                            "fingerprint": constraint_fingerprint(constraint),
                            "status": "duplicate_constraint_id",
                            "error_type": "DuplicateConstraintId",
                            "error_message": (
                                "constraint IDs must be unique in a tool call"
                            ),
                        }
                    elif constraint.id in accepted_ids:
                        result = {
                            "id": constraint.id,
                            "fingerprint": constraint_fingerprint(constraint),
                            "status": "duplicate_accepted_id",
                            "error_type": "DuplicateConstraintId",
                            "error_message": (
                                "an accepted constraint with this ID is frozen"
                            ),
                        }
                    else:
                        result = verifier.verify(constraint)

                    seen_call_ids.add(constraint.id)
                    result["equation_family"] = list(family)
                    fingerprint = result["fingerprint"]
                    candidate_record = {
                        "constraint": constraint.model_dump(),
                        "verification": result,
                        "discovery_phase": phase,
                        "refinement_round": refinement_round,
                    }
                    if result["status"] == "accepted":
                        accepted[fingerprint] = candidate_record
                        accepted_ids.add(constraint.id)
                    elif result["status"] in {
                        "duplicate_constraint_id",
                        "duplicate_accepted_id",
                        "invalid_refinement",
                    }:
                        terminal_rejections.append(candidate_record)
                    else:
                        failed[constraint.id] = candidate_record
                    results.append(result)
                tool_result = {"constraints": results}
            except (ValidationError, ValueError) as exc:
                tool_result = {
                    "error": "invalid tool arguments",
                    "details": str(exc),
                }

        call_traces.append(
            {
                "name": call.name,
                "arguments": raw_arguments,
                "result": tool_result,
                "call_id": call.call_id,
            }
        )

    return call_traces, failed, terminal_rejections


def _run_refinement_conversation(
    client: ModelBackend | Any,
    model: str,
    candidate: dict[str, Any],
    verifier: ConstraintVerifier,
    accepted: dict[str, dict[str, Any]],
    rejected: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    dataset_description: str,
    numerical_descriptions: dict[str, str],
    phase: int,
    max_refinement_rounds: int,
    refinement_sample_rows: int,
) -> None:
    """Refine one equation in one manually preserved conversation."""
    constraint_id = candidate["constraint"]["id"]
    current = candidate
    candidate_history: list[dict[str, Any]] = [candidate]
    input_items: list[Any] = []

    for refinement_round in range(1, max_refinement_rounds + 1):
        descriptions, sample = _refinement_context(
            candidate=current,
            numerical_descriptions=numerical_descriptions,
            verifier=verifier,
            sample_rows=refinement_sample_rows,
        )
        input_items.append(
            {
                "role": "user",
                "content": build_refinement_prompt(
                    dataset_description=dataset_description,
                    involved_descriptions=descriptions,
                    sample=sample,
                    candidate_history=candidate_history,
                    phase=phase,
                    refinement_round=refinement_round,
                    max_refinement_rounds=max_refinement_rounds,
                ),
            }
        )
        response = _response_request(
            client, model, REFINEMENT_SYSTEM_PROMPT, input_items
        )
        items = _response_items(response)
        input_items.extend(items)
        calls = [
            item
            for item in items
            if getattr(item, "type", None) == "function_call"
        ]
        trace_entry: dict[str, Any] = {
            "discovery_phase": phase,
            "stage": "refinement",
            "constraint_id": constraint_id,
            "refinement_round": refinement_round,
            "response_id": getattr(response, "id", None),
            "usage": _usage_dict(response),
            "tool_calls": [],
        }
        if not calls:
            completion = _parse_phase_output(response)
            trace_entry["completion"] = completion.model_dump()
            trace.append(trace_entry)
            reason = "model stopped without another principled revision"
            for item in completion.rejected_hypotheses:
                if item.id == constraint_id:
                    reason = item.reason
                    break
            _record_rejection(
                rejected, current, constraint_id, reason, phase
            )
            return

        call_traces, next_failed, terminal_rejections = (
            _run_verifier_calls(
                calls=calls,
                verifier=verifier,
                accepted=accepted,
                phase=phase,
                refinement_round=refinement_round,
                eligible_ids={constraint_id},
            )
        )
        trace_entry["tool_calls"] = call_traces
        trace.append(trace_entry)
        for call_trace in call_traces:
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_trace["call_id"],
                    "output": json.dumps(
                        call_trace["result"], ensure_ascii=False
                    ),
                }
            )

        if any(
            record["constraint"]["id"] == constraint_id
            for record in accepted.values()
        ):
            return

        terminal = next(
            (
                record
                for record in terminal_rejections
                if record["constraint"]["id"] == constraint_id
            ),
            None,
        )
        if terminal is not None:
            verification = terminal["verification"]
            _record_rejection(
                rejected,
                terminal,
                constraint_id,
                verification["error_message"],
                phase,
            )
            return

        revised = next_failed.get(constraint_id)
        if revised is not None:
            current = revised
            candidate_history.append(revised)

    _record_rejection(
        rejected,
        current,
        constraint_id,
        f"reached the limit of {max_refinement_rounds} refinement rounds",
        phase,
    )


def run_discovery_phase(
    client: ModelBackend | Any,
    model: str,
    discovery_prompt: str,
    verifier: ConstraintVerifier,
    accepted: dict[str, dict[str, Any]],
    phase: int,
    max_refinement_rounds: int,
    dataset_description: str,
    numerical_descriptions: dict[str, str],
    refinement_sample_rows: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run discovery, then one isolated conversation per failed equation."""
    input_items: list[Any] = [{"role": "user", "content": discovery_prompt}]
    trace: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    response = _response_request(
        client, model, DISCOVERY_SYSTEM_PROMPT, input_items
    )
    items = _response_items(response)
    input_items.extend(items)
    calls = [
        item for item in items if getattr(item, "type", None) == "function_call"
    ]
    trace_entry: dict[str, Any] = {
        "discovery_phase": phase,
        "stage": "discovery",
        "response_id": getattr(response, "id", None),
        "usage": _usage_dict(response),
        "tool_calls": [],
    }
    if not calls:
        completion = _parse_phase_output(response)
        trace_entry["completion"] = completion.model_dump()
        trace.append(trace_entry)
        for item in completion.rejected_hypotheses:
            _record_rejection(rejected, None, item.id, item.reason, phase)
        return trace, rejected

    call_traces, failed, terminal_rejections = _run_verifier_calls(
        calls=calls,
        verifier=verifier,
        accepted=accepted,
        phase=phase,
        refinement_round=0,
        eligible_ids=None,
    )
    accepted_ids = {
        record["constraint"]["id"] for record in accepted.values()
    }
    failed = {
        constraint_id: candidate
        for constraint_id, candidate in failed.items()
        if constraint_id not in accepted_ids
    }
    for candidate in terminal_rejections:
        verification = candidate["verification"]
        _record_rejection(
            rejected,
            candidate,
            candidate["constraint"]["id"],
            verification["error_message"],
            phase,
        )
    trace_entry["tool_calls"] = call_traces
    trace.append(trace_entry)
    for candidate in failed.values():
        _run_refinement_conversation(
            client=client,
            model=model,
            candidate=candidate,
            verifier=verifier,
            accepted=accepted,
            rejected=rejected,
            trace=trace,
            dataset_description=dataset_description,
            numerical_descriptions=numerical_descriptions,
            phase=phase,
            max_refinement_rounds=max_refinement_rounds,
            refinement_sample_rows=refinement_sample_rows,
        )
    return trace, rejected


def main() -> None:
    args = parse_args()
    if args.sample_rows < 1:
        raise ValueError("--sample-rows must be positive")
    if args.refinement_sample_rows < 1:
        raise ValueError("--refinement-sample-rows must be positive")
    if not 1 <= args.max_constraints <= 20:
        raise ValueError("--max-constraints must be between 1 and 20")
    if args.max_discovery_phases < 1:
        raise ValueError("--max-discovery-phases must be positive")
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be positive")
    if args.max_refinement_rounds < 0:
        raise ValueError("--max-refinement-rounds cannot be negative")
    if args.fix_sample_rows < 1:
        raise ValueError("--fix-sample-rows must be positive")
    if args.max_fix_refinements < 0:
        raise ValueError("--max-fix-refinements cannot be negative")
    if args.max_counterexamples < 0:
        raise ValueError("--max-counterexamples cannot be negative")
    if not 0 <= args.violation_threshold <= 1:
        raise ValueError("--violation-threshold must be between 0 and 1")
    if not 0 <= args.fix_violation_threshold <= 1:
        raise ValueError("--fix-violation-threshold must be between 0 and 1")

    meta_path = resolve_input(args.meta, "meta.json")
    data_path = resolve_input(args.data, "data.csv")
    meta, data, numerical_descriptions = load_inputs(meta_path, data_path)
    samples = build_discovery_samples(
        data=data,
        numerical_columns=list(numerical_descriptions),
        sample_rows=args.sample_rows,
        max_phases=args.max_discovery_phases,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fix_model = args.fix_model or args.model

    run_metadata = {
        "provider": args.provider,
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "meta_path": str(meta_path),
        "data_path": str(data_path),
        "rows": len(data),
        "numerical_columns": list(numerical_descriptions),
        "sample_rows_per_phase": args.sample_rows,
        "refinement_sample_rows": args.refinement_sample_rows,
        "sample_seed": args.seed,
        "max_discovery_phases": args.max_discovery_phases,
        "discovery_phases_run": len(samples),
        "max_refinement_rounds": args.max_refinement_rounds,
        "consolidation_enabled": not args.skip_consolidation,
        "consolidation_method": (
            "disabled"
            if args.skip_consolidation
            else "exact_column_deduplication_then_binary_optimization"
        ),
        "fix_model": fix_model,
        "fix_sample_rows": args.fix_sample_rows,
        "max_fix_refinements": args.max_fix_refinements,
        "fix_violation_threshold": args.fix_violation_threshold,
        "fix_generation_enabled": not args.skip_fix_generation,
        "violation_threshold": args.violation_threshold,
        "max_counterexamples": args.max_counterexamples,
        "discovery_samples": [
            {
                "phase": index,
                "rows": len(sample),
            }
            for index, (sample, _) in enumerate(samples, start=1)
        ],
    }

    if args.dry_run:
        previews = [
            {
                "phase": phase,
                "user_prompt": build_discovery_prompt(
                    dataset_description=meta["dataset_description"],
                    numerical_descriptions=numerical_descriptions,
                    sample=sample,
                    max_constraints=args.max_constraints,
                    phase=phase,
                    max_phases=len(samples),
                    accepted_summaries=[],
                    rejected_summaries=[],
                ),
            }
            for phase, (sample, _) in enumerate(samples, start=1)
        ]
        write_json(
            args.output_dir / "request_preview_equational.json",
            {
                **run_metadata,
                "discovery_instructions": DISCOVERY_SYSTEM_PROMPT,
                "refinement_instructions": REFINEMENT_SYSTEM_PROMPT,
                "fix_instructions": (
                    None if args.skip_fix_generation else FIX_SYSTEM_PROMPT
                ),
                "fix_output_schema": (
                    None
                    if args.skip_fix_generation
                    else FixProposal.model_json_schema()
                ),
                "phase_previews": previews,
                "tool": verification_tool_schema(),
            },
        )
        print(
            "Wrote dry-run preview to "
            f"{args.output_dir / 'request_preview_equational.json'}"
        )
        return

    load_dotenv()
    verifier = ConstraintVerifier(
        data=data,
        numerical_columns=set(numerical_descriptions),
        violation_threshold=args.violation_threshold,
        max_counterexamples=args.max_counterexamples,
        timeout_seconds=args.timeout_seconds,
        sample_seed=args.seed,
    )
    client = create_backend(
        args.provider, max_output_tokens=args.max_output_tokens
    )
    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    for phase, (sample, _) in enumerate(samples, start=1):
        discovery_prompt = build_discovery_prompt(
            dataset_description=meta["dataset_description"],
            numerical_descriptions=numerical_descriptions,
            sample=sample,
            max_constraints=args.max_constraints,
            phase=phase,
            max_phases=len(samples),
            accepted_summaries=[
                _constraint_summary(record) for record in accepted.values()
            ],
            rejected_summaries=[
                _rejection_summary(record) for record in rejected
            ],
        )
        phase_trace, phase_rejected = run_discovery_phase(
            client=client,
            model=args.model,
            discovery_prompt=discovery_prompt,
            verifier=verifier,
            accepted=accepted,
            phase=phase,
            max_refinement_rounds=args.max_refinement_rounds,
            dataset_description=meta["dataset_description"],
            numerical_descriptions=numerical_descriptions,
            refinement_sample_rows=args.refinement_sample_rows,
        )
        trace.extend(phase_trace)
        rejected.extend(phase_rejected)

    accepted_records = list(accepted.values())
    if args.skip_consolidation:
        published_records = accepted_records
        kept_ids = [
            record["constraint"]["id"] for record in published_records
        ]
        consolidation_report = {
            "method": "disabled",
            "status": "skipped",
            "reason": "disabled_by_flag",
            "input_constraints": len(accepted_records),
            "published_constraints": len(published_records),
            "kept_ids": kept_ids,
            "dropped_ids": [],
            "dropped_rules": [],
        }
    else:
        published_records, consolidation_report = consolidate_records(
            accepted_records
        )
        kept_ids = consolidation_report["kept_ids"]
    published = [
        EquationalConstraint.model_validate(record["constraint"])
        for record in published_records
    ]
    constraints_path = args.output_dir / "equational_constraint.json"
    report_path = args.output_dir / "run_report_equational.json"
    published_payload = [constraint.model_dump() for constraint in published]
    fix_report: dict[str, Any] | None = None
    if not args.skip_fix_generation:
        published_payload, fix_report = generate_fix_codes(
            client=client,
            model=fix_model,
            raw_constraints=published_payload,
            constraints=published,
            data=data,
            column_descriptions=numerical_descriptions,
            sample_rows=args.fix_sample_rows,
            seed=args.seed,
            max_refinements=args.max_fix_refinements,
            violation_threshold=args.fix_violation_threshold,
            max_counterexamples=args.max_counterexamples,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.skip_fix_generation:
        fix_report = {
            "status": "skipped",
            "reason": "disabled_by_flag",
        }
    write_json_atomic(constraints_path, published_payload)
    write_json(
        report_path,
        {
            **run_metadata,
            "published_constraints": len(published),
            "published_constraint_ids": kept_ids,
            "accepted_constraints": accepted_records,
            "accepted_verifications": [
                record["verification"] for record in accepted_records
            ],
            "rejected_hypotheses": rejected,
            "trace": trace,
            "consolidation": consolidation_report,
            "fix_generation": fix_report,
        },
    )
    fix_suffix = "" if args.skip_fix_generation else " with fix paths"
    print(f"Wrote {len(published)} constraints{fix_suffix} to {constraints_path}")
    print(f"Wrote run report to {report_path}")


if __name__ == "__main__":
    main()
