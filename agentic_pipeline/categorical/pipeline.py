"""Discover categorical dependencies through a persistent model tool loop.

Both positional inputs accept either the expected file or a directory containing
it. Only columns marked ``cat`` in the ``info.json`` beside ``meta.json`` are
shown to the model or accepted by the tools. Missing categorical values are not
supported.
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

from .config import (
    DEFAULT_MAX_AGENT_TURNS,
    DEFAULT_MAX_ATOMIC_CONFIGURATIONS,
    DEFAULT_MAX_CONFIGURATION_SUMMARIES,
    DEFAULT_MAX_COUNTEREXAMPLES,
    DEFAULT_MAX_DETERMINANTS,
    DEFAULT_MODEL,
    DEFAULT_VIOLATION_THRESHOLD,
)
from .consolidation import consolidate_records
from .dataset_io import load_inputs, resolve_input, write_json, write_json_atomic
from .models import CategoricalConstraintProposal, ProposerCompletion
from .prompting import SYSTEM_PROMPT, build_user_prompt
from .tools import CategoricalToolContext, execute_tool, tool_schemas
from .verifier import CategoricalConstraintVerifier


DEFAULT_OUTPUT_DIR = Path("agentic_constraints/categorical")
CONSTRAINT_FILENAME = "categorical_dependency_constraint.json"
REPORT_FILENAME = "run_report_categorical.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meta", type=Path, help="Path to meta.json or a directory containing it."
    )
    parser.add_argument(
        "data", type=Path, help="Path to data.csv or a directory containing it."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider", choices=SUPPORTED_PROVIDERS, default="openai"
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--sample-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-constraints", type=int, default=30)
    parser.add_argument(
        "--max-agent-turns", type=int, default=DEFAULT_MAX_AGENT_TURNS
    )
    parser.add_argument(
        "--max-determinants", type=int, default=DEFAULT_MAX_DETERMINANTS
    )
    parser.add_argument(
        "--violation-threshold", type=float, default=DEFAULT_VIOLATION_THRESHOLD
    )
    parser.add_argument(
        "--max-counterexamples", type=int, default=DEFAULT_MAX_COUNTEREXAMPLES
    )
    parser.add_argument(
        "--max-configuration-summaries",
        type=int,
        default=DEFAULT_MAX_CONFIGURATION_SUMMARIES,
    )
    parser.add_argument(
        "--max-atomic-configurations",
        type=int,
        default=DEFAULT_MAX_ATOMIC_CONFIGURATIONS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write a request preview without model calls.",
    )
    return parser.parse_args()


def build_sample(
    data: pd.DataFrame,
    categorical_columns: list[str],
    rows: int,
    seed: int,
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    positions = generator.choice(
        len(data), size=min(rows, len(data)), replace=False
    )
    return data.iloc[np.sort(positions)][categorical_columns].copy()


def _usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)


def run_proposer_conversation(
    client: ModelBackend | Any,
    model: str,
    user_prompt: str,
    context: CategoricalToolContext,
    max_turns: int = DEFAULT_MAX_AGENT_TURNS,
) -> tuple[ProposerCompletion, list[dict[str, Any]]]:
    """Let one proposer choose tools until it emits structured completion."""
    backend = ensure_backend(client)
    input_items: list[Any] = [{"role": "user", "content": user_prompt}]
    trace: list[dict[str, Any]] = []
    schemas = tool_schemas()

    for turn in range(1, max_turns + 1):
        response = backend.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input_items=list(input_items),
            tools=schemas,
            output_schema=ProposerCompletion.model_json_schema(),
            output_name="categorical_proposer_completion",
        )
        items = list(getattr(response, "output", []) or [])
        input_items.extend(items)
        calls = [
            item
            for item in items
            if getattr(item, "type", None) == "function_call"
        ]
        trace_entry: dict[str, Any] = {
            "turn": turn,
            "response_id": getattr(response, "id", None),
            "usage": _usage(response),
            "tool_calls": [],
        }
        if not calls:
            try:
                completion = ProposerCompletion.model_validate_json(
                    getattr(response, "output_text", "") or ""
                )
            except ValidationError as exc:
                raise RuntimeError(
                    f"model returned invalid categorical completion: {exc}"
                ) from exc
            trace_entry["completion"] = completion.model_dump()
            trace.append(trace_entry)
            return completion, trace

        for call in calls:
            arguments, result = execute_tool(
                getattr(call, "name", ""),
                getattr(call, "arguments", None),
                context,
            )
            trace_entry["tool_calls"].append(
                {
                    "name": getattr(call, "name", None),
                    "call_id": getattr(call, "call_id", None),
                    "arguments": arguments,
                    "result": result,
                }
            )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": getattr(call, "call_id", None),
                    "output": json.dumps(
                        result, ensure_ascii=False, allow_nan=False
                    ),
                }
            )
        trace.append(trace_entry)

    completion = ProposerCompletion(rejected_hypotheses=[])
    trace.append(
        {
            "turn": max_turns + 1,
            "host_termination": "max_agent_turns_reached",
            "max_agent_turns": max_turns,
            "accepted_constraints_retained": len(context.accepted),
            "completion": completion.model_dump(),
        }
    )
    return completion, trace


def published_document(records: list[dict[str, Any]]) -> dict[str, Any]:
    constraints = []
    for record in records:
        proposal = CategoricalConstraintProposal.model_validate(
            record["constraint"]
        )
        verification = record["verification"]
        constraints.append(
            {
                **proposal.model_dump(),
                "support": verification["support"],
                "violation_rate": verification["violation_rate"],
            }
        )
    return {"dsl_version": "2.0", "constraints": constraints}


def _validate_args(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if args.sample_rows < 1:
        raise ValueError("--sample-rows must be positive")
    if not 1 <= args.max_constraints <= 100:
        raise ValueError("--max-constraints must be between 1 and 100")
    if args.max_agent_turns < 1:
        raise ValueError("--max-agent-turns must be positive")
    if args.max_determinants < 1:
        raise ValueError("--max-determinants must be positive")
    if not 0 < args.violation_threshold <= 1:
        raise ValueError("--violation-threshold must be in (0, 1]")
    if args.max_counterexamples < 0:
        raise ValueError("--max-counterexamples cannot be negative")
    if args.max_configuration_summaries < 0:
        raise ValueError("--max-configuration-summaries cannot be negative")
    if args.max_atomic_configurations < 1:
        raise ValueError("--max-atomic-configurations must be positive")
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be positive")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    meta_path = resolve_input(args.meta, "meta.json")
    data_path = resolve_input(args.data, "data.csv")
    meta, data, descriptions = load_inputs(meta_path, data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    verifier = CategoricalConstraintVerifier(
        data=data,
        categorical_columns=set(descriptions),
        violation_threshold=args.violation_threshold,
        max_determinants=args.max_determinants,
        max_counterexamples=args.max_counterexamples,
        max_atomic_configurations=args.max_atomic_configurations,
    )
    samples = [
        build_sample(
            data,
            list(descriptions),
            args.sample_rows,
            args.seed + run - 1,
        )
        for run in range(1, args.runs + 1)
    ]
    prompts = [
        build_user_prompt(
            dataset_description=meta["dataset_description"],
            categorical_descriptions=descriptions,
            sample=sample,
            run=run,
            runs=args.runs,
            max_constraints=args.max_constraints,
            violation_threshold=args.violation_threshold,
        )
        for run, sample in enumerate(samples, start=1)
    ]
    metadata = {
        "provider": args.provider,
        "model": args.model,
        "meta_path": str(meta_path),
        "data_path": str(data_path),
        "rows": len(data),
        "categorical_columns": list(descriptions),
        "runs": args.runs,
        "sample_rows": args.sample_rows,
        "seed": args.seed,
        "max_constraints_per_run": args.max_constraints,
        "max_agent_turns": args.max_agent_turns,
        "max_determinants": args.max_determinants,
        "violation_threshold": args.violation_threshold,
        "max_counterexamples": args.max_counterexamples,
        "max_configuration_summaries": args.max_configuration_summaries,
        "max_atomic_configurations": args.max_atomic_configurations,
    }

    if args.dry_run:
        preview_path = args.output_dir / "request_preview_categorical.json"
        write_json(
            preview_path,
            {
                **metadata,
                "instructions": SYSTEM_PROMPT,
                "tools": tool_schemas(),
                "completion_schema": ProposerCompletion.model_json_schema(),
                "run_prompts": prompts,
            },
        )
        print(f"Wrote dry-run preview to {preview_path}")
        return

    load_dotenv()
    client = create_backend(
        args.provider, max_output_tokens=args.max_output_tokens
    )
    all_records: list[dict[str, Any]] = []
    dependency_graph: dict[str, set[str]] = {}
    run_reports: list[dict[str, Any]] = []
    for run, prompt in enumerate(prompts, start=1):
        context = CategoricalToolContext(
            data=data,
            descriptions=descriptions,
            verifier=verifier,
            max_configuration_summaries=args.max_configuration_summaries,
            max_constraints=args.max_constraints,
            dependency_graph=dependency_graph,
        )
        completion, trace = run_proposer_conversation(
            client=client,
            model=args.model,
            user_prompt=prompt,
            context=context,
            max_turns=args.max_agent_turns,
        )
        records = list(context.accepted.values())
        all_records.extend(records)
        run_reports.append(
            {
                "run": run,
                "accepted_constraints": records,
                "completion": completion.model_dump(),
                "trace": trace,
            }
        )

    consolidated, consolidation_report = consolidate_records(
        all_records, verifier
    )
    constraint_path = args.output_dir / CONSTRAINT_FILENAME
    report_path = args.output_dir / REPORT_FILENAME
    write_json_atomic(constraint_path, published_document(consolidated))
    write_json(
        report_path,
        {
            **metadata,
            "accepted_before_consolidation": len(all_records),
            "published_constraints": len(consolidated),
            "consolidation": consolidation_report,
            "consolidated_records": consolidated,
            "run_reports": run_reports,
        },
    )
    print(f"Wrote {len(consolidated)} constraints to {constraint_path}")
    print(f"Wrote run report to {report_path}")


if __name__ == "__main__":
    main()
