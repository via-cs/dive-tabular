"""Static and full-data verification for LLM-generated equation checks."""

from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .models import EquationalConstraint


BANNED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
BANNED_ATTRIBUTES = {
    "dump",
    "dumps",
    "at",
    "columns",
    "compat",
    "core",
    "ctypeslib",
    "f2py",
    "iat",
    "iloc",
    "io",
    "lib",
    "load",
    "loads",
    "loc",
    "os",
    "testing",
    "util",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_sql",
    "read_stata",
    "read_table",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
}
BANNED_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.DictComp,
    ast.For,
    ast.GeneratorExp,
    ast.ListComp,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.SetComp,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Yield,
    ast.YieldFrom,
)
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sum": sum,
}


class CheckCodeError(ValueError):
    """Raised when generated check code violates the static contract."""


def _rooted_at_df(node: ast.AST) -> bool:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return isinstance(node, ast.Name) and node.id == "df"


def _literal_columns(slice_node: ast.AST) -> set[str]:
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return {slice_node.value}
    if isinstance(slice_node, (ast.List, ast.Tuple)):
        values = set()
        for element in slice_node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(
                element.value, str
            ):
                raise CheckCodeError("df column lists must contain string literals")
            values.add(element.value)
        return values
    raise CheckCodeError("df columns must be referenced with literal names")


class _ContractVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.columns: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, BANNED_NODES):
            raise CheckCodeError(f"{type(node).__name__} is not allowed")
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise CheckCodeError("private and dunder attributes are not allowed")
        if node.attr in BANNED_ATTRIBUTES:
            raise CheckCodeError(f"attribute {node.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
            raise CheckCodeError(f"call to {node.func.id!r} is not allowed")
        for keyword in node.keywords:
            if keyword.arg == "inplace":
                raise CheckCodeError("inplace operations are not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            raise CheckCodeError(f"name {node.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "df":
            self.columns.update(_literal_columns(node.slice))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(_rooted_at_df(target) for target in node.targets):
            raise CheckCodeError("check_code must not mutate df")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _rooted_at_df(node.target):
            raise CheckCodeError("check_code must not mutate df")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if _rooted_at_df(node.target):
            raise CheckCodeError("check_code must not mutate df")
        self.generic_visit(node)


def validate_check_code(
    constraint: EquationalConstraint,
    available_columns: set[str],
    max_ast_nodes: int = 300,
) -> set[str]:
    """Validate the restricted check-code format and return referenced columns."""
    try:
        tree = ast.parse(constraint.check_code, mode="exec")
    except SyntaxError as exc:
        raise CheckCodeError(f"invalid Python syntax: {exc.msg}") from exc

    if sum(1 for _ in ast.walk(tree)) > max_ast_nodes:
        raise CheckCodeError(f"check_code exceeds {max_ast_nodes} AST nodes")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise CheckCodeError("check_code must define exactly one function")

    function = tree.body[0]
    if function.name != "check":
        raise CheckCodeError("the function must be named check")
    if function.decorator_list:
        raise CheckCodeError("function decorators are not allowed")
    if (
        len(function.args.args) != 1
        or function.args.args[0].arg != "df"
        or function.args.posonlyargs
        or function.args.kwonlyargs
        or function.args.vararg
        or function.args.kwarg
        or function.args.defaults
        or function.args.kw_defaults
    ):
        raise CheckCodeError("check must have the exact signature check(df)")
    if not any(isinstance(node, ast.Return) for node in ast.walk(function)):
        raise CheckCodeError("check must contain a return statement")
    if sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)) != 1:
        raise CheckCodeError("nested functions are not allowed")

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != "df":
            continue
        parent = parents.get(node)
        direct_column_access = (
            isinstance(parent, ast.Subscript) and parent.value is node
        )
        direct_index_access = (
            isinstance(parent, ast.Attribute)
            and parent.value is node
            and parent.attr == "index"
        )
        if not direct_column_access and not direct_index_access:
            raise CheckCodeError(
                "df may only be used as df['column'] or df.index"
            )

    visitor = _ContractVisitor()
    visitor.visit(tree)
    if not visitor.columns:
        raise CheckCodeError("check_code does not reference any df columns")

    declared = set(constraint.columns)
    if visitor.columns != declared:
        raise CheckCodeError(
            "declared columns do not match code references: "
            f"declared={sorted(declared)}, referenced={sorted(visitor.columns)}"
        )
    unknown = declared - available_columns
    if unknown:
        raise CheckCodeError(f"unknown or non-numerical columns: {sorted(unknown)}")
    return visitor.columns


def constraint_fingerprint(constraint: EquationalConstraint) -> str:
    payload = json.dumps(
        constraint.model_dump(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_records(
    data: pd.DataFrame, positions: np.ndarray, columns: list[str]
) -> list[dict[str, Any]]:
    if positions.size == 0:
        return []
    rows = data.iloc[positions][columns].copy()
    return json.loads(rows.to_json(orient="records"))


def _verification_worker(
    result_queue: Any,
    data: pd.DataFrame,
    constraint: EquationalConstraint,
    threshold: float,
    max_examples: int,
    sample_seed: int,
) -> None:
    try:
        namespace: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "np": np,
            "pd": pd,
        }
        compiled = compile(
            constraint.check_code,
            f"<constraint:{constraint.id}>",
            "exec",
        )
        exec(compiled, namespace)
        check = namespace["check"]
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            pass_mask = check(data.copy(deep=True))

        if not isinstance(pass_mask, pd.Series):
            raise TypeError("check(df) must return a pandas Series")
        if len(pass_mask) != len(data) or not pass_mask.index.equals(data.index):
            raise ValueError("check(df) returned a misaligned Series")
        if pass_mask.isna().any():
            raise ValueError("check(df) returned missing values")
        if not pd.api.types.is_bool_dtype(pass_mask.dtype):
            raise TypeError("check(df) must return a Boolean Series")

        violation_positions = np.flatnonzero(~pass_mask.to_numpy(dtype=bool))
        violations = int(violation_positions.size)
        rows = int(len(data))
        violation_rate = float(violations / rows) if rows else 0.0
        if violations > max_examples:
            generator = np.random.default_rng(sample_seed)
            example_positions = np.sort(
                generator.choice(
                    violation_positions, size=max_examples, replace=False
                )
            )
        else:
            example_positions = violation_positions

        result_queue.put(
            {
                "status": (
                    "accepted"
                    if violation_rate <= threshold
                    else "high_violation_rate"
                ),
                "rows_checked": rows,
                "passes": rows - violations,
                "violations": violations,
                "violation_rate": violation_rate,
                "threshold": threshold,
                "counterexamples": _json_records(
                    data, example_positions, constraint.columns
                ),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(
            {
                "status": "execution_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )


@dataclass
class ConstraintVerifier:
    """Verify generated checks against one immutable full dataframe."""

    data: pd.DataFrame
    numerical_columns: set[str]
    violation_threshold: float = 0.0005
    max_counterexamples: int = 20
    timeout_seconds: float = 10.0
    sample_seed: int = 42

    def verify(self, constraint: EquationalConstraint) -> dict[str, Any]:
        fingerprint = constraint_fingerprint(constraint)
        base = {"id": constraint.id, "fingerprint": fingerprint}
        try:
            validate_check_code(constraint, self.numerical_columns)
        except CheckCodeError as exc:
            return {
                **base,
                "status": "invalid_code",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        context = mp.get_context("spawn")
        result_queue = context.Queue(maxsize=1)
        stable_seed = int(fingerprint[:8], 16) ^ self.sample_seed
        process = context.Process(
            target=_verification_worker,
            args=(
                result_queue,
                self.data,
                constraint,
                self.violation_threshold,
                self.max_counterexamples,
                stable_seed,
            ),
        )
        process.start()
        process.join(self.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            return {
                **base,
                "status": "execution_error",
                "error_type": "TimeoutError",
                "error_message": (
                    f"check(df) exceeded {self.timeout_seconds:g} seconds"
                ),
            }

        try:
            result = result_queue.get(timeout=1.0)
        except queue.Empty:
            result = {
                "status": "execution_error",
                "error_type": "WorkerError",
                "error_message": (
                    f"verification worker exited with code {process.exitcode}"
                ),
            }
        finally:
            result_queue.close()
        return {**base, **result}

    def verify_batch(
        self, constraints: list[EquationalConstraint]
    ) -> dict[str, Any]:
        seen: set[str] = set()
        results = []
        for constraint in constraints:
            if constraint.id in seen:
                results.append(
                    {
                        "id": constraint.id,
                        "fingerprint": constraint_fingerprint(constraint),
                        "status": "invalid_code",
                        "error_type": "DuplicateConstraintId",
                        "error_message": "constraint IDs must be unique in a batch",
                    }
                )
                continue
            seen.add(constraint.id)
            results.append(self.verify(constraint))
        return {"constraints": results}
