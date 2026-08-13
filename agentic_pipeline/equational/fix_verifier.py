"""Static validation and full-data verification for generated repair code."""

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
from .verifier import (
    BANNED_ATTRIBUTES,
    BANNED_NAMES,
    BANNED_NODES,
    SAFE_BUILTINS,
    CheckCodeError,
    _literal_columns,
    _rooted_at_df,
    validate_check_code,
)


BANNED_AGGREGATE_ATTRIBUTES = {
    "agg",
    "aggregate",
    "count",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "describe",
    "expanding",
    "groupby",
    "max",
    "mean",
    "median",
    "min",
    "mode",
    "nunique",
    "prod",
    "quantile",
    "rank",
    "rolling",
    "std",
    "sum",
    "unique",
    "value_counts",
    "var",
}
BANNED_FIX_CALL_NAMES = {"all", "any", "len", "max", "min", "sum"}
BANNED_FIX_NODES = BANNED_NODES + (ast.If, ast.IfExp, ast.Match, ast.NamedExpr)


class FixCodeError(ValueError):
    """Raised when generated repair code violates its static contract."""


class _FixContractVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.columns: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, BANNED_FIX_NODES):
            raise FixCodeError(f"{type(node).__name__} is not allowed")
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise FixCodeError("private and dunder attributes are not allowed")
        if node.attr in BANNED_ATTRIBUTES | BANNED_AGGREGATE_ATTRIBUTES:
            raise FixCodeError(f"attribute {node.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
            raise FixCodeError(f"call to {node.func.id!r} is not allowed")
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_FIX_CALL_NAMES:
            raise FixCodeError(
                f"aggregate-like call to {node.func.id!r} is not allowed"
            )
        for keyword in node.keywords:
            if keyword.arg == "inplace":
                raise FixCodeError("inplace operations are not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            raise FixCodeError(f"name {node.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "df":
            try:
                self.columns.update(_literal_columns(node.slice))
            except CheckCodeError as exc:
                raise FixCodeError(str(exc)) from exc
        elif _rooted_at_df(node.value):
            raise FixCodeError(
                "row-level subscripting of a source Series is not allowed"
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(_rooted_at_df(target) for target in node.targets):
            raise FixCodeError("fix code must not mutate df")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _rooted_at_df(node.target):
            raise FixCodeError("fix code must not mutate df")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if _rooted_at_df(node.target):
            raise FixCodeError("fix code must not mutate df")
        self.generic_visit(node)


def validate_fix_code(
    code: str,
    source_columns: set[str],
    target_column: str,
    max_ast_nodes: int = 300,
) -> set[str]:
    """Validate a source-only `fix(df)` and return its referenced columns."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise FixCodeError(f"invalid Python syntax: {exc.msg}") from exc

    if sum(1 for _ in ast.walk(tree)) > max_ast_nodes:
        raise FixCodeError(f"fix code exceeds {max_ast_nodes} AST nodes")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise FixCodeError("fix code must define exactly one function")
    function = tree.body[0]
    if function.name != "fix":
        raise FixCodeError("the function must be named fix")
    if function.decorator_list:
        raise FixCodeError("function decorators are not allowed")
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
        raise FixCodeError("fix must have the exact signature fix(df)")
    if not any(isinstance(node, ast.Return) for node in ast.walk(function)):
        raise FixCodeError("fix must contain a return statement")
    if sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)) != 1:
        raise FixCodeError("nested functions are not allowed")

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != "df":
            continue
        parent = parents.get(node)
        if not (isinstance(parent, ast.Subscript) and parent.value is node):
            raise FixCodeError("df may only be used as df['source_column']")

    visitor = _FixContractVisitor()
    visitor.visit(tree)
    if not visitor.columns:
        raise FixCodeError("fix code must reference at least one source column")
    if target_column in visitor.columns:
        raise FixCodeError(f"fix code must not reference target {target_column!r}")
    unknown = visitor.columns - source_columns
    if unknown:
        raise FixCodeError(f"fix code references unavailable columns: {sorted(unknown)}")
    return visitor.columns


def fix_fingerprint(
    constraint: EquationalConstraint, target_column: str, code: str
) -> str:
    payload = "\0".join((constraint.id, target_column, code)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _failed_records(
    data: pd.DataFrame,
    positions: np.ndarray,
    columns: list[str],
    target_column: str,
    proposed: pd.Series,
) -> list[dict[str, Any]]:
    if positions.size == 0:
        return []
    rows = data.iloc[positions][columns].copy()
    rows[f"_proposed_{target_column}"] = proposed.iloc[positions].to_numpy()
    return json.loads(rows.to_json(orient="records"))


def _fix_verification_worker(
    result_queue: Any,
    data: pd.DataFrame,
    constraint: EquationalConstraint,
    target_column: str,
    code: str,
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
        exec(compile(code, f"<fix:{constraint.id}:{target_column}>", "exec"), namespace)
        source_columns = [
            column for column in constraint.columns if column != target_column
        ]
        source_data = data[source_columns].copy(deep=True)
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            proposed = namespace["fix"](source_data)

        if not isinstance(proposed, pd.Series):
            raise TypeError("fix(df) must return a pandas Series")
        if len(proposed) != len(data) or not proposed.index.equals(data.index):
            raise ValueError("fix(df) returned a misaligned Series")
        if proposed.isna().any():
            raise ValueError("fix(df) returned missing values")
        if not pd.api.types.is_numeric_dtype(proposed.dtype) or pd.api.types.is_bool_dtype(
            proposed.dtype
        ):
            raise TypeError("fix(df) must return a non-Boolean numeric Series")
        if not np.isfinite(proposed.to_numpy(dtype=float)).all():
            raise ValueError("fix(df) returned non-finite values")

        repaired = data.copy(deep=True)
        repaired[target_column] = proposed
        check_namespace: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "np": np,
            "pd": pd,
        }
        exec(
            compile(
                constraint.check_code,
                f"<constraint:{constraint.id}>",
                "exec",
            ),
            check_namespace,
        )
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            pass_mask = check_namespace["check"](repaired)
        if not isinstance(pass_mask, pd.Series):
            raise TypeError("check(df) must return a pandas Series")
        if len(pass_mask) != len(data) or not pass_mask.index.equals(data.index):
            raise ValueError("check(df) returned a misaligned Series")
        if pass_mask.isna().any() or not pd.api.types.is_bool_dtype(pass_mask.dtype):
            raise TypeError("check(df) must return a non-missing Boolean Series")

        violation_positions = np.flatnonzero(~pass_mask.to_numpy(dtype=bool))
        violations = int(violation_positions.size)
        rows = len(data)
        violation_rate = float(violations / rows) if rows else 0.0
        if violations > max_examples:
            generator = np.random.default_rng(sample_seed)
            example_positions = np.sort(
                generator.choice(violation_positions, size=max_examples, replace=False)
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
                "counterexamples": _failed_records(
                    data,
                    example_positions,
                    constraint.columns,
                    target_column,
                    proposed,
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
class FixVerifier:
    """Verify one target repair against a frozen constraint and full dataframe."""

    data: pd.DataFrame
    violation_threshold: float = 0.005
    max_counterexamples: int = 20
    timeout_seconds: float = 10.0
    sample_seed: int = 42

    def verify(
        self, constraint: EquationalConstraint, target_column: str, code: str
    ) -> dict[str, Any]:
        fingerprint = fix_fingerprint(constraint, target_column, code)
        base = {
            "constraint_id": constraint.id,
            "target_column": target_column,
            "fingerprint": fingerprint,
        }
        if target_column not in constraint.columns:
            return {
                **base,
                "status": "invalid_code",
                "error_type": "FixCodeError",
                "error_message": "target column is not involved in the constraint",
            }
        missing = set(constraint.columns) - set(self.data.columns)
        if missing:
            return {
                **base,
                "status": "invalid_code",
                "error_type": "FixCodeError",
                "error_message": f"data is missing involved columns: {sorted(missing)}",
            }
        try:
            validate_check_code(constraint, set(self.data.columns))
            validate_fix_code(
                code,
                set(constraint.columns) - {target_column},
                target_column,
            )
        except (CheckCodeError, FixCodeError) as exc:
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
            target=_fix_verification_worker,
            args=(
                result_queue,
                self.data,
                constraint,
                target_column,
                code,
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
                "error_message": f"fix(df) exceeded {self.timeout_seconds:g} seconds",
            }
        try:
            result = result_queue.get(timeout=1.0)
        except queue.Empty:
            result = {
                "status": "execution_error",
                "error_type": "WorkerError",
                "error_message": (
                    f"fix verification worker exited with code {process.exitcode}"
                ),
            }
        finally:
            result_queue.close()
        return {**base, **result}
