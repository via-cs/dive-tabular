"""Deterministic dependency-graph helpers for categorical constraints."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from typing import Any


Adjacency = dict[str, set[str]]


def ensure_nodes(graph: Adjacency, nodes: Iterable[str]) -> None:
    for node in nodes:
        graph.setdefault(node, set())


def find_path(graph: Adjacency, start: str, target: str) -> list[str] | None:
    """Return one deterministic directed path, including both endpoints."""
    if start == target:
        return [start]
    pending = [start]
    parent: dict[str, str | None] = {start: None}
    while pending:
        current = pending.pop(0)
        for neighbor in sorted(graph.get(current, ())):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            if neighbor == target:
                path = [target]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            pending.append(neighbor)
    return None


def cycle_details(
    graph: Adjacency,
    determinants: Iterable[str],
    dependent: str,
) -> list[dict[str, Any]]:
    """Describe cycles that would be closed by determinant-to-dependent edges."""
    details = []
    for determinant in sorted(set(determinants)):
        path = find_path(graph, dependent, determinant)
        if path is not None:
            details.append(
                {
                    "determinant": determinant,
                    "dependent": dependent,
                    "existing_path": path,
                    "cycle_if_accepted": [*path, dependent],
                }
            )
    return details


def add_dependency(
    graph: Adjacency,
    determinants: Iterable[str],
    dependent: str,
) -> None:
    determinants = list(dict.fromkeys(determinants))
    ensure_nodes(graph, [*determinants, dependent])
    for determinant in determinants:
        graph[determinant].add(dependent)


def graph_copy_with_dependency(
    graph: Adjacency,
    determinants: Iterable[str],
    dependent: str,
) -> Adjacency:
    copied = {node: set(neighbors) for node, neighbors in graph.items()}
    add_dependency(copied, determinants, dependent)
    return copied


def topological_order(graph: Adjacency) -> list[str]:
    """Return a stable topological order or raise when the graph is cyclic."""
    nodes = set(graph)
    for neighbors in graph.values():
        nodes.update(neighbors)
    indegree = {node: 0 for node in nodes}
    for neighbors in graph.values():
        for neighbor in neighbors:
            indegree[neighbor] += 1

    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        node = heapq.heappop(ready)
        ordered.append(node)
        for neighbor in sorted(graph.get(node, ())):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(ready, neighbor)
    if len(ordered) != len(nodes):
        raise ValueError("categorical dependency graph contains a cycle")
    return ordered


def direction_result(
    graph: Adjacency,
    determinants: Iterable[str],
    dependent: str,
) -> dict[str, Any]:
    determinants = list(dict.fromkeys(determinants))
    cycles = cycle_details(graph, determinants, dependent)
    edges = [[determinant, dependent] for determinant in determinants]
    if cycles:
        return {
            "status": "rejected_direction",
            "reason": "would_create_cycle",
            "proposed_edges": edges,
            "blocking_paths": cycles,
            "current_topological_order": topological_order(graph),
        }
    prospective = graph_copy_with_dependency(graph, determinants, dependent)
    return {
        "status": "direction_admissible",
        "proposed_edges": edges,
        "topological_order_if_accepted": topological_order(prospective),
    }
