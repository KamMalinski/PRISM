from __future__ import annotations

import heapq
import html
import itertools
import math
import re
import shutil
import subprocess
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

from PIL import Image, ImageDraw, ImageFont

Point = tuple[float, float]
Cell = tuple[int, int]
Segment = tuple[Point, Point]
ProgressFn = Callable[["OptimizationProgress"], None]

SYMBOL_GRID = 2.54
ROUTING_GRID = 1.27
MIN_ROUTE_CHANNEL = 5 * SYMBOL_GRID
EPS = 1e-6


def _try_replace_labels_connections(
    router: "_Autorouter",
    net: str,
    label_points: list[Point],
    route: list[list[Point]],
) -> tuple[list[Point], int]:
    """Replace long route segments with labels when direct routing is not worthwhile."""

    remaining: list[Point] = []
    connected = 0
    i = 0
    while i < len(label_points):
        if i + 1 >= len(label_points):
            remaining.append(label_points[i])
            break
        start = label_points[i]
        end = label_points[i + 1]
        path = router.plan_route(start, end, net)
        if path is None:
            remaining.extend([start, end])
        else:
            path = _normalize_variant(_orthogonalize_points(path))
            path = _smooth_route_locally(path, router, net)
            router.occupy_path(path, net)
            route.append(path)
            connected += 1
        i += 2
    return remaining, connected


def _reduce_route_to_tree(router: "_Autorouter", route: list[list[Point]], required_points: list[Point]) -> list[list[Point]]:
    """Remove route branches that are not needed to connect required points."""

    graph: dict[Cell, set[Cell]] = defaultdict(set)
    for trasa in route:
        for a, b in _segments(_orthogonalize_points(trasa)):
            komorki = router._cells_on_segment(router.cell(a), router.cell(b))
            for c1, c2 in zip(komorki, komorki[1:], strict=False):
                graph[c1].add(c2)
                graph[c2].add(c1)
    if not graph:
        return route

    required = sorted({router.cell(point) for point in required_points if router.cell(point) in graph})
    if len(required) < 2:
        return route

    selected: set[tuple[Cell, Cell]] = set()
    for component in _graph_components(graph):
        required_in_component = [cell for cell in required if cell in component]
        if len(required_in_component) < 2:
            continue
        tree: set[Cell] = {required_in_component[0]}
        for cel in required_in_component[1:]:
            if cel in tree:
                continue
            path = _shortest_path_to_tree(graph, cel, tree, component)
            for c1, c2 in zip(path, path[1:], strict=False):
                selected.add(_edge(c1, c2))
            tree.update(path)

    if not selected:
        return route
    return _route_from_edges(router, selected)


def _graph_components(graph: dict[Cell, set[Cell]]) -> list[set[Cell]]:
    """Split a cell graph into connected components."""

    result: list[set[Cell]] = []
    odwiedzone: set[Cell] = set()
    for start in sorted(graph):
        if start in odwiedzone:
            continue
        component: set[Cell] = set()
        queue = [start]
        odwiedzone.add(start)
        for cell in queue:
            component.add(cell)
            for neighbor in sorted(graph[cell]):
                if neighbor not in odwiedzone:
                    odwiedzone.add(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return result


def _shortest_path_to_tree(graph: dict[Cell, set[Cell]], start: Cell, tree: set[Cell], allowed: set[Cell]) -> list[Cell]:
    """Find the shortest graph path from a cell to an existing route tree."""

    queue = [start]
    parent: dict[Cell, Cell | None] = {start: None}
    znaleziony = start if start in tree else None
    for cell in queue:
        if cell in tree:
            znaleziony = cell
            break
        for neighbor in sorted(graph[cell]):
            if neighbor not in allowed or neighbor in parent:
                continue
            parent[neighbor] = cell
            queue.append(neighbor)
    if znaleziony is None:
        return [start]
    path = [znaleziony]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore[arg-type]
    return list(reversed(path))


def _route_from_edges(router: "_Autorouter", edges: set[tuple[Cell, Cell]]) -> list[list[Point]]:
    """Convert routed grid edges back into schematic point paths."""

    graph: dict[Cell, set[Cell]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    wazne = {cell for cell in graph if _important_route_node(cell, graph)}
    odwiedzone: set[tuple[Cell, Cell]] = set()
    route: list[list[Point]] = []

    for start in sorted(wazne):
        for neighbor in sorted(graph[start]):
            edge = _edge(start, neighbor)
            if edge in odwiedzone:
                continue
            komorki = [start, neighbor]
            odwiedzone.add(edge)
            previous, current = start, neighbor
            while current not in wazne:
                nastepne = [cell for cell in sorted(graph[current]) if cell != previous]
                if not nastepne:
                    break
                kolejny = nastepne[0]
                odwiedzone.add(_edge(current, kolejny))
                komorki.append(kolejny)
                previous, current = current, kolejny
            route.append([router.point(komorki[0]), router.point(komorki[-1])])

    for a, b in sorted(edges):
        edge = _edge(a, b)
        if edge not in odwiedzone:
            route.append([router.point(a), router.point(b)])
    return [_normalize_variant(trasa) for trasa in route if len(trasa) >= 2 and _dist(trasa[0], trasa[-1]) > 0.001]


def _important_route_node(cell: Cell, graph: dict[Cell, set[Cell]]) -> bool:
    """Keep route cells that are endpoints, bends, or branching points."""

    sasiedzi = list(graph[cell])
    if len(sasiedzi) != 2:
        return True
    v1 = (sasiedzi[0][0] - cell[0], sasiedzi[0][1] - cell[1])
    v2 = (sasiedzi[1][0] - cell[0], sasiedzi[1][1] - cell[1])
    return v1[0] + v2[0] != 0 or v1[1] + v2[1] != 0


def _edge(a: Cell, b: Cell) -> tuple[Cell, Cell]:
    """Return a stable undirected edge tuple for two grid cells."""

    return (a, b) if a <= b else (b, a)


def _add_label_from_tail(
    router: "_Autorouter",
    net: str,
    point: Point,
    route: list[list[Point]],
    tree_paths: list[list[Point]],
    used_connection_points: list[Point],
    label_points: list[Point],
) -> None:
    """Create a net label at the end of a route tail and return the used points."""

    label_point, stub = _find_label_point_near(router, point, net)
    if stub:
        router.occupy_path(stub, net)
        _insert_point_to_tree(tree_paths, point)
        _insert_point_to_tree(route, point)
        route.append(stub)
        tree_paths.append(stub)
        used_connection_points.append(point)
    label_points.append(label_point)


def _find_label_point_near(router: "_Autorouter", point: Point, net: str) -> tuple[Point, list[Point]]:
    """Find a nearby free location for a net label and its short connection stub."""

    point = router.snap_point(point)
    kierunki = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    candidates: list[tuple[tuple[float, ...], Point, list[Point]]] = []
    for kroki in (2, 3, 4, 5):
        for dx, dy in kierunki:
            candidate = router.snap_point((point[0] + dx * router.grid * kroki, point[1] + dy * router.grid * kroki))
            if _same_point(candidate, point):
                continue
            cell = router.cell(candidate)
            if cell in router.obstacles:
                continue
            stub = _normalize_variant(_orthogonalize_points([point, candidate]))
            if not router.points_path_clear(stub, net):
                continue
            cost = (
                1.0 if router._near_obstacle(cell) else 0.0,
                0.0 if dy == 0 else 1.0,
                float(kroki),
                candidate[1],
                candidate[0],
            )
            candidates.append((cost, candidate, stub))
    if not candidates:
        return point, []
    _cost, candidate, stub = min(candidates, key=lambda item: item[0])
    return candidate, stub


def _best_point_connection_to_tree(terminal: Point, tree_paths: list[list[Point]], router: "_Autorouter") -> Point:
    """Choose the best anchor point for connecting a terminal to an existing tree."""

    candidates: list[Point] = []
    for path in tree_paths:
        if len(path) == 1:
            candidates.append(path[0])
            continue
        for a, b in _segments(path):
            candidates.append(_manhattan_projection_to_segment(terminal, a, b, router))
            candidates.extend([a, b])
    candidates = _unique_points([router.snap_point(p) for p in candidates])
    return min(candidates, key=lambda p: (_cost_point_connection(terminal, p), p[1], p[0]))


def _manhattan_projection_to_segment(point: Point, a: Point, b: Point, router: "_Autorouter") -> Point:
    """Project a point onto an orthogonal segment using Manhattan geometry."""

    if abs(a[0] - b[0]) < 0.001:
        y = min(max(point[1], min(a[1], b[1])), max(a[1], b[1]))
        return router.snap_point((a[0], y))
    if abs(a[1] - b[1]) < 0.001:
        x = min(max(point[0], min(a[0], b[0])), max(a[0], b[0]))
        return router.snap_point((x, a[1]))
    return a


def _cost_point_connection(terminal: Point, anchor: Point) -> float:
    """Estimate the cost of connecting a terminal to a candidate anchor."""

    bends = 0 if abs(terminal[0] - anchor[0]) < 0.001 or abs(terminal[1] - anchor[1]) < 0.001 else 1
    near_pin = 3.0 if _manhattan(terminal, anchor) < 3.0 else 0.0
    return _manhattan(terminal, anchor) + bends * 5.0 + near_pin


def _insert_point_to_tree(tree_paths: list[list[Point]], point: Point) -> None:
    """Insert a point into an existing tree path when it lies on a segment."""

    for i, path in enumerate(tree_paths):
        if len(path) < 2:
            continue
        result: list[Point] = [path[0]]
        mode_changed = False
        for a, b in zip(path, path[1:], strict=False):
            if _point_on_segment(point, a, b) and not _same_point(point, a) and not _same_point(point, b):
                result.append(point)
                mode_changed = True
            result.append(b)
        if mode_changed:
            tree_paths[i] = _remove_only_duplicates(result)

