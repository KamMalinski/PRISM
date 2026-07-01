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


def _autoroute(
    nets: list[RoutingNet],
    symbols: list[SchematicSymbol],
    labels_input: list[SchematicLabel],
    junctions_input: list[SchematicJunction],
    stop_event: threading.Event | None,
    routing_iterations: int = 1,
) -> AutoroutingResult:
    """Try multiple routing orders and keep the best autorouting result."""

    best: AutoroutingResult | None = None
    best_cost: tuple[float, ...] | None = None
    for order in _routing_orders(nets, routing_iterations):
        if stop_event and stop_event.is_set() and best is not None:
            break
        result = _autoroute_pass(order, symbols, labels_input, junctions_input, stop_event)
        cost = _cost_autorouting(result, symbols)
        if best is None or best_cost is None or cost < best_cost:
            best = result
            best_cost = cost
    if best is None:
        return _autoroute_pass([], symbols, labels_input, junctions_input, stop_event)
    return best


def _autoroute_pass(
    nets: list[RoutingNet],
    symbols: list[SchematicSymbol],
    labels_input: list[SchematicLabel],
    junctions_input: list[SchematicJunction],
    stop_event: threading.Event | None,
) -> AutoroutingResult:
    """Route all nets in a single order and collect replacement schematic objects."""

    terminals_by_net = {net.name: net.terminals for net in nets}
    router = _Autorouter(symbols, terminals_by_net)
    wires: list[SchematicWire] = []
    labels: list[SchematicLabel] = []
    junctions: list[SchematicJunction] = []
    remove_spans: list[tuple[int, int]] = []
    label_jumps = 0
    index_wire = 0

    for net in sorted(nets, key=lambda n: (-len(n.terminals), n.name)):
        if stop_event and stop_event.is_set():
            break
        for label_index in net.label_indices:
            if 0 <= label_index < len(labels_input):
                remove_spans.append(labels_input[label_index].span)
        for junction_index in net.junction_indices:
            if 0 <= junction_index < len(junctions_input):
                remove_spans.append(junctions_input[junction_index].span)
        route, label_points, points_junctions, jumps = _route_net(router, net)
        label_jumps += jumps
        for points in route:
            points = _normalize_variant(_orthogonalize_points(points))
            if len(points) < 2:
                continue
            wires.append(SchematicWire(
                index_wire,
                points,
                (0, 0),
                _stable_uuid("wire", net.name, str(index_wire), _points_key(points)),
                "",
                True,
                net.name,
            ))
            index_wire += 1
        for i, point in enumerate(label_points):
            labels.append(SchematicLabel(
                len(labels_input) + len(labels),
                net.name,
                point[0],
                point[1],
                (0, 0),
                _stable_uuid("label", net.name, str(i), f"{point[0]:.3f},{point[1]:.3f}"),
                "",
                True,
            ))
        for i, point in enumerate(points_junctions):
            junctions.append(SchematicJunction(
                len(junctions_input) + len(junctions),
                point[0],
                point[1],
                (0, 0),
                _stable_uuid("junction", net.name, str(i), f"{point[0]:.3f},{point[1]:.3f}"),
                "",
            ))

    return AutoroutingResult(wires, labels, junctions, remove_spans, label_jumps)


def _routing_orders(nets: list[RoutingNet], limit: int) -> list[list[RoutingNet]]:
    """Generate deterministic net orders that expose different autorouting tradeoffs."""

    limit = max(1, limit)
    candidates = [
        sorted(nets, key=lambda n: (-len(n.terminals), n.name)),
        sorted(nets, key=lambda n: (len(n.terminals), n.name)),
        sorted(nets, key=lambda n: (-_net_span(n), -len(n.terminals), n.name)),
        sorted(nets, key=lambda n: (_net_span(n), -len(n.terminals), n.name)),
        sorted(nets, key=lambda n: (-_area_bbox_net(n), -len(n.terminals), n.name)),
        sorted(nets, key=lambda n: (_area_bbox_net(n), n.name)),
        sorted(nets, key=lambda n: n.name),
    ]
    base_order = candidates[0]
    for offset in range(1, min(len(base_order), limit)):
        candidates.append(base_order[offset:] + base_order[:offset])
    if len(nets) <= 5:
        candidates.extend([list(permutation) for permutation in itertools.permutations(sorted(nets, key=lambda n: n.name))])

    result: list[list[RoutingNet]] = []
    seen: set[tuple[str, ...]] = set()
    for order in candidates:
        key = tuple(net.name for net in order)
        if key in seen:
            continue
        seen.add(key)
        result.append(order)
        if len(result) >= limit:
            break
    return result


def _net_span(net: RoutingNet) -> float:
    """Measure the maximum point-to-point spread of a routing net."""

    if not net.terminals:
        return 0.0
    xs = [p[0] for p in net.terminals]
    ys = [p[1] for p in net.terminals]
    return max(xs) - min(xs) + max(ys) - min(ys)


def _area_bbox_net(net: RoutingNet) -> float:
    """Measure the bounding-box area covered by routing net terminals."""

    if not net.terminals:
        return 0.0
    xs = [p[0] for p in net.terminals]
    ys = [p[1] for p in net.terminals]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _cost_autorouting(result: AutoroutingResult, symbols: list[SchematicSymbol]) -> tuple[float, ...]:
    """Score an autorouting result by wire quality, label use, and symbol conflicts."""

    metrics = _metrics(symbols, result.wires)
    return (
        float(result.label_jumps),
        float(len(result.labels)),
        float(metrics.crossing_count),
        float(metrics.conflicts_wire_element),
        float(metrics.count_backtracks),
        float(metrics.count_bends),
        round(metrics.length_wires, 3),
        float(len(result.wires)),
    )


def _route_net(router: "_Autorouter", net: RoutingNet) -> tuple[list[list[Point]], list[Point], list[Point], int]:
    """Route one net as a tree connecting terminals and optional label jump points."""

    terminals = [router.snap_point(p) for p in net.terminals]
    terminals = sorted(_unique_points(terminals), key=lambda p: (p[1], p[0]))
    if len(terminals) < 2:
        return [], [], [], 0

    route: list[list[Point]] = []
    label_points: list[Point] = []
    label_jumps = 0
    tree_paths: list[list[Point]] = [[terminals[0]]]
    used_connection_points: list[Point] = []

    pending = terminals[1:]
    while pending:
        best: tuple[tuple[float, ...], int, Point, list[Point]] | None = None
        for index, terminal in enumerate(pending):
            anchor = _best_point_connection_to_tree(terminal, tree_paths, router)
            path = router.plan_route(anchor, terminal, net.name)
            if path is None:
                continue
            path = _normalize_variant(_orthogonalize_points(path))
            path = _smooth_route_locally(path, router, net.name)
            cost = (_route_aesthetic_cost(path), _manhattan(anchor, terminal), terminal[1], terminal[0])
            if best is None or cost < best[0]:
                best = (cost, index, anchor, path)

        if best is None:
            for terminal in pending:
                anchor = _best_point_connection_to_tree(terminal, tree_paths, router)
                _add_label_from_tail(router, net.name, anchor, route, tree_paths, used_connection_points, label_points)
                _add_label_from_tail(router, net.name, terminal, route, tree_paths, used_connection_points, label_points)
                label_jumps += 1
            break

        _cost, index, anchor, path = best
        router.occupy_path(path, net.name)
        _insert_point_to_tree(tree_paths, anchor)
        _insert_point_to_tree(route, anchor)
        route.append(path)
        tree_paths.append(path)
        used_connection_points.append(anchor)
        pending.pop(index)

    if route:
        route = _reduce_route_to_tree(router, route, terminals + label_points)
        router.replace_net_paths(net.name, route)
    if label_points:
        label_points, connected_label_jumps = _try_replace_labels_connections(router, net.name, label_points, route)
        label_jumps = max(0, label_jumps - connected_label_jumps)
        if connected_label_jumps:
            route = _reduce_route_to_tree(router, route, terminals + label_points)
            router.replace_net_paths(net.name, route)

    if label_jumps:
        label_points = sorted(_unique_points(label_points), key=lambda p: (p[1], p[0]))
    else:
        label_points = []

    junctions = _junctions_for_routes(route, used_connection_points)
    return route, label_points, junctions, label_jumps

