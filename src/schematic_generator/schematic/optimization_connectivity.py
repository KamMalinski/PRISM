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


def _build_nets_pin(
    symbols: list[SchematicSymbol],
    wires: list[SchematicWire],
    labels: list[SchematicLabel],
    junctions: list[SchematicJunction],
) -> list[RoutingNet]:
    """Build net connectivity by combining wire contact, labels, junctions, and symbol pins."""

    pins: dict[str, tuple[str, str, Point]] = {}
    for symbol in symbols:
        for pin, point in symbol.pins.items():
            pins[f"pin:{symbol.ref}:{pin}"] = (symbol.ref, pin, point)
    if not pins or not wires:
        return []

    nodes = set(pins)
    nodes.update(f"wire:{i}" for i in range(len(wires)))
    nodes.update(f"label:{label.text}" for label in labels)
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        """Return a union-find representative while compressing the parent chain."""

        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        """Merge two union-find sets used during net reconstruction."""

        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    segments_by_wire = [_segments(wire.points) for wire in wires]
    junction_points = [(junction.x, junction.y) for junction in junctions]

    for i, segments_a in enumerate(segments_by_wire):
        for j in range(i + 1, len(segments_by_wire)):
            if _wires_touch(segments_a, segments_by_wire[j], junction_points):
                union(f"wire:{i}", f"wire:{j}")

    for pin_id, (_ref, _pin, point) in pins.items():
        for index, segments in enumerate(segments_by_wire):
            if any(_point_on_segment(point, a, b) for a, b in segments):
                union(pin_id, f"wire:{index}")

    for label in labels:
        label_id = f"label:{label.text}"
        point = (label.x, label.y)
        for index, segments in enumerate(segments_by_wire):
            if any(_point_on_segment(point, a, b) for a, b in segments):
                union(label_id, f"wire:{index}")
        for pin_id, (_ref, _pin, pin_point) in pins.items():
            if _dist(point, pin_point) < 0.01:
                union(label_id, pin_id)

    groups_pins: dict[str, list[tuple[str, str, Point]]] = defaultdict(list)
    names_groups: dict[str, set[str]] = defaultdict(set)
    for pin_id, pin_data in pins.items():
        groups_pins[find(pin_id)].append(pin_data)
    for label in labels:
        names_groups[find(f"label:{label.text}")].add(label.text)

    result: list[RoutingNet] = []
    number = 1
    for root, pins in sorted(groups_pins.items(), key=lambda item: (min((p[0], p[1]) for p in item[1]), item[0])):
        if len(pins) < 2:
            continue
        names = sorted(names_groups.get(root, set()))
        name = names[0] if names else f"NET_{number:03d}"
        number += 1
        pins = sorted(pins, key=lambda p: (p[0], p[1]))
        result.append(RoutingNet(
            name,
            [point for _ref, _pin, point in pins],
            [],
            [],
            [],
            [(ref, pin) for ref, pin, _point in pins],
        ))
    return result


def _wires_touch(segments_a: list[Segment], segments_b: list[Segment], junction_points: list[Point]) -> bool:
    """Return whether two wire shapes touch through endpoints, crossings, or junctions."""

    for a in segments_a:
        for b in segments_b:
            if _shared_endpoint(a[0], a[1], b[0], b[1]):
                return True
            if _point_on_segment(a[0], b[0], b[1]) or _point_on_segment(a[1], b[0], b[1]):
                return True
            if _point_on_segment(b[0], a[0], a[1]) or _point_on_segment(b[1], a[0], a[1]):
                return True
            if _collinear_segments_overlap(a, b):
                return True
            if any(_point_on_segment(j, a[0], a[1]) and _point_on_segment(j, b[0], b[1]) for j in junction_points):
                return True
    return False


def _collinear_segments_overlap(a: Segment, b: Segment) -> bool:
    """Check whether two collinear wire segments overlap."""

    if abs(_orient(a[0], a[1], b[0])) > 0.001 or abs(_orient(a[0], a[1], b[1])) > 0.001:
        return False
    return _bbox_overlap(a, b)


def _refresh_net_terminals(nets: list[RoutingNet], symbols: list[SchematicSymbol]) -> list[RoutingNet]:
    """Refresh routing net terminals after symbols have moved."""

    by_ref = {symbol.ref: symbol for symbol in symbols}
    result: list[RoutingNet] = []
    for net in nets:
        terminals: list[Point] = []
        pin_refs: list[tuple[str, str]] = []
        for ref, pin in net.pin_refs:
            symbol = by_ref.get(ref)
            if not symbol or pin not in symbol.pins:
                continue
            pin_refs.append((ref, pin))
            terminals.append(symbol.pins[pin])
        if len(terminals) >= 2:
            result.append(RoutingNet(net.name, terminals, net.wire_indices, net.label_indices, net.junction_indices, pin_refs))
    return result


def _build_nets_to_routing(
    wires: list[SchematicWire],
    labels: list[SchematicLabel],
    junctions: list[SchematicJunction],
) -> list[RoutingNet]:
    """Convert parsed schematic connectivity into routable net descriptions."""

    if not wires:
        return []
    parent = list(range(len(wires)))

    def find(a: int) -> int:
        """Return a union-find representative while compressing the parent chain."""

        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        """Merge two union-find sets used during net reconstruction."""

        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    endpoints: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, wire in enumerate(wires):
        endpoints[_key(wire.points[0])].append(index)
        endpoints[_key(wire.points[-1])].append(index)
    for indices in endpoints.values():
        for index in indices[1:]:
            union(indices[0], index)

    segments_by_wire = [_segments(wire.points) for wire in wires]
    for i, wire in enumerate(wires):
        for point in (wire.points[0], wire.points[-1]):
            for j, segments in enumerate(segments_by_wire):
                if i == j:
                    continue
                if any(_point_on_segment(point, a, b) for a, b in segments):
                    union(i, j)

    junction_indices_by_root: dict[int, list[int]] = defaultdict(list)
    for junction in junctions:
        touched = [i for i, segments in enumerate(segments_by_wire) if any(_point_on_segment((junction.x, junction.y), a, b) for a, b in segments)]
        if not touched:
            continue
        for index in touched[1:]:
            union(touched[0], index)
        junction_indices_by_root[find(touched[0])].append(junction.index)

    label_indices_by_root: dict[int, list[int]] = defaultdict(list)
    for label in labels:
        touched = [i for i, segments in enumerate(segments_by_wire) if any(_point_on_segment((label.x, label.y), a, b) for a, b in segments)]
        if touched:
            label_indices_by_root[find(touched[0])].append(label.index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(wires)):
        groups[find(index)].append(index)

    named_groups: dict[str, dict[str, object]] = {}
    without_labels: list[tuple[str, list[int], list[int], list[int]]] = []
    for root, wire_indices in sorted(groups.items()):
        label_indices = sorted(label_indices_by_root.get(root, []))
        junction_indices = sorted(junction_indices_by_root.get(root, []))
        names = sorted({labels[i].text for i in label_indices if 0 <= i < len(labels)})
        if names:
            name = names[0]
            data = named_groups.setdefault(name, {"wires": [], "labels": [], "junctions": []})
            data["wires"].extend(wire_indices)
            data["labels"].extend(label_indices)
            data["junctions"].extend(junction_indices)
        else:
            without_labels.append((f"NET_{len(without_labels) + 1:03d}", wire_indices, label_indices, junction_indices))

    result: list[RoutingNet] = []
    for name, data in sorted(named_groups.items()):
        wire_indices = sorted(set(data["wires"]))
        result.append(RoutingNet(
            name,
            _terminals_net(wires, wire_indices),
            wire_indices,
            sorted(set(data["labels"])),
            sorted(set(data["junctions"])),
        ))
    for name, wire_indices, label_indices, junction_indices in without_labels:
        result.append(RoutingNet(name, _terminals_net(wires, wire_indices), wire_indices, label_indices, junction_indices))
    return [net for net in result if len(net.terminals) >= 2]


def _terminals_net(wires: list[SchematicWire], wire_indices: list[int]) -> list[Point]:
    """Find terminal points for a net from wire endpoints that occur only once."""

    counter: dict[tuple[int, int], int] = defaultdict(int)
    points_by_key: dict[tuple[int, int], Point] = {}
    for index in wire_indices:
        for point in (wires[index].points[0], wires[index].points[-1]):
            key = _key(point)
            counter[key] += 1
            points_by_key.setdefault(key, (round(point[0], 3), round(point[1], 3)))
    terminals = [points_by_key[key] for key, count in counter.items() if count == 1]
    if len(terminals) < 2:
        terminals = list(points_by_key.values())
    return sorted(_unique_points(terminals), key=lambda p: (p[1], p[0]))

