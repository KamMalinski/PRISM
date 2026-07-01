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


def _junctions_for_routes(route: list[list[Point]], extra: list[Point]) -> list[Point]:
    """Create junction points where generated route paths branch or meet."""

    stopnie: dict[tuple[int, int], int] = defaultdict(int)
    points_by_key: dict[tuple[int, int], Point] = {}
    for trasa in route:
        for a, b in _segments(trasa):
            for point in (a, b):
                key = _key(point)
                stopnie[key] += 1
                points_by_key.setdefault(key, point)
    for point in extra:
        key = _key(point)
        stopnie[key] += 1
        points_by_key.setdefault(key, point)
    return sorted([points_by_key[k] for k, stopien in stopnie.items() if stopien >= 3], key=lambda p: (p[1], p[0]))


def _segments(points: list[Point]) -> list[Segment]:
    """Convert a point polyline into adjacent line segments."""

    return [(a, b) for a, b in zip(points, points[1:], strict=False) if _dist(a, b) > 0.001]


def _segments_intersect(a: Segment, b: Segment) -> bool:
    """Return whether two line segments intersect."""

    (a1, a2), (b1, b2) = a, b
    if _shared_endpoint(a1, a2, b1, b2):
        return False
    return _orient(a1, a2, b1) * _orient(a1, a2, b2) <= 0 and _orient(b1, b2, a1) * _orient(b1, b2, a2) <= 0 and _bbox_overlap(a, b)


def _segment_intersects_bbox(segment: Segment, bbox: tuple[float, float, float, float]) -> bool:
    """Return whether a segment crosses a rectangular bounding box."""

    (x1, y1), (x2, y2) = segment
    bx1, by1, bx2, by2 = bbox
    if min(x1, x2) > bx2 or max(x1, x2) < bx1 or min(y1, y2) > by2 or max(y1, y2) < by1:
        return False
    if bx1 <= x1 <= bx2 and by1 <= y1 <= by2:
        return False
    if bx1 <= x2 <= bx2 and by1 <= y2 <= by2:
        return False
    edges = [((bx1, by1), (bx2, by1)), ((bx2, by1), (bx2, by2)), ((bx2, by2), (bx1, by2)), ((bx1, by2), (bx1, by1))]
    return any(_segments_intersect(segment, edge) for edge in edges)


def _count_bends_in_points(points: list[Point]) -> int:
    """Count direction changes in an orthogonal point path."""

    result = 0
    points = _orthogonalize_points(points)
    for a, b, c in zip(points, points[1:], points[2:], strict=False):
        if abs(_orient(a, b, c)) > 0.001:
            result += 1
    return result


def _count_backtracks_in_points(points: list[Point]) -> int:
    """Count immediate reversals in an orthogonal point path."""

    result = 0
    points = _orthogonalize_points(points)
    for a, b, c in zip(points, points[1:], points[2:], strict=False):
        if _same_point(a, c):
            result += 1
            continue
        if abs(_orient(a, b, c)) <= 0.001:
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            if v1[0] * v2[0] + v1[1] * v2[1] < -0.001:
                result += 1
    return result


def _normalize_variant(points: list[Point]) -> list[Point]:
    """Remove redundant route points and normalize a candidate path."""

    points = _remove_only_duplicates(points)
    mode_changed = True
    while mode_changed:
        mode_changed = False
        result: list[Point] = []
        for point in points:
            result.append(point)
            while len(result) >= 3 and _point_on_segment(result[-2], result[-3], result[-1]):
                result.pop(-2)
                mode_changed = True
            while len(result) >= 3 and _same_point(result[-3], result[-1]):
                result.pop()
                result.pop()
                mode_changed = True
        points = _remove_only_duplicates(result)
    return points


def _remove_only_duplicates(points: list[Point]) -> list[Point]:
    """Remove adjacent duplicate points without otherwise changing a path."""

    result: list[Point] = []
    for point in points:
        point = (round(point[0], 3), round(point[1], 3))
        if not result or _dist(result[-1], point) > 0.001:
            result.append(point)
    return result


def _orthogonalize_points(points: list[Point]) -> list[Point]:
    """Insert bend points so every segment is horizontal or vertical."""

    if len(points) <= 1:
        return points
    result: list[Point] = [points[0]]
    for x2, y2 in points[1:]:
        x1, y1 = result[-1]
        x2, y2 = round(x2, 3), round(y2, 3)
        if abs(x1 - x2) > 0.001 and abs(y1 - y2) > 0.001:
            result.append((x2, y1))
        if result[-1] != (x2, y2):
            result.append((x2, y2))
    return result


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    """Return whether a point lies on a segment within numeric tolerance."""

    px, py = point
    x1, y1 = start
    x2, y2 = end
    if abs(_orient(start, end, point)) > 0.001:
        return False
    return min(x1, x2) - 0.001 <= px <= max(x1, x2) + 0.001 and min(y1, y2) - 0.001 <= py <= max(y1, y2) + 0.001


def _bbox_symbol(symbol: SchematicSymbol, margin: float = 0.0) -> tuple[float, float, float, float]:
    """Return a symbol bounding box with optional margin."""

    return (
        symbol.x - symbol.width / 2 - margin,
        symbol.y - symbol.height / 2 - margin,
        symbol.x + symbol.width / 2 + margin,
        symbol.y + symbol.height / 2 + margin,
    )


def _bboxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Return whether two bounding boxes overlap."""

    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _bbox_overlap(a: Segment, b: Segment) -> bool:
    """Return whether bounding boxes of two segments overlap."""

    (a1, a2), (b1, b2) = a, b
    return not (
        max(a1[0], a2[0]) < min(b1[0], b2[0])
        or max(b1[0], b2[0]) < min(a1[0], a2[0])
        or max(a1[1], a2[1]) < min(b1[1], b2[1])
        or max(b1[1], b2[1]) < min(a1[1], a2[1])
    )


def _orient(a: Point, b: Point, c: Point) -> float:
    """Return the orientation determinant for three points."""

    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _shared_endpoint(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Return whether two segments share one endpoint."""

    return any(_dist(a, b) < 0.001 for a in (a1, a2) for b in (b1, b2))


def _same_point(a: Point, b: Point) -> bool:
    """Compare points using schematic-coordinate tolerance."""

    return _dist(a, b) < 0.001


def _dist(a: Point, b: Point) -> float:
    """Return Euclidean distance between two points."""

    return math.hypot(a[0] - b[0], a[1] - b[1])


def _manhattan(a: Point, b: Point) -> float:
    """Return Manhattan distance between two points."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _snap(value: float, grid: float) -> float:
    """Snap a scalar coordinate to a grid."""

    return round(round(value / grid) * grid, 3)


def _key(point: Point) -> tuple[int, int]:
    """Convert a point to an integer key for stable grouping."""

    return int(round(point[0] * 1000)), int(round(point[1] * 1000))


def _cell_dist(a: Cell, b: Cell) -> int:
    """Return Manhattan distance between routing cells."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _unique_points(points: list[Point]) -> list[Point]:
    """Return unique points while preserving first occurrence order."""

    result: list[Point] = []
    seen: set[tuple[int, int]] = set()
    for point in points:
        key = _key(point)
        if key in seen:
            continue
        seen.add(key)
        result.append((round(point[0], 3), round(point[1], 3)))
    return result


def _points_key(points: list[Point]) -> str:
    """Create a stable text key from a point path."""

    return ";".join(f"{x:.3f},{y:.3f}" for x, y in points)


def _stable_uuid(kind: str, *parts: str) -> str:
    """Create a deterministic UUID for generated KiCad objects."""

    return str(uuid5(NAMESPACE_URL, "schematic_generator.autorouter:" + kind + ":" + ":".join(parts)))


def _txt(value: str) -> str:
    """Escape text for KiCad s-expression output."""

    return str(value).replace("\\", "\\\\").replace('"', '\\"')
