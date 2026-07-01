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


def _smooth_route_locally(path: list[Point], router: "_Autorouter", net: str) -> list[Point]:
    """Try local routing alternatives that reduce bends and visual clutter."""

    if len(path) < 3:
        return path
    start = path[0]
    end = path[-1]
    candidates = [path]
    if abs(start[0] - end[0]) > 0.001 and abs(start[1] - end[1]) > 0.001:
        candidates.extend([
            [start, (end[0], start[1]), end],
            [start, (start[0], end[1]), end],
        ])
        for ratio in (0.25, 0.5, 0.75):
            x = router.snap_point((start[0] + (end[0] - start[0]) * ratio, start[1]))[0]
            y = router.snap_point((start[0], start[1] + (end[1] - start[1]) * ratio))[1]
            candidates.extend([
                [start, (x, start[1]), (x, end[1]), end],
                [start, (start[0], y), (end[0], y), end],
            ])
    czyste = [
        _normalize_variant(_orthogonalize_points(candidate))
        for candidate in candidates
        if router.points_path_clear(candidate, net)
    ]
    if not czyste:
        return path
    return min(czyste, key=lambda p: (_route_aesthetic_cost(p), len(p), p))


def _route_aesthetic_cost(path: list[Point]) -> float:
    """Score a route path for length, bends, and backtracking."""

    length = sum(_dist(a, b) for a, b in _segments(path))
    return length + 4.0 * _count_bends_in_points(path) + _bend_penalty_near_ends(path)


def _bend_penalty_near_ends(path: list[Point]) -> float:
    """Penalize bends placed very close to route endpoints."""

    if len(path) < 3:
        return 0.0
    distance_from_start = [0.0]
    for a, b in _segments(path):
        distance_from_start.append(distance_from_start[-1] + _dist(a, b))
    calosc = distance_from_start[-1]
    penalty = 0.0
    for index in range(1, len(path) - 1):
        a, b, c = path[index - 1], path[index], path[index + 1]
        if abs(_orient(a, b, c)) <= 0.001:
            continue
        distance = min(distance_from_start[index], calosc - distance_from_start[index])
        penalty += max(0.0, 7.62 - distance) * 1.5
    return penalty

