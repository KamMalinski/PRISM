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


class _Autorouter:
    """Maintain routing grid occupancy, symbol obstacles, and A* path planning helpers."""

    def __init__(self, symbols: list[SchematicSymbol], terminals_by_net: dict[str, list[Point]], grid: float = ROUTING_GRID) -> None:
        """Initialize the autorouter grid, occupancy maps, and symbol obstacles."""

        self.grid = grid
        all_points = [p for points in terminals_by_net.values() for p in points]
        if symbols:
            all_points.extend((s.x, s.y) for s in symbols)
        if all_points:
            min_x = min(x for x, _y in all_points) - 30.0
            max_x = max(x for x, _y in all_points) + 30.0
            min_y = min(y for _x, y in all_points) - 30.0
            max_y = max(y for _x, y in all_points) + 30.0
        else:
            min_x, max_x, min_y, max_y = 0.0, 120.0, 0.0, 120.0
        for symbol in symbols:
            x1, y1, x2, y2 = _bbox_symbol(symbol, margin=2.0)
            min_x = min(min_x, x1 - 8.0)
            max_x = max(max_x, x2 + 8.0)
            min_y = min(min_y, y1 - 8.0)
            max_y = max(max_y, y2 + 8.0)
        self.min_x = math.floor(min_x / grid) - 2
        self.max_x = math.ceil(max_x / grid) + 2
        self.min_y = math.floor(min_y / grid) - 2
        self.max_y = math.ceil(max_y / grid) + 2
        self.obstacles: set[Cell] = set()
        self.used: dict[Cell, str] = {}
        self.reserved: dict[Cell, set[str]] = defaultdict(set)
        for symbol in symbols:
            self._add_symbol_obstacle(symbol)
        for net, points in terminals_by_net.items():
            for point in points:
                self.reserved[self.cell(point)].add(net)

    def snap_point(self, point: Point) -> Point:
        """Snap a schematic point to the routing grid."""

        return self.point(self.cell(point))

    def route(self, start: Point, end: Point, net: str) -> list[Point] | None:
        """Plan and reserve a path for one net between two schematic points."""

        points = self.plan_route(start, end, net)
        if points is None:
            return None
        self.occupy_path(points, net)
        return points

    def plan_route(self, start: Point, end: Point, net: str) -> list[Point] | None:
        """Find a path between two points without permanently reserving it."""

        start_cell = self.cell(start)
        end_cell = self.cell(end)
        cells = self._astar(start_cell, end_cell, net)
        if not cells:
            return None
        points = [self.point(cell) for cell in self._simplify_cells(cells)]
        points[0] = (round(start[0], 3), round(start[1], 3))
        points[-1] = (round(end[0], 3), round(end[1], 3))
        points = _normalize_variant(_orthogonalize_points(points))
        if self._length(points) > max(80.0, _manhattan(start, end) * 6.0 + 30.0):
            return None
        return points

    def occupy_path(self, points: list[Point], net: str) -> None:
        """Reserve all cells used by an accepted route path."""

        self._occupy_cells(points, net)

    def replace_net_paths(self, net: str, route: list[list[Point]]) -> None:
        """Replace previously reserved cells for one net with new paths."""

        for cell, used_net in list(self.used.items()):
            if used_net == net:
                del self.used[cell]
        for trasa in route:
            self._occupy_cells(trasa, net)

    def points_path_clear(self, points: list[Point], net: str) -> bool:
        """Check whether a point path can be drawn without hitting obstacles."""

        points = _normalize_variant(_orthogonalize_points(points))
        if len(points) < 2:
            return False
        start = self.cell(points[0])
        end = self.cell(points[-1])
        for a, b in _segments(points):
            for cell in self._cells_on_segment(self.cell(a), self.cell(b)):
                if not self._cell_available(cell, net, start, end):
                    return False
        return True

    def _add_symbol_obstacle(self, symbol: SchematicSymbol) -> None:
        """Mark cells around a symbol as blocked for routing."""

        x1, y1, x2, y2 = _bbox_symbol(symbol, margin=0.8)
        c1 = self.cell((x1, y1))
        c2 = self.cell((x2, y2))
        for cx in range(min(c1[0], c2[0]), max(c1[0], c2[0]) + 1):
            for cy in range(min(c1[1], c2[1]), max(c1[1], c2[1]) + 1):
                self.obstacles.add((cx, cy))

    def _astar(self, start: Cell, end: Cell, net: str) -> list[Cell]:
        """Run A* search on the routing grid between two cells."""

        queue: list[tuple[float, int, Cell, Cell | None]] = []
        heapq.heappush(queue, (0.0, 0, start, None))
        came_from: dict[Cell, tuple[Cell, Cell | None]] = {}
        cost: dict[Cell, float] = {start: 0.0}
        counter = 0
        while queue:
            _priorytet, _counter, current, direction = heapq.heappop(queue)
            if current == end:
                return self._reconstruct_path(came_from, current)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (current[0] + dx, current[1] + dy)
                if not self._cell_available(nxt, net, start, end):
                    continue
                new_direction = (dx, dy)
                dodatkowy = 1.0
                if direction and direction != new_direction:
                    dodatkowy += 4.0
                if self._near_obstacle(nxt):
                    dodatkowy += 0.25
                new_cost = cost[current] + dodatkowy
                if new_cost >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = new_cost
                came_from[nxt] = (current, new_direction)
                priorytet = new_cost + abs(nxt[0] - end[0]) + abs(nxt[1] - end[1])
                counter += 1
                heapq.heappush(queue, (priorytet, counter, nxt, new_direction))
        return []

    def _cell_available(self, cell: Cell, net: str, start: Cell, end: Cell) -> bool:
        """Return whether a grid cell can be used by the current net."""

        if not self._within_bounds(cell):
            return False
        if cell in self.obstacles and _cell_dist(cell, start) > 2 and _cell_dist(cell, end) > 2:
            return False
        reserved = self.reserved.get(cell)
        if reserved and net not in reserved and cell not in {start, end}:
            return False
        used_net = self.used.get(cell)
        if used_net and used_net != net and cell not in {start, end}:
            return False
        return True

    def _near_obstacle(self, cell: Cell) -> bool:
        """Return whether a cell is adjacent to an obstacle and should be penalized."""

        x, y = cell
        return any((x + dx, y + dy) in self.obstacles for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))

    def _reconstruct_path(self, came_from: dict[Cell, tuple[Cell, Cell | None]], current: Cell) -> list[Cell]:
        """Rebuild a cell path from A* predecessor links."""

        result = [current]
        while current in came_from:
            current = came_from[current][0]
            result.append(current)
        return list(reversed(result))

    def _simplify_cells(self, komorki: list[Cell]) -> list[Cell]:
        """Remove unnecessary intermediate cells from a routed cell path."""

        if len(komorki) <= 2:
            return komorki
        result = [komorki[0]]
        previous = (komorki[1][0] - komorki[0][0], komorki[1][1] - komorki[0][1])
        for i in range(1, len(komorki) - 1):
            direction = (komorki[i + 1][0] - komorki[i][0], komorki[i + 1][1] - komorki[i][1])
            if direction != previous:
                result.append(komorki[i])
            previous = direction
        result.append(komorki[-1])
        return result

    def _occupy_cells(self, points: list[Point], net: str) -> None:
        """Reserve cells and edge segments for a net route."""

        for a, b in _segments(_orthogonalize_points(points)):
            ca = self.cell(a)
            cb = self.cell(b)
            for cell in self._cells_on_segment(ca, cb):
                self.used[cell] = net

    def _cells_on_segment(self, a: Cell, b: Cell) -> list[Cell]:
        """Enumerate grid cells touched by a horizontal or vertical segment."""

        if a[0] == b[0]:
            return [(a[0], y) for y in range(min(a[1], b[1]), max(a[1], b[1]) + 1)]
        if a[1] == b[1]:
            return [(x, a[1]) for x in range(min(a[0], b[0]), max(a[0], b[0]) + 1)]
        return self._cells_on_segment(a, (b[0], a[1])) + self._cells_on_segment((b[0], a[1]), b)

    def _length(self, points: list[Point]) -> float:
        """Calculate total Manhattan length of a cell path."""

        return sum(_dist(a, b) for a, b in _segments(points))

    def cell(self, point: Point) -> Cell:
        """Convert a schematic point to a routing grid cell."""

        return int(round(point[0] / self.grid)), int(round(point[1] / self.grid))

    def point(self, cell: Cell) -> Point:
        """Convert a routing grid cell to schematic coordinates."""

        return round(cell[0] * self.grid, 3), round(cell[1] * self.grid, 3)

    def _within_bounds(self, cell: Cell) -> bool:
        """Return whether a grid cell is inside the autorouter bounding box."""

        return self.min_x <= cell[0] <= self.max_x and self.min_y <= cell[1] <= self.max_y

