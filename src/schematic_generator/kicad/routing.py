from __future__ import annotations

import heapq
import math

from schematic_generator.kicad.models import SymbolLayout
from schematic_generator.kicad.sexpr import _orthogonalize_points


class _Router:
    """Grid-based orthogonal router that avoids symbols and already occupied wire cells."""
    def __init__(self, layout: dict[str, SymbolLayout], grid: float = 1.27) -> None:
        """Initialize routing grid bounds, occupied cells, obstacles, and conflict tracking."""
        self.grid = grid
        self.used: dict[tuple[int, int], str] = {}
        self.obstacles: set[tuple[int, int]] = set()
        self._outside_lanes: dict[str, int] = {}
        self.conflicts: list[dict[str, object]] = []
        if layout:
            min_x = min(symbol.bbox[0] for symbol in layout.values()) - 16.0
            max_x = max(symbol.bbox[2] for symbol in layout.values()) + 22.0
            min_y = min(symbol.bbox[1] for symbol in layout.values()) - 16.0
            max_y = max(symbol.bbox[3] for symbol in layout.values()) + 22.0
        else:
            min_x, max_x, min_y, max_y = 0.0, 120.0, 0.0, 120.0
        self.min_x = math.floor(min_x / grid) - 2
        self.max_x = math.ceil(max_x / grid) + 2
        self.min_y = math.floor(min_y / grid) - 2
        self.max_y = math.ceil(max_y / grid) + 2
        for symbol in layout.values():
            self._add_symbol_obstacle(symbol)
    def route(self, start: tuple[float, float], end: tuple[float, float], net: str = "") -> list[tuple[float, float]]:
        """Route an orthogonal path between two schematic points."""
        start_cell = self._cell(start)
        end_cell = self._cell(end)
        cells = self._astar(start_cell, end_cell, net)
        if not cells:
            points = self._fallback(start, end, net)
        else:
            points = [self._point(cell) for cell in self._simplify_cells(cells)]
            points[0] = (round(start[0], 2), round(start[1], 2))
            points[-1] = (round(end[0], 2), round(end[1], 2))
            if not self._points_path_clear(points, start_cell, end_cell, net):
                points = self._fallback(start, end, net)
        self._occupy(points, net)
        return points
    def occupy(self, points: list[tuple[float, float]], net: str = "") -> None:
        """Mark a routed point list as occupied by a net."""
        self._occupy(points, net)
    def occupy_point(self, point: tuple[float, float], net: str = "") -> None:
        """Mark a single schematic point as occupied by a net."""
        self._mark_used(self._cell(point), net)
    def nearest_free_point(self, preferred: tuple[float, float], net: str) -> tuple[float, float]:
        """Find the nearest grid point available for a requested net anchor."""
        start = self._cell(preferred)
        if self._cell_available(start, net):
            return self._point(start)
        for radius in range(1, 20):
            candidates: list[tuple[int, int]] = []
            for dx in range(-radius, radius + 1):
                candidates.append((start[0] + dx, start[1] - radius))
                candidates.append((start[0] + dx, start[1] + radius))
            for dy in range(-radius + 1, radius):
                candidates.append((start[0] - radius, start[1] + dy))
                candidates.append((start[0] + radius, start[1] + dy))
            for cell in sorted(set(candidates), key=lambda c: (abs(c[0] - start[0]) + abs(c[1] - start[1]), c[1], c[0])):
                if self._cell_available(cell, net):
                    return self._point(cell)
        return self._point(start)
    def _cell_available(self, cell: tuple[int, int], net: str) -> bool:
        """Check whether a router cell can be used by a net."""
        if not self._in_bounds(cell):
            return False
        if cell in self.obstacles:
            return False
        return cell not in self.used or self.used[cell] == net
    def _add_symbol_obstacle(self, symbol: SymbolLayout) -> None:
        """Reserve the cells covered by one symbol bounding box."""
        if symbol.kind == "discrete_2pin":
            x1, y1, x2, y2 = symbol.x - 3.8, symbol.y - 3.8, symbol.x + 3.8, symbol.y + 3.8
        elif symbol.kind == "pinrow":
            x1, y1, x2, y2 = symbol.x - 2.8, symbol.y - symbol.height / 2, symbol.x + 2.8, symbol.y + symbol.height / 2
        else:
            x1, y1, x2, y2 = symbol.x - 2.0, symbol.y - 2.0, symbol.x + 2.0, symbol.y + 2.0
        cx1, cy1 = self._cell((x1, y1))
        cx2, cy2 = self._cell((x2, y2))
        for cx in range(min(cx1, cx2), max(cx1, cx2) + 1):
            for cy in range(min(cy1, cy2), max(cy1, cy2) + 1):
                self.obstacles.add((cx, cy))
    def _astar(self, start: tuple[int, int], end: tuple[int, int], net: str) -> list[tuple[int, int]]:
        """Find a low-cost grid path between two cells while avoiding obstacles."""
        queue: list[tuple[float, int, tuple[int, int], tuple[int, int] | None]] = []
        heapq.heappush(queue, (0.0, 0, start, None))
        came_from: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int] | None]] = {}
        cost: dict[tuple[int, int], float] = {start: 0.0}
        counter = 0
        while queue:
            _priority, _counter, current, direction = heapq.heappop(queue)
            if current == end:
                return self._reconstruct_path(came_from, current)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (current[0] + dx, current[1] + dy)
                if not self._in_bounds(nxt):
                    continue
                if nxt in self.obstacles and nxt not in {start, end}:
                    continue
                new_direction = (dx, dy)
                extra_cost = 1.0
                if direction and direction != new_direction:
                    extra_cost += 3.0
                if nxt in self.used and self.used[nxt] != net:
                    continue
                new_cost = cost[current] + extra_cost
                if new_cost >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = new_cost
                came_from[nxt] = (current, new_direction)
                priority = new_cost + abs(nxt[0] - end[0]) + abs(nxt[1] - end[1])
                counter += 1
                heapq.heappush(queue, (priority, counter, nxt, new_direction))
        return []
    def _reconstruct_path(
        self,
        came_from: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int] | None]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Reconstruct a grid path from A* parent links."""
        result = [current]
        while current in came_from:
            current = came_from[current][0]
            result.append(current)
        return list(reversed(result))
    def _fallback(self, start: tuple[float, float], end: tuple[float, float], net: str) -> list[tuple[float, float]]:
        """Build a deterministic orthogonal fallback path when A* cannot find one."""
        start_cell = self._cell(start)
        end_cell = self._cell(end)
        for cells in self._fallback_candidates(start_cell, end_cell):
            if self._cells_path_clear(cells, start_cell, end_cell, net):
                points = [self._point(cell) for cell in self._simplify_cells(cells)]
                points[0] = (round(start[0], 2), round(start[1], 2))
                points[-1] = (round(end[0], 2), round(end[1], 2))
                return points
        outside = self._outside_fallback_cells(start_cell, end_cell, net)
        points = [self._point(cell) for cell in self._simplify_cells(outside)]
        points[0] = (round(start[0], 2), round(start[1], 2))
        points[-1] = (round(end[0], 2), round(end[1], 2))
        return points
    def _fallback_candidates(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[list[tuple[int, int]]]:
        """Generate candidate fallback paths around the current routing area."""
        sx, sy = start
        ex, ey = end
        y_lanes = self._ordered_lanes(sy, ey, self.min_y, self.max_y)
        x_lanes = self._ordered_lanes(sx, ex, self.min_x, self.max_x)
        candidates: list[list[tuple[int, int]]] = []
        for y in y_lanes:
            candidates.append([start, (sx, y), (ex, y), end])
        for x in x_lanes:
            candidates.append([start, (x, sy), (x, ey), end])
        return candidates
    def _outside_fallback_cells(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        net: str,
    ) -> list[tuple[int, int]]:
        """Route around the outside lanes of the drawing when local paths are blocked."""
        lane = self._outside_lanes.setdefault(net, len(self._outside_lanes))
        sx, sy = start
        ex, ey = end
        candidates: list[list[tuple[int, int]]] = []
        for offset in range(lane, lane + 12):
            y_top = self.min_y - 4 - offset * 3
            y_bottom = self.max_y + 4 + offset * 3
            x_left = self.min_x - 4 - offset * 3
            x_right = self.max_x + 4 + offset * 3
            candidates.extend([
                [start, (sx, y_top), (ex, y_top), end],
                [start, (sx, y_bottom), (ex, y_bottom), end],
                [start, (x_left, sy), (x_left, ey), end],
                [start, (x_right, sy), (x_right, ey), end],
            ])
        for cells in candidates:
            if self._cells_path_clear(cells, start, end, net):
                return cells
        return min(candidates, key=lambda cells: self._path_conflict_count(cells, start, end, net))
    def _ordered_lanes(self, a: int, b: int, lo: int, hi: int) -> list[int]:
        """Return candidate lane coordinates ordered by distance from the direct span."""
        center = (a + b) / 2.0
        lanes = [value for value in range(lo, hi + 1) if value not in {a, b}]
        return sorted(lanes, key=lambda value: (abs(value - center), value))
    def _cells_path_clear(
        self,
        cells: list[tuple[int, int]],
        start: tuple[int, int],
        end: tuple[int, int],
        net: str,
    ) -> bool:
        """Check whether every cell on a candidate path can be occupied."""
        endpoints = {start, end}
        for a, b in zip(cells, cells[1:], strict=False):
            for cell in self._cells_on_segment(a, b):
                if cell in self.obstacles:
                    if cell in endpoints:
                        continue
                    return False
                if cell in self.used and self.used[cell] != net:
                    return False
        return True
    def _points_path_clear(
        self,
        points: list[tuple[float, float]],
        start: tuple[int, int],
        end: tuple[int, int],
        net: str,
    ) -> bool:
        """Check whether a schematic-space polyline is available for a net."""
        cells = [self._cell(point) for point in _orthogonalize_points(points)]
        return self._cells_path_clear(cells, start, end, net)
    def _path_conflict_count(
        self,
        cells: list[tuple[int, int]],
        start: tuple[int, int],
        end: tuple[int, int],
        net: str,
    ) -> int:
        """Count how many occupied cells a fallback path would conflict with."""
        endpoints = {start, end}
        conflicts = 0
        for a, b in zip(cells, cells[1:], strict=False):
            endpoint_column = a[0] == b[0] and a[0] in {start[0], end[0]}
            for cell in self._cells_on_segment(a, b):
                if cell in self.obstacles and cell not in endpoints:
                    conflicts += 1
                if cell in self.used and self.used[cell] != net:
                    conflicts += 25 if endpoint_column and cell not in endpoints else 1
        return conflicts
    def _cells_on_segment(self, a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
        """Enumerate grid cells touched by an orthogonal segment."""
        ax, ay = a
        bx, by = b
        if ax == bx:
            return [(ax, y) for y in range(min(ay, by), max(ay, by) + 1)]
        if ay == by:
            return [(x, ay) for x in range(min(ax, bx), max(ax, bx) + 1)]
        return self._cells_on_segment(a, (bx, ay)) + self._cells_on_segment((bx, ay), b)
    def _occupy(self, points: list[tuple[float, float]], net: str = "") -> None:
        """Mark all cells under a routed polyline."""
        points = _orthogonalize_points(points)
        for a, b in zip(points, points[1:]):
            ax, ay = self._cell(a)
            bx, by = self._cell(b)
            if ax == bx:
                for cy in range(min(ay, by), max(ay, by) + 1):
                    self._mark_used((ax, cy), net)
            elif ay == by:
                for cx in range(min(ax, bx), max(ax, bx) + 1):
                    self._mark_used((cx, ay), net)
    def _mark_used(self, cell: tuple[int, int], net: str) -> None:
        """Record one occupied router cell and report cross-net conflicts."""
        previous = self.used.get(cell)
        if previous and previous != net:
            self.conflicts.append({
                "cell": [cell[0], cell[1]],
                "existing_net": previous,
                "new_net": net,
            })
            return
        self.used[cell] = net
    def _simplify_cells(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Remove redundant intermediate cells from an orthogonal grid path."""
        if len(cells) <= 2:
            return cells
        result = [cells[0]]
        previous = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
        for i in range(1, len(cells) - 1):
            direction = (cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1])
            if direction != previous:
                result.append(cells[i])
            previous = direction
        result.append(cells[-1])
        return result
    def _cell(self, point: tuple[float, float]) -> tuple[int, int]:
        """Convert a schematic point to a router grid cell."""
        return int(round(point[0] / self.grid)), int(round(point[1] / self.grid))
    def _point(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Convert a router grid cell to a schematic point."""
        return round(cell[0] * self.grid, 2), round(cell[1] * self.grid, 2)
    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        """Check whether a router cell is within the expanded routing canvas."""
        return self.min_x <= cell[0] <= self.max_x and self.min_y <= cell[1] <= self.max_y
