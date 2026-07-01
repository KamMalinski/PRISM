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


def _align_symbols_to_grid(symbols: list[SchematicSymbol]) -> list[SchematicSymbol]:
    """Snap symbols to the routing grid while preserving their relative movement metadata."""

    result: list[SchematicSymbol] = []
    for symbol in symbols:
        x = _snap(symbol.x, SYMBOL_GRID)
        y = _snap(symbol.y, SYMBOL_GRID)
        dx = round(x - symbol.x, 3)
        dy = round(y - symbol.y, 3)
        result.append(SchematicSymbol(
            symbol.ref,
            x,
            y,
            symbol.rotation,
            symbol.span,
            symbol.source_shape,
            abs(dx) > 0.001 or abs(dy) > 0.001,
            dx,
            dy,
            symbol.width,
            symbol.height,
            symbol.lib_id,
            _move_pins(symbol.pins, dx, dy),
        ))

    placed_symbols: list[SchematicSymbol] = []
    for symbol in sorted(result, key=lambda s: (s.y, s.x, s.ref)):
        candidate = symbol
        moveiecia = 0
        while any(_bboxes_overlap(_bbox_symbol(candidate, 1.5), _bbox_symbol(other, 1.5)) for other in placed_symbols):
            moveiecia += 1
            x = round(candidate.x + SYMBOL_GRID, 3)
            dx = round(x - symbol.x + symbol.dx, 3)
            candidate = SchematicSymbol(
                symbol.ref,
                x,
                candidate.y,
                symbol.rotation,
                symbol.span,
                symbol.source_shape,
                True,
                dx,
                symbol.dy,
                symbol.width,
                symbol.height,
                symbol.lib_id,
                _move_pins(_source_pins(symbol), dx, symbol.dy),
            )
            if moveiecia > 200:
                break
        placed_symbols.append(candidate)
    return sorted(placed_symbols, key=lambda s: s.span[0])


def _spread_symbols_on_grid(symbols: list[SchematicSymbol]) -> list[SchematicSymbol]:
    """Move overlapping symbols apart on the schematic grid."""

    placed_symbols = _align_symbols_to_grid(symbols)
    if len(placed_symbols) < 2:
        return placed_symbols
    by_ref = {symbol.ref: symbol for symbol in placed_symbols}
    for _ in range(250):
        mode_changed = False
        pairs = [
            (a.ref, b.ref)
            for i, a in enumerate(sorted(by_ref.values(), key=lambda s: (s.y, s.x, s.ref)))
            for b in sorted(by_ref.values(), key=lambda s: (s.y, s.x, s.ref))[i + 1:]
        ]
        for ref_a, ref_b in pairs:
            a = by_ref[ref_a]
            b = by_ref[ref_b]
            if not _bboxes_overlap(_bbox_symbol(a, MIN_ROUTE_CHANNEL / 2), _bbox_symbol(b, MIN_ROUTE_CHANNEL / 2)):
                continue
            ax1, ay1, ax2, ay2 = _bbox_symbol(a, MIN_ROUTE_CHANNEL / 2)
            bx1, by1, bx2, by2 = _bbox_symbol(b, MIN_ROUTE_CHANNEL / 2)
            overlap_x = min(ax2, bx2) - max(ax1, bx1)
            overlap_y = min(ay2, by2) - max(ay1, by1)
            if overlap_x <= 0 or overlap_y <= 0:
                continue
            if overlap_x <= overlap_y:
                direction = 1 if b.x >= a.x else -1
                move = max(SYMBOL_GRID, math.ceil((overlap_x + SYMBOL_GRID) / SYMBOL_GRID) * SYMBOL_GRID)
                by_ref[ref_b] = _move_symbol(b, b.x + direction * move, b.y)
            else:
                direction = 1 if b.y >= a.y else -1
                move = max(SYMBOL_GRID, math.ceil((overlap_y + SYMBOL_GRID) / SYMBOL_GRID) * SYMBOL_GRID)
                by_ref[ref_b] = _move_symbol(b, b.x, b.y + direction * move)
            mode_changed = True
        if not mode_changed:
            break
    return sorted(by_ref.values(), key=lambda s: s.span[0])


def _move_symbol(symbol: SchematicSymbol, x: float, y: float) -> SchematicSymbol:
    """Return a copy of a symbol moved to a new location with updated pins and deltas."""

    x = _snap(x, SYMBOL_GRID)
    y = _snap(y, SYMBOL_GRID)
    dx = round(x - _source_symbol_x(symbol), 3)
    dy = round(y - _source_symbol_y(symbol), 3)
    return SchematicSymbol(
        symbol.ref,
        x,
        y,
        symbol.rotation,
        symbol.span,
        symbol.source_shape,
        abs(dx) > 0.001 or abs(dy) > 0.001,
        dx,
        dy,
        symbol.width,
        symbol.height,
        symbol.lib_id,
        _move_pins(_source_pins(symbol), dx, dy),
    )


def _source_symbol_x(symbol: SchematicSymbol) -> float:
    """Recover the original symbol x coordinate before optimization movement."""

    return round(symbol.x - symbol.dx, 3)


def _source_symbol_y(symbol: SchematicSymbol) -> float:
    """Recover the original symbol y coordinate before optimization movement."""

    return round(symbol.y - symbol.dy, 3)


def _source_pins(symbol: SchematicSymbol) -> dict[str, Point]:
    """Recover original pin coordinates for a moved symbol."""

    return _move_pins(symbol.pins, -symbol.dx, -symbol.dy)


def _move_pins(pins: dict[str, Point], dx: float, dy: float) -> dict[str, Point]:
    """Translate all pin coordinates by a fixed delta."""

    return {pin: (round(x + dx, 3), round(y + dy, 3)) for pin, (x, y) in pins.items()}


def _build_endpoint_attachments(symbols: list[SchematicSymbol], wires: list[SchematicWire]) -> list[tuple[int | None, int | None]]:
    """Map wire endpoints to nearby symbols so moved symbols can drag connected wires."""

    return [
        (_nearest_endpoint_symbol(wire.points[0], symbols), _nearest_endpoint_symbol(wire.points[-1], symbols))
        for wire in wires
    ]


def _nearest_endpoint_symbol(point: Point, symbols: list[SchematicSymbol]) -> int | None:
    """Find the closest symbol whose bounding box can own a wire endpoint."""

    best: tuple[float, int] | None = None
    for index, symbol in enumerate(symbols):
        x1, y1, x2, y2 = _bbox_symbol(symbol, margin=9.0)
        distance = _dist(point, (symbol.x, symbol.y))
        if (x1 <= point[0] <= x2 and y1 <= point[1] <= y2) or distance <= 18.0:
            if best is None or distance < best[0]:
                best = (distance, index)
    return best[1] if best else None


def _move_wire_endpoints(
    wires: list[SchematicWire],
    symbols: list[SchematicSymbol],
    attachments: list[tuple[int | None, int | None]],
) -> list[SchematicWire]:
    """Move wire endpoints attached to symbols that changed position."""

    result: list[SchematicWire] = []
    for wire, (start_symbol, end_symbol) in zip(wires, attachments, strict=False):
        points = [tuple(p) for p in wire.points]
        changed = False
        if start_symbol is not None:
            symbol = symbols[start_symbol]
            points[0] = (round(points[0][0] + symbol.dx, 3), round(points[0][1] + symbol.dy, 3))
            changed = changed or abs(symbol.dx) > 0.001 or abs(symbol.dy) > 0.001
        if end_symbol is not None:
            symbol = symbols[end_symbol]
            points[-1] = (round(points[-1][0] + symbol.dx, 3), round(points[-1][1] + symbol.dy, 3))
            changed = changed or abs(symbol.dx) > 0.001 or abs(symbol.dy) > 0.001
        result.append(SchematicWire(wire.index, _remove_only_duplicates(points), wire.span, wire.uuid, wire.source_shape, changed))
    return result


def _move_labels_near_symbols(
    labels: list[SchematicLabel],
    symbols_before: list[SchematicSymbol],
    symbols_after: list[SchematicSymbol],
) -> list[SchematicLabel]:
    """Move labels that were visually attached to symbols before placement changes."""

    result: list[SchematicLabel] = []
    for label in labels:
        index = _nearest_endpoint_symbol((label.x, label.y), symbols_before)
        if index is None:
            result.append(label)
            continue
        symbol = symbols_after[index]
        x = round(label.x + symbol.dx, 3)
        y = round(label.y + symbol.dy, 3)
        result.append(SchematicLabel(
            label.index,
            label.text,
            x,
            y,
            label.span,
            label.uuid,
            label.source_shape,
            abs(symbol.dx) > 0.001 or abs(symbol.dy) > 0.001,
            symbol.dx,
            symbol.dy,
        ))
    return result


def _move_junctions_near_symbols(
    junctions: list[SchematicJunction],
    symbols_before: list[SchematicSymbol],
    symbols_after: list[SchematicSymbol],
) -> list[SchematicJunction]:
    """Move junctions that were visually attached to symbols before placement changes."""

    result: list[SchematicJunction] = []
    for junction in junctions:
        index = _nearest_endpoint_symbol((junction.x, junction.y), symbols_before)
        if index is None:
            result.append(junction)
            continue
        symbol = symbols_after[index]
        result.append(SchematicJunction(
            junction.index,
            round(junction.x + symbol.dx, 3),
            round(junction.y + symbol.dy, 3),
            junction.span,
            junction.uuid,
            junction.source_shape,
        ))
    return result

