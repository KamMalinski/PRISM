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


def _replace_schematic(
    text: str,
    symbols: list[SchematicSymbol],
    labels_input: list[SchematicLabel],
    new_labels: list[SchematicLabel],
    new_junctions: list[SchematicJunction],
    extra_remove_spans: list[tuple[int, int]],
    new_wires: list[SchematicWire],
) -> str:
    """Replace moved symbols and generated routes inside the KiCad schematic text."""

    replacements: list[tuple[int, int, str]] = []
    remove = [span for span in extra_remove_spans if span != (0, 0)]
    remove.extend(wire.span for wire in _load_wires(text))
    remove.extend(label.span for label in labels_input)
    remove.extend(junction.span for junction in _load_junctions(text))
    remove = _unique_spans(remove)
    for start, end in remove:
        replacements.append((start, end, ""))
    for symbol in symbols:
        if symbol.source_shape:
            replacements.append((symbol.span[0], symbol.span[1], _symbol_shape(symbol)))
    result = text
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        result = result[:start] + replacement + result[end:]

    new_forms: list[str] = []
    for wire in new_wires:
        new_forms.append(_wire_shape(wire))
    for junction in new_junctions:
        new_forms.append(_junction_shape(junction))
    for label in new_labels:
        new_forms.append(_label_shape(label))
    if new_forms:
        insert_at = result.rfind("\n)")
        block = "\n" + "\n".join(new_forms)
        if insert_at >= 0:
            result = result[:insert_at] + block + result[insert_at:]
        else:
            result += block + "\n"
    return _remove_empty_lines(result)


def _symbol_shape(symbol: SchematicSymbol) -> str:
    """Serialize a moved symbol back into a KiCad symbol form."""

    if not symbol.source_shape:
        return symbol.source_shape

    def move(match: re.Match[str]) -> str:
        """Rewrite one regex match while preserving unrelated schematic text."""

        x = float(match.group(1)) + symbol.dx
        y = float(match.group(2)) + symbol.dy
        return f"(at {x:.3f} {y:.3f}{match.group(3)})"

    result = re.sub(
        r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)([^)]*)\)",
        move,
        symbol.source_shape,
    )
    return re.sub(
        r'(\(property\s+"Reference"\s+"[^"]*"\s+\(at\s+)([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)([^)]*)\)',
        lambda match: f"{match.group(1)}{symbol.x:.3f} {symbol.y:.3f}{match.group(4)})",
        result,
        count=1,
    )


def _wire_shape(wire: SchematicWire) -> str:
    """Serialize a routed wire as a KiCad wire form."""

    forms: list[str] = []
    for index, (start, end) in enumerate(_segments(_orthogonalize_points(wire.points))):
        uid = wire.uuid if index == 0 else _stable_uuid("wire-segment", wire.uuid, str(index))
        forms.append("\n".join([
            "  (wire",
            f"    (pts (xy {start[0]:.3f} {start[1]:.3f}) (xy {end[0]:.3f} {end[1]:.3f}))",
            "    (stroke (width 0) (type default))",
            f'    (uuid "{uid}")',
            "  )",
        ]))
    return "\n".join(forms)


def _label_shape(label: SchematicLabel) -> str:
    """Serialize a generated net label as a KiCad label form."""

    if label.source_shape and label.changed:
        return re.sub(
            r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)([^)]*)\)",
            f"(at {label.x:.3f} {label.y:.3f}\\3)",
            label.source_shape,
        )
    return "\n".join([
        f'  (label "{_txt(label.text)}"',
        f"    (at {label.x:.3f} {label.y:.3f} 0)",
        "    (effects (font (size 1.27 1.27)))",
        f'    (uuid "{label.uuid}")',
        "  )",
    ])


def _junction_shape(junction: SchematicJunction) -> str:
    """Serialize a generated junction as a KiCad junction form."""

    return "\n".join([
        "  (junction",
        f"    (at {junction.x:.3f} {junction.y:.3f})",
        "    (diameter 0)",
        "    (color 0 0 0 0)",
        f'    (uuid "{junction.uuid}")',
        "  )",
    ])


def _remove_empty_lines(text: str) -> str:
    """Collapse repeated blank lines in generated schematic text."""

    lines = text.splitlines()
    result: list[str] = []
    previous_empty = False
    for line in lines:
        empty = not line.strip()
        if empty and previous_empty:
            continue
        result.append(line)
        previous_empty = empty
    return "\n".join(result).rstrip() + "\n"


def _unique_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return sorted non-overlapping source spans for text replacement."""

    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for span in spans:
        if span in seen:
            continue
        seen.add(span)
        result.append(span)
    return result


def _metrics(symbols: list[SchematicSymbol], wires: list[SchematicWire]) -> SchematicMetrics:
    """Compute schematic readability metrics from symbols and wires."""

    segments_by_wire = [_segments(p.points) for p in wires]
    segments = [segment for group in segments_by_wire for segment in group]
    length = sum(_dist(a, b) for a, b in segments)
    bend_count = sum(_count_bends_in_points(p.points) for p in wires)
    backtrack_count = sum(_count_backtracks_in_points(p.points) for p in wires)
    crossing_count = 0
    for i, group_a in enumerate(segments_by_wire):
        for j in range(i + 1, len(segments_by_wire)):
            if wires[i].net and wires[i].net == wires[j].net:
                continue
            for a in group_a:
                for b in segments_by_wire[j]:
                    if _segments_intersect(a, b):
                        crossing_count += 1
    distances = [
        _dist((a.x, a.y), (b.x, b.y))
        for i, a in enumerate(symbols)
        for b in symbols[i + 1:]
    ]
    min_distance = min(distances) if distances else 0.0
    nearest_distances: list[float] = []
    for symbol in symbols:
        other_distances = [_dist((symbol.x, symbol.y), (second.x, second.y)) for second in symbols if second is not symbol]
        if other_distances:
            nearest_distances.append(min(other_distances))
    conflicts = sum(1 for seg in segments for symbol in symbols if _segment_intersects_bbox(seg, _bbox_symbol(symbol)))
    return SchematicMetrics(
        len(symbols),
        len(wires),
        round(length, 3),
        crossing_count,
        round(min_distance, 3),
        round(sum(nearest_distances) / len(nearest_distances), 3) if nearest_distances else 0.0,
        conflicts,
        bend_count,
        backtrack_count,
    )


def _score(metrics: SchematicMetrics, config: OptimizationConfig) -> float:
    """Convert layout metrics into a scalar optimizer score."""

    nearby = max(0.0, 12.0 - metrics.min_distance_elements) if metrics.count_elements > 1 else 0.0
    return (
        config.reward_distances * metrics.average_distance_nearest
        - config.penalty_length * metrics.length_wires
        - config.penalty_crossings * metrics.crossing_count
        - config.penalty_wire_element * metrics.conflicts_wire_element
        - config.penalty_nearby_elements * nearby
        - 1.5 * metrics.count_bends
        - 80.0 * metrics.count_backtracks
    )

