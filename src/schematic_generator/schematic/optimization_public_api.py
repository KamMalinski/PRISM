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


def optimize_schematic_automatically(
    path: str | Path,
    log: Callable[[str], None] | None = None,
    time_limit_s: int = 30,
) -> dict[str, object]:
    """Run schematic optimization as an export step and return serializable metadata."""

    path = Path(path)
    try:
        result = optimize_kicad_schematic(
            path,
            OptimizationConfig(time_limit_s=time_limit_s),
        )
    except Exception as exc:
        if log:
            log(f"Schematic optimization: error for {path.name}: {exc}")
        return {
            "status": "error",
            "source": str(path),
            "error": str(exc),
        }

    svg = str(result.svg_path) if result.svg_path else ""
    data = {
        "status": "ok",
        "source": str(path),
        "kicad": str(result.kicad_path),
        "png": str(result.png_path),
        "svg": svg,
        "kicad_exists": result.kicad_path.exists(),
        "png_exists": result.png_path.exists(),
        "svg_exists": bool(result.svg_path and result.svg_path.exists()),
        "improved": bool(result.improved),
        "generations": int(result.generations),
        "score": float(result.result),
        "changed_wire_count": int(result.changed_wire_count),
        "changed_element_count": int(result.changed_element_count),
        "metrics_before": _metrics_to_dict(result.metrics_before),
        "metrics_after": _metrics_to_dict(result.metrics_after),
    }
    if log:
        log(
            "Schematic optimization: saved "
            f"{result.kicad_path.name}, {result.png_path.name}, "
            f"{result.svg_path.name if result.svg_path else 'without SVG'}."
        )
    return data


def _metrics_to_dict(metrics: SchematicMetrics) -> dict[str, int | float]:
    """Convert metrics dataclass fields into JSON-friendly scalar values."""

    return {
        "element_count": int(metrics.count_elements),
        "wire_count": int(metrics.count_wires),
        "wire_length": float(metrics.length_wires),
        "crossing_count": int(metrics.crossing_count),
        "min_element_distance": float(metrics.min_distance_elements),
        "average_nearest_distance": float(metrics.average_distance_nearest),
        "wire_element_conflicts": int(metrics.conflicts_wire_element),
        "bend_count": int(metrics.count_bends),
        "backtrack_count": int(metrics.count_backtracks),
    }


def analyze_kicad_schematic(path: str | Path) -> SchematicMetrics:
    """Parse a KiCad schematic and calculate baseline layout quality metrics."""

    model = _load_model(Path(path))
    return _metrics(model["symbols"], model["wires"])


def optimize_kicad_schematic(
    path: str | Path,
    config: OptimizationConfig,
    progress: ProgressFn | None = None,
    stop_event: threading.Event | None = None,
) -> OptimizationResult:
    """Optimize symbol placement and routing, then save schematic and preview artifacts."""

    path = Path(path)
    model = _load_model(path)
    text = model["text"]
    symbols = model["symbols"]
    wires = model["wires"]
    labels = model["labels"]
    junctions = model["junctions"]
    metrics_before = _metrics(symbols, wires)
    result_before = _score(metrics_before, config)

    if progress:
        progress(OptimizationProgress(0, result_before, result_before, metrics_before))

    if config.time_limit_s <= 0 or stop_event and stop_event.is_set() or not wires:
        return _save_result(
            path,
            text,
            text,
            symbols,
            wires,
            metrics_before,
            metrics_before,
            result_before,
            0,
            result_without_changes=True,
        )

    nets_pin = _build_nets_pin(symbols, wires, labels, junctions)
    if nets_pin:
        symbols_after = _spread_symbols_on_grid(symbols)
        best_labels = labels
        best_junctions: list[SchematicJunction] = []
        nets = _refresh_net_terminals(nets_pin, symbols_after)
    else:
        symbols_after = _spread_symbols_on_grid(symbols)
        attachments = _build_endpoint_attachments(symbols, wires)
        base_wires = _move_wire_endpoints(wires, symbols_after, attachments)
        best_labels = _move_labels_near_symbols(labels, symbols, symbols_after)
        best_junctions = _move_junctions_near_symbols(junctions, symbols, symbols_after)
        nets = _build_nets_to_routing(base_wires, best_labels, best_junctions)
        _assign_wire_nets(base_wires, nets)
    if not nets:
        return _save_result(
            path,
            text,
            text,
            symbols,
            wires,
            metrics_before,
            metrics_before,
            result_before,
            0,
            result_without_changes=True,
        )
    routing_iterations = max(1, int(config.routing_iterations))
    result_routing = _autoroute(nets, symbols_after, best_labels, best_junctions, stop_event, routing_iterations)

    new_text = _replace_schematic(
        text,
        symbols_after,
        best_labels,
        result_routing.labels,
        result_routing.junctions,
        result_routing.remove_spans,
        result_routing.wires,
    )
    metrics_after = _metrics(symbols_after, result_routing.wires)
    best_result = _score(metrics_after, config)
    generations = 1 + len(nets) * routing_iterations
    if progress:
        progress(OptimizationProgress(generations, best_result, best_result, metrics_after))

    return _save_result(
        path,
        text,
        new_text,
        symbols_after,
        result_routing.wires,
        metrics_before,
        metrics_after,
        best_result,
        generations,
        result_without_changes=False,
    )


def _should_accept_autoroute(
    metrics_before: SchematicMetrics,
    metrics_after: SchematicMetrics,
    wires_before: list[SchematicWire],
    result: AutoroutingResult,
    symbols_changed: bool,
) -> bool:
    """Decide whether a routed variant is better enough to replace the original wires."""

    if symbols_changed:
        return True
    if metrics_after.count_wires == 0 and metrics_before.count_wires > 0:
        return metrics_after.conflicts_wire_element < metrics_before.conflicts_wire_element
    if metrics_after.conflicts_wire_element > metrics_before.conflicts_wire_element:
        return False
    if metrics_after.crossing_count > metrics_before.crossing_count:
        return False
    if _has_non_orthogonal_segments(wires_before):
        return True
    if (
        metrics_before.crossing_count == 0
        and metrics_before.conflicts_wire_element == 0
        and metrics_before.count_backtracks == 0
        and metrics_after.count_bends > metrics_before.count_bends + 6
        and metrics_after.length_wires >= metrics_before.length_wires * 0.9
    ):
        return False
    if metrics_after.crossing_count < metrics_before.crossing_count:
        return True
    if metrics_after.count_backtracks < metrics_before.count_backtracks:
        return True
    if metrics_after.conflicts_wire_element < metrics_before.conflicts_wire_element:
        return True
    if result.label_jumps > 0 and metrics_after.crossing_count <= metrics_before.crossing_count:
        return True
    if result.labels and metrics_after.count_bends <= metrics_before.count_bends + 2:
        return True
    if metrics_after.length_wires < metrics_before.length_wires * 0.9 and metrics_after.count_bends <= metrics_before.count_bends + 2:
        return True
    return False


def _has_non_orthogonal_segments(wires: list[SchematicWire]) -> bool:
    """Detect diagonal wire segments that should be repaired by autorouting."""

    for wire in wires:
        for a, b in _segments(wire.points):
            if abs(a[0] - b[0]) > 0.001 and abs(a[1] - b[1]) > 0.001:
                return True
    return False


def _assign_wire_nets(wires: list[SchematicWire], nets: list[RoutingNet]) -> None:
    """Annotate parsed wires with net names derived from routing net membership."""

    for net in nets:
        for index in net.wire_indices:
            if 0 <= index < len(wires):
                wires[index].net = net.name

