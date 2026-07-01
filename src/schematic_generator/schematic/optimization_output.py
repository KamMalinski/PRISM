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


def _save_result(
    path: Path,
    source_text: str,
    text_result: str,
    symbols: list[SchematicSymbol],
    wires: list[SchematicWire],
    metrics_before: SchematicMetrics,
    metrics_after: SchematicMetrics,
    result: float,
    generations: int,
    result_without_changes: bool,
) -> OptimizationResult:
    """Write optimized schematic, previews, and final optimization metadata."""

    base = path.with_name(f"{path.stem}_optimized")
    kicad_path = base.with_suffix(".kicad_sch")
    png_path = base.with_suffix(".png")
    kicad_path.write_text(text_result if not result_without_changes else source_text, encoding="utf-8")
    svg_path = _export_kicad_svg(kicad_path)
    if svg_path is None:
        svg_path = _save_svg_work(kicad_path.with_suffix(".svg"), symbols, wires)
    else:
        _trim_svg_to_schematic(svg_path, symbols, wires)
    changed_wire_count = 0 if result_without_changes else sum(1 for wire in wires if wire.changed)
    changed_element_count = 0 if result_without_changes else sum(1 for symbol in symbols if symbol.changed)
    improved = not result_without_changes and (text_result != source_text)
    _save_png(png_path, symbols, wires, path, svg_path, result_without_changes)
    return OptimizationResult(
        kicad_path,
        png_path,
        svg_path,
        metrics_before,
        metrics_after,
        result,
        generations,
        improved,
        changed_wire_count,
        changed_element_count,
    )


def _export_kicad_svg(kicad_path: Path) -> Path | None:
    """Ask KiCad CLI to export an SVG preview of the optimized schematic."""

    try:
        from schematic_generator.schematic.validation import find_kicad_cli

        cli = find_kicad_cli()
        if not cli:
            return None
        result = _run([
            str(cli),
            "sch",
            "export",
            "svg",
            "--output",
            str(kicad_path.parent),
            str(kicad_path),
        ])
        if result.returncode != 0:
            return None
        expected = kicad_path.with_suffix(".svg")
        if expected.exists():
            return expected
        candidates = sorted(kicad_path.parent.glob(f"{kicad_path.stem}*.svg"))
        return candidates[0] if candidates else None
    except Exception:
        return None


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture its text output for diagnostics."""

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "check": False,
    }
    if creationflags:
        kwargs["creationflags"] = creationflags
    return subprocess.run(args, **kwargs)


def _save_png(
    path: Path,
    symbols: list[SchematicSymbol],
    wires: list[SchematicWire],
    path_source: Path,
    svg_path: Path | None,
    result_without_changes: bool,
) -> None:
    """Create a PNG preview using KiCad SVG export when available, otherwise local drawing."""

    if result_without_changes:
        source_png = _neighboring_png(path_source)
        if source_png:
            shutil.copy2(source_png, path)
            return
    if svg_path and _convert_svg_to_png(svg_path, path):
        return
    _save_png_work(path, symbols, wires)


def _neighboring_png(path_source: Path) -> Path | None:
    """Find a PNG generated next to a source file by external conversion tools."""

    candidate = path_source.with_suffix(".png")
    return candidate if candidate.exists() else None


def _convert_svg_to_png(svg_path: Path, png_path: Path) -> bool:
    """Convert an SVG preview to PNG through available local libraries."""

    converters = [
        ["magick", str(svg_path), str(png_path)],
        ["rsvg-convert", "-o", str(png_path), str(svg_path)],
    ]
    for args in converters:
        if shutil.which(args[0]) is None:
            continue
        try:
            result = _run(args)
        except Exception:
            continue
        if result.returncode == 0 and png_path.exists():
            return True
    return False


def _trim_svg_to_schematic(
    path: Path,
    symbols: list[SchematicSymbol],
    wires: list[SchematicWire],
) -> None:
    """Trim an exported KiCad SVG to the schematic drawing bounds."""

    bbox = _svg_model_bbox(symbols, wires, margin=3.0)
    if bbox is None:
        return
    min_x, min_y, max_x, max_y = bbox
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    text = path.read_text(encoding="utf-8")
    new = re.sub(
        r'width="[^"]+"\s+height="[^"]+"\s+viewBox="[^"]+"',
        f'width="{width:.4f}mm" height="{height:.4f}mm" viewBox="{min_x:.4f} {min_y:.4f} {width:.4f} {height:.4f}"',
        text,
        count=1,
    )
    if new != text:
        path.write_text(new, encoding="utf-8")
    _remove_svg_side_frame(path)


def _remove_svg_side_frame(path: Path) -> None:
    """Remove page-frame elements from a KiCad SVG export."""

    try:
        tree = ET.parse(path)
    except Exception:
        return
    root = tree.getroot()
    for child in list(root):
        tag = child.tag.split("}", 1)[-1]
        style = child.attrib.get("style", "")
        if tag != "g" or "stroke:#840000" not in style:
            continue
        serialized = ET.tostring(child, encoding="unicode")
        if "M10.0000 10.0000" in serialized or "M12.0000 12.0000" in serialized:
            root.remove(child)
            try:
                tree.write(path, encoding="utf-8", xml_declaration=True)
            except Exception:
                return
            return


def _svg_model_bbox(
    symbols: list[SchematicSymbol],
    wires: list[SchematicWire],
    margin: float = 3.0,
) -> tuple[float, float, float, float] | None:
    """Estimate the drawing bounding box for exported SVG preview trimming."""

    points: list[Point] = []
    for symbol in symbols:
        x1, y1, x2, y2 = _bbox_symbol(symbol, margin=4.0)
        points.extend([(x1, y1), (x2, y2)])
        points.extend(symbol.pins.values())
    for wire in wires:
        points.extend(wire.points)
    if not points:
        return None
    min_x = min(x for x, _y in points) - margin
    max_x = max(x for x, _y in points) + margin
    min_y = min(y for _x, y in points) - margin
    max_y = max(y for _x, y in points) + margin
    return min_x, min_y, max_x, max_y

