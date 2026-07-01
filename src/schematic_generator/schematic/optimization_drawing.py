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


def _save_svg_work(
    path: Path,
    symbols: list[SchematicSymbol],
    wires: list[SchematicWire],
) -> Path | None:
    """Write a lightweight SVG preview generated from parsed schematic geometry."""

    points: list[Point] = [(s.x, s.y) for s in symbols]
    for symbol in symbols:
        x1, y1, x2, y2 = _bbox_symbol(symbol, margin=4.0)
        points.extend([(x1, y1), (x2, y2)])
        points.extend(symbol.pins.values())
    for wire in wires:
        points.extend(wire.points)
    if not points:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500">'
            '<rect width="100%" height="100%" fill="white"/></svg>\n',
            encoding="utf-8",
        )
        return path

    min_x = min(x for x, _y in points) - 15
    max_x = max(x for x, _y in points) + 15
    min_y = min(y for _x, y in points) - 15
    max_y = max(y for _x, y in points) + 15
    skala = max(4.0, min(12.0, 1600 / max(max_x - min_x, max_y - min_y, 1)))
    width = max(800, int((max_x - min_x) * skala))
    height = max(500, int((max_y - min_y) * skala))

    def p(point: Point) -> tuple[float, float]:
        """Map schematic coordinates into preview image coordinates."""

        return (point[0] - min_x) * skala, (point[1] - min_y) * skala

    stroke_width = max(1.0, skala * 0.18)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g fill="none" stroke="#3c3c3c" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for wire in wires:
        for a, b in _segments(wire.points):
            ax, ay = p(a)
            bx, by = p(b)
            lines.append(f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" stroke-width="{stroke_width:.2f}"/>')
    lines.append("</g>")
    lines.append('<g stroke="#1e1e1e" fill="white" stroke-width="1.4">')
    for symbol in symbols:
        x1, y1, x2, y2 = _bbox_symbol(symbol)
        sx1, sy1 = p((x1, y1))
        sx2, sy2 = p((x2, y2))
        lines.append(
            f'<rect x="{sx1:.2f}" y="{sy1:.2f}" width="{sx2 - sx1:.2f}" '
            f'height="{sy2 - sy1:.2f}" rx="0" ry="0"/>'
        )
        for point in symbol.pins.values():
            px, py = p(point)
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{max(2.0, stroke_width + 1):.2f}"/>')
    lines.append("</g>")
    lines.append('<g fill="#141414" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" dominant-baseline="middle">')
    for symbol in symbols:
        sx, sy = p((symbol.x, symbol.y))
        lines.append(f'<text x="{sx:.2f}" y="{sy:.2f}">{html.escape(symbol.ref)}</text>')
    lines.append("</g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_png_work(path: Path, symbols: list[SchematicSymbol], wires: list[SchematicWire]) -> None:
    """Draw a lightweight PNG preview from parsed schematic geometry."""

    points: list[Point] = [(s.x, s.y) for s in symbols]
    for symbol in symbols:
        x1, y1, x2, y2 = _bbox_symbol(symbol, margin=4.0)
        points.extend([(x1, y1), (x2, y2)])
        points.extend(symbol.pins.values())
    for wire in wires:
        points.extend(wire.points)
    if not points:
        Image.new("RGB", (800, 500), "white").save(path)
        return
    min_x = min(x for x, _y in points) - 15
    max_x = max(x for x, _y in points) + 15
    min_y = min(y for _x, y in points) - 15
    max_y = max(y for _x, y in points) + 15
    skala = max(4.0, min(12.0, 1600 / max(max_x - min_x, max_y - min_y, 1)))
    image = Image.new("RGB", (max(800, int((max_x - min_x) * skala)), max(500, int((max_y - min_y) * skala))), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def p(point: Point) -> tuple[int, int]:
        """Map schematic coordinates into preview image coordinates."""

        return int(round((point[0] - min_x) * skala)), int(round((point[1] - min_y) * skala))

    width = max(2, int(skala * 0.18))
    for wire in wires:
        for a, b in _segments(wire.points):
            draw.line([p(a), p(b)], fill=(60, 60, 60), width=width)
    for symbol in symbols:
        _draw_symbol_png(draw, p, font, symbol, width)
    image.save(path)


def _draw_symbol_png(
    draw: ImageDraw.ImageDraw,
    p: Callable[[Point], tuple[int, int]],
    font: ImageFont.ImageFont,
    symbol: SchematicSymbol,
    wire_width: int,
) -> None:
    """Draw one schematic symbol in the fallback PNG preview."""

    stroke = (30, 30, 30)
    text = (20, 20, 20)
    body_width = max(1, wire_width // 2)
    pin_width = max(2, wire_width)
    for pin, point in sorted(symbol.pins.items(), key=lambda item: item[0]):
        _draw_pin_png(draw, p, symbol, pin, point, pin_width)

    if symbol.lib_id.endswith(":Resistor") or symbol.lib_id == "Device:R":
        _draw_resistor_png(draw, p, symbol, stroke, body_width)
    elif symbol.lib_id.endswith(":Capacitor") or symbol.lib_id in {"Device:C", "Device:CP"}:
        _draw_capacitor_png(draw, p, symbol, stroke, body_width)
    elif symbol.lib_id.endswith(":Diode") or symbol.lib_id.startswith("Device:D"):
        _draw_diode_png(draw, p, symbol, stroke, body_width)
    elif symbol.lib_id.endswith(":Inductor") or symbol.lib_id.startswith("Device:L"):
        _draw_inductor_png(draw, p, symbol, stroke, body_width)
    elif "PinRow" in symbol.lib_id:
        _draw_pinrow_png(draw, p, symbol, stroke, body_width)
    elif symbol.lib_id.endswith(":TP"):
        _draw_testpoint_png(draw, p, symbol, stroke, body_width)
    else:
        _draw_fallback_symbol_png(draw, p, symbol, stroke, body_width)
    _draw_reference_png(draw, p, font, symbol, text)


def _draw_reference_png(
    draw: ImageDraw.ImageDraw,
    p: Callable[[Point], tuple[int, int]],
    font: ImageFont.ImageFont,
    symbol: SchematicSymbol,
    color: tuple[int, int, int],
) -> None:
    """Draw a symbol reference designator near its fallback preview shape."""

    cx, cy = p((symbol.x, symbol.y))
    bbox = draw.textbbox((0, 0), symbol.ref, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = cx - width // 2
    y = cy - height // 2
    padding = 1
    draw.rectangle((x - padding, y - padding, x + width + padding, y + height + padding), fill=(255, 255, 255))
    draw.text((x, y), symbol.ref, fill=color, font=font)


def _draw_pin_png(
    draw: ImageDraw.ImageDraw,
    p: Callable[[Point], tuple[int, int]],
    symbol: SchematicSymbol,
    pin: str,
    point: Point,
    width: int,
) -> None:
    """Draw one pin stub in the fallback PNG preview."""

    sx, sy = symbol.x, symbol.y
    px, py = point
    dx = px - sx
    dy = py - sy
    if abs(dx) >= abs(dy):
        body = (sx + math.copysign(symbol.width / 2, dx or 1.0), py)
    else:
        body = (px, sy + math.copysign(symbol.height / 2, dy or 1.0))
    draw.line([p(point), p(body)], fill=(30, 30, 30), width=width)
    x, y = p(point)
    r = max(2, width + 1)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(30, 30, 30), width=max(1, width // 2), fill=(255, 255, 255))


def _draw_resistor_png(draw: ImageDraw.ImageDraw, p: Callable[[Point], tuple[int, int]], symbol: SchematicSymbol, color: tuple[int, int, int], width: int) -> None:
    """Draw a resistor body in the fallback PNG preview."""

    pts = [
        _symbol_point(symbol, -2.54, 0.0),
        _symbol_point(symbol, -1.9, -1.0),
        _symbol_point(symbol, -1.1, 1.0),
        _symbol_point(symbol, -0.3, -1.0),
        _symbol_point(symbol, 0.5, 1.0),
        _symbol_point(symbol, 1.3, -1.0),
        _symbol_point(symbol, 2.1, 1.0),
        _symbol_point(symbol, 2.54, 0.0),
    ]
    draw.line([p(pt) for pt in pts], fill=color, width=max(1, width))


def _draw_capacitor_png(draw: ImageDraw.ImageDraw, p: Callable[[Point], tuple[int, int]], symbol: SchematicSymbol, color: tuple[int, int, int], width: int) -> None:
    """Draw a capacitor body in the fallback PNG preview."""

    draw.line([p(_symbol_point(symbol, -0.8, -2.0)), p(_symbol_point(symbol, -0.8, 2.0))], fill=color, width=max(1, width))
    draw.line([p(_symbol_point(symbol, 0.8, -2.0)), p(_symbol_point(symbol, 0.8, 2.0))], fill=color, width=max(1, width))


def _draw_diode_png(draw: ImageDraw.ImageDraw, p: Callable[[Point], tuple[int, int]], symbol: SchematicSymbol, color: tuple[int, int, int], width: int) -> None:
    """Draw a diode body in the fallback PNG preview."""

    tri = [
        p(_symbol_point(symbol, -1.8, -1.8)),
        p(_symbol_point(symbol, -1.8, 1.8)),
        p(_symbol_point(symbol, 1.2, 0.0)),
        p(_symbol_point(symbol, -1.8, -1.8)),
    ]
    draw.line(tri, fill=color, width=max(1, width))
    draw.line([p(_symbol_point(symbol, 1.2, -1.8)), p(_symbol_point(symbol, 1.2, 1.8))], fill=color, width=max(1, width))


def _draw_inductor_png(draw: ImageDraw.ImageDraw, p: Callable[[Point], tuple[int, int]], symbol: SchematicSymbol, color: tuple[int, int, int], width: int) -> None:
    """Draw an inductor body in the fallback PNG preview."""

    for i in range(4):
        x0 = symbol.x - 2.4 + i * 1.2
        if int(symbol.rotation) % 180 == 0:
            px1, py1 = p((x0, symbol.y - 1.0))
            px2, py2 = p((x0 + 1.2, symbol.y + 1.0))
            draw.arc((px1, py1, px2, py2), 180, 360, fill=color, width=max(1, width))
        else:
            y0 = symbol.y - 2.4 + i * 1.2
            px1, py1 = p((symbol.x - 1.0, y0))
            px2, py2 = p((symbol.x + 1.0, y0 + 1.2))
            draw.arc((px1, py1, px2, py2), 90, 270, fill=color, width=max(1, width))


def _draw_pinrow_png(draw: ImageDraw.ImageDraw, p: Callable[[Point], tuple[int, int]], symbol: SchematicSymbol, color: tuple[int, int, int], width: int) -> None:
    """Draw a connector pin row in the fallback PNG preview."""

    x1, y1, x2, y2 = _bbox_symbol(symbol)
    draw.rectangle((*p((x1, y1)), *p((x2, y2))), outline=color, width=max(1, width))


def _draw_testpoint_png(draw: ImageDraw.ImageDraw, p: Callable[[Point], tuple[int, int]], symbol: SchematicSymbol, color: tuple[int, int, int], width: int) -> None:
    """Draw a test point symbol in the fallback PNG preview."""

    x, y = p((symbol.x, symbol.y))
    r = max(4, int(round(1.27 * max(1, width + 1))))
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(1, width))


def _draw_fallback_symbol_png(draw: ImageDraw.ImageDraw, p: Callable[[Point], tuple[int, int]], symbol: SchematicSymbol, color: tuple[int, int, int], width: int) -> None:
    """Draw a generic rectangle for unknown schematic symbols."""

    x1, y1, x2, y2 = _bbox_symbol(symbol)
    draw.rectangle((*p((x1, y1)), *p((x2, y2))), outline=color, width=max(1, width))


def _symbol_point(symbol: SchematicSymbol, x: float, y: float) -> Point:
    """Transform a local symbol drawing point into schematic coordinates."""

    dx, dy = _rotate_local_point((x, y), symbol.rotation)
    return round(symbol.x + dx, 3), round(symbol.y + dy, 3)

