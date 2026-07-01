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


def _load_model(path: Path) -> dict[str, object]:
    """Read the schematic text and parse symbols, wires, labels, junctions, and pin definitions."""

    text = path.read_text(encoding="utf-8")
    pin_definitions = _load_pin_definitions(text)
    return {
        "text": text,
        "symbols": _load_symbols(text, pin_definitions),
        "wires": _load_wires(text),
        "labels": _load_labels(text),
        "junctions": _load_junctions(text),
    }


def _load_pin_definitions(text: str) -> dict[str, dict[str, Point]]:
    """Extract local library pin coordinates from KiCad symbol definitions."""

    result: dict[str, dict[str, Point]] = {}
    for _start, _end, shape in _forms(text, "symbol"):
        header = re.match(r'\(symbol\s+"([^"]+)"', shape.strip())
        if not header:
            continue
        lib_id = header.group(1)
        if ":" not in lib_id:
            continue
        pins: dict[str, Point] = {}
        for _p_start, _p_end, pin_shape in _forms(shape, "pin"):
            at = re.search(r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)", pin_shape)
            number = re.search(r'\(number\s+"([^"]+)"', pin_shape)
            if not at or not number:
                continue
            pins[number.group(1)] = (float(at.group(1)), float(at.group(2)))
        if pins:
            result[lib_id] = pins
    return result


def _load_symbols(text: str, pin_definitions: dict[str, dict[str, Point]] | None = None) -> list[SchematicSymbol]:
    """Parse placed symbol instances and attach pin coordinates when definitions are available."""

    result: list[SchematicSymbol] = []
    for start, end, shape in _forms(text, "symbol"):
        if "(lib_id" not in shape:
            continue
        lib = re.search(r'\(lib_id\s+"([^"]+)"\)', shape)
        lib_id = lib.group(1) if lib else ""
        at = re.search(r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)(?:\s+([-+]?\d+(?:\.\d+)?))?", shape)
        if not at:
            continue
        ref = re.search(r'\(property\s+"Reference"\s+"([^"]*)"', shape)
        x = float(at.group(1))
        y = float(at.group(2))
        rotation = float(at.group(3) or 0.0)
        width, height = _symbol_size(shape, rotation)
        pins = _instance_pins(x, y, rotation, pin_definitions.get(lib_id, {}) if pin_definitions else {})
        result.append(SchematicSymbol(
            ref.group(1) if ref else f"S{len(result) + 1}",
            x,
            y,
            rotation,
            (start, end),
            shape,
            width=width,
            height=height,
            lib_id=lib_id,
            pins=pins,
        ))
    return result


def _instance_pins(x: float, y: float, rotation: float, definition: dict[str, Point]) -> dict[str, Point]:
    """Transform library-local pin coordinates into schematic coordinates for one instance."""

    return {
        number: (round(x + dx, 3), round(y + dy, 3))
        for number, point in definition.items()
        for dx, dy in [_rotate_local_point(point, rotation)]
    }


def _rotate_local_point(point: Point, rotation: float) -> Point:
    """Rotate a local symbol point according to a KiCad right-angle rotation."""

    x, y = point
    rot = int(round(rotation)) % 360
    if rot == 90:
        return (-y, x)
    if rot == 180:
        return (-x, -y)
    if rot == 270:
        return (y, -x)
    return (x, y)


def _symbol_size(shape: str, rotation: float) -> tuple[float, float]:
    """Estimate symbol bounding-box size from known generated symbol types and rotation."""

    lib = re.search(r'\(lib_id\s+"([^"]+)"\)', shape)
    lib_id = lib.group(1) if lib else ""
    if lib_id in {"GeneratedSymbols:Resistor", "GeneratedSymbols:Capacitor", "GeneratedSymbols:Diode", "GeneratedSymbols:Inductor"}:
        width, height = 7.6, 7.6
    elif lib_id == "GeneratedSymbols:TP":
        width, height = 5.2, 5.2
    else:
        pinrow = re.search(r"GeneratedSymbols:PinRow_(\d+)", lib_id)
        if pinrow:
            count = max(1, int(pinrow.group(1)))
            width, height = 5.6, max(5.08, count * 2.54)
        else:
            width, height = 11.0, 11.0
    if int(rotation) % 180 == 90:
        return height, width
    return width, height


def _load_wires(text: str) -> list[SchematicWire]:
    """Parse KiCad wire objects from the schematic source text."""

    result: list[SchematicWire] = []
    for start, end, shape in _forms(text, "wire"):
        points = [(float(x), float(y)) for x, y in re.findall(r"\(xy\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\)", shape)]
        if len(points) < 2:
            continue
        uid = re.search(r'\(uuid\s+"([^"]+)"\)', shape)
        result.append(SchematicWire(len(result), _remove_only_duplicates(points), (start, end), uid.group(1) if uid else str(uuid4()), shape))
    return result


def _load_labels(text: str) -> list[SchematicLabel]:
    """Parse KiCad net labels and their coordinates from the schematic source text."""

    result: list[SchematicLabel] = []
    for start, end, shape in _forms(text, "label"):
        name = re.search(r'\(label\s+"([^"]*)"', shape)
        at = re.search(r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)", shape)
        if not name or not at:
            continue
        uid = re.search(r'\(uuid\s+"([^"]+)"\)', shape)
        result.append(SchematicLabel(
            len(result),
            name.group(1),
            float(at.group(1)),
            float(at.group(2)),
            (start, end),
            uid.group(1) if uid else str(uuid4()),
            shape,
        ))
    return result


def _load_junctions(text: str) -> list[SchematicJunction]:
    """Parse KiCad junction markers from the schematic source text."""

    result: list[SchematicJunction] = []
    for start, end, shape in _forms(text, "junction"):
        at = re.search(r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)", shape)
        if not at:
            continue
        uid = re.search(r'\(uuid\s+"([^"]+)"\)', shape)
        result.append(SchematicJunction(
            len(result),
            float(at.group(1)),
            float(at.group(2)),
            (start, end),
            uid.group(1) if uid else str(uuid4()),
            shape,
        ))
    return result


def _forms(text: str, head: str) -> list[tuple[int, int, str]]:
    """Find top-level s-expression forms matching a requested head token."""

    result: list[tuple[int, int, str]] = []
    pattern = f"({head}"
    i = 0
    while True:
        start = text.find(pattern, i)
        if start < 0:
            return result
        if start > 0 and text[start - 1] not in " \t\r\n(":
            i = start + 1
            continue
        end = _form_end(text, start)
        if end > start:
            result.append((start, end, text[start:end]))
            i = end
        else:
            i = start + 1


def _form_end(text: str, start: int) -> int:
    """Find the end offset of an s-expression form starting at a character position."""

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1

