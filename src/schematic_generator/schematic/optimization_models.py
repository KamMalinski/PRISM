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


@dataclass(slots=True)
class SchematicSymbol:
    """Store one schematic symbol instance together with placement and parsed pin geometry."""

    ref: str
    x: float
    y: float
    rotation: float
    span: tuple[int, int]
    source_shape: str = ""
    changed: bool = False
    dx: float = 0.0
    dy: float = 0.0
    width: float = 14.0
    height: float = 10.0
    lib_id: str = ""
    pins: dict[str, Point] = field(default_factory=dict)


@dataclass(slots=True)
class SchematicWire:
    """Store one KiCad wire with parsed points, source span, and optimization metadata."""

    index: int
    points: list[Point]
    span: tuple[int, int]
    uuid: str
    source_shape: str = ""
    changed: bool = False
    net: str = ""


@dataclass(slots=True)
class SchematicLabel:
    """Store one schematic net label and the movement applied during optimization."""

    index: int
    text: str
    x: float
    y: float
    span: tuple[int, int]
    uuid: str
    source_shape: str = ""
    changed: bool = False
    dx: float = 0.0
    dy: float = 0.0


@dataclass(slots=True)
class SchematicJunction:
    """Store one schematic junction marker parsed from the KiCad file."""

    index: int
    x: float
    y: float
    span: tuple[int, int]
    uuid: str
    source_shape: str = ""


@dataclass(slots=True)
class RoutingNet:
    """Describe one net as terminals plus source objects that can be replaced during autorouting."""

    name: str
    terminals: list[Point]
    wire_indices: list[int]
    label_indices: list[int]
    junction_indices: list[int]
    pin_refs: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class AutoroutingResult:
    """Collect wires, labels, junctions, and source spans produced by one routing pass."""

    wires: list[SchematicWire]
    labels: list[SchematicLabel]
    junctions: list[SchematicJunction]
    remove_spans: list[tuple[int, int]]
    label_jumps: int = 0


@dataclass(slots=True)
class SchematicMetrics:
    """Summarize layout quality metrics used to score optimization candidates."""

    count_elements: int
    count_wires: int
    length_wires: float
    crossing_count: int
    min_distance_elements: float
    average_distance_nearest: float
    conflicts_wire_element: int
    count_bends: int = 0
    count_backtracks: int = 0


@dataclass(slots=True)
class OptimizationConfig:
    """Hold user-tunable limits and scoring weights for schematic optimization."""

    time_limit_s: int = 30
    thread_count: int = 1
    routing_iterations: int = 24
    population_size: int = 96
    mutation_rate: float = 0.18
    penalty_length: float = 1.0
    penalty_crossings: float = 40.0
    penalty_wire_element: float = 20.0
    penalty_nearby_elements: float = 5.0
    reward_distances: float = 0.2


@dataclass(slots=True)
class OptimizationProgress:
    """Report one optimizer progress sample to the GUI or caller callback."""

    generation: int
    result: float
    best_result: float
    metrics: SchematicMetrics


@dataclass(slots=True)
class OptimizationResult:
    """Return generated artifacts, quality metrics, and change counters after optimization."""

    kicad_path: Path
    png_path: Path
    svg_path: Path | None
    metrics_before: SchematicMetrics
    metrics_after: SchematicMetrics
    result: float
    generations: int
    improved: bool
    changed_wire_count: int
    changed_element_count: int

