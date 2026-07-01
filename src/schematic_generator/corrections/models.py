from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schematic_generator.models import Element, HolePair, ManualCorrection, Net, Pad

LogFn = Callable[[str], None]

CORRECTED_SCHEMATIC_STEM = "schematic_corrected"
OPTIMIZED_CORRECTED_SCHEMATIC_KEY = "optimized_corrected_schematic"
CORRECTED_HOLES_ALIGNMENT_PNG = "alignment_holes_corrected.png"
CORRECTED_TOP_CONNECTIONS_PNG = "top_connections_color_corrected.png"
CORRECTED_BOTTOM_CONNECTIONS_PNG = "bottom_connections_color_corrected.png"


@dataclass(slots=True)
class CorrectionState:
    """Mutable reconstruction state edited by the correction window before final recalculation."""

    output_folder: Path
    work_folder: Path
    parameters: dict[str, Any]
    top_pads: list[Pad]
    bottom_pads: list[Pad]
    pairs: list[HolePair]
    ocr_texts: list[dict[str, float | int | str]]
    corrections: list[ManualCorrection]
    manual_components: list[dict[str, Any]]
    pending_recalculation: bool = False


@dataclass(slots=True)
class RecalculationResult:
    """Paths and in-memory objects produced after applying manual corrections."""

    nets: list[Net]
    elements: list[Element]
    netlist_path: Path
    devices_path: Path
    kicad_path: Path
