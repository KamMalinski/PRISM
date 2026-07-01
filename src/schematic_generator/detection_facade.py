from __future__ import annotations

from schematic_generator.detection.colors import analyze_colors_pcb
from schematic_generator.detection.common import ColorProfile, LogFn
from schematic_generator.detection.pads import detect_pads
from schematic_generator.detection.pairs import find_pairs_holes
from schematic_generator.detection.plane import detect_plane_copper
from schematic_generator.detection.traces import refine_trace_mask

__all__ = [
    "ColorProfile",
    "LogFn",
    "analyze_colors_pcb",
    "detect_pads",
    "detect_plane_copper",
    "find_pairs_holes",
    "refine_trace_mask",
]
