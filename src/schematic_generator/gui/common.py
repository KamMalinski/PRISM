from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

SAMPLE_STEPS = (
    ("TOP", "background_bgr"),
    ("TOP", "trace_bgr"),
    ("BOTTOM", "background_bgr"),
    ("BOTTOM", "trace_bgr"),
)

SAMPLE_NAMES = {
    "trace_bgr": "trace / copper",
    "background_bgr": "soldermask background",
}

OPT_TIME_LIMIT_FIELD = "time_limit_s"
OPT_THREADS_FIELD = "thread_count"
OPT_POPULATION_FIELD = "population_size"
OPT_IMPROVED_FIELD = "improved"
OPT_GENERATION_FIELD = "generation"
OPTIMIZED_SCHEMATIC_PNG = "schematic_optimized.png"
CORRECTED_SCHEMATIC_PNG = "schematic_corrected.png"
OPTIMIZED_CORRECTED_SCHEMATIC_PNG = "schematic_corrected_optimized.png"
CORRECTED_TOP_CONNECTIONS_PNG = "top_connections_color_corrected.png"
CORRECTED_BOTTOM_CONNECTIONS_PNG = "bottom_connections_color_corrected.png"


def ui_font(widget: tk.Misc, size: int, *, bold: bool = False) -> tuple[str, int, str]:
    """Return a native Tk font family with the requested size and weight."""

    family = str(tkfont.nametofont("TkDefaultFont", root=widget).actual("family"))
    return family, size, "bold" if bold else "normal"


def hex_from_bgr(bgr: tuple[int, int, int]) -> str:
    """Convert an OpenCV BGR color tuple into a Tkinter-compatible hex color."""

    b, g, r = (max(0, min(255, int(x))) for x in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def text_on_background(bgr: tuple[int, int, int]) -> str:
    """Choose readable dark or light text for a BGR color swatch."""

    b, g, r = (int(x) for x in bgr)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if luminance > 145 else "#ffffff"
