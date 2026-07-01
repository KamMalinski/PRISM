from __future__ import annotations

import copy
import math
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from schematic_generator.correction_editor.support import *
from schematic_generator.corrections_facade import (
    add_correction,
    add_manual_element,
    recalculate_and_save_by_corrections,
    save_mask_by_corrections,
    save_state_corrections_without_recalculation,
    find_similar_pads,
)
from schematic_generator.models import Pad
from schematic_generator.netlist_facade import build_nets
from schematic_generator.ocr import read_texts, save_texts_ocr as save_ocr_texts
from schematic_generator.preview import draw_connections_colored


class CorrectionEditorGeometryMixin:
    def _pad_description(self, pad: Pad | None) -> str:
        """Formats a compact pad label for dialogs and status messages."""
        if not pad:
            return ""
        label = _label_pad(pad)
        return f"{label} ({pad.node})" if label != _pad_identifier(pad) else pad.node

    def _pads_side(self, side: str) -> list[Pad]:
        """Provides the editable pad list for the requested board side."""
        return self.state.top_pads if side == "TOP" else self.state.bottom_pads

    def _set_pads_side(self, side: str, pads: list[Pad]) -> None:
        """Replace the editable pad list for the requested board side."""
        if side == "TOP":
            self.state.top_pads = pads
        else:
            self.state.bottom_pads = pads

    def _pads_by_node(self) -> dict[str, Pad]:
        """Creates a lookup from full node name to pad object across both board sides."""
        return {pad.node: pad for pad in [*self.state.top_pads, *self.state.bottom_pads]}

    def _paired_nodes(self) -> set[str]:
        """Collect every pad node currently participating in a TOP/BOTTOM pair."""
        result: set[str] = set()
        for pair in self.state.pairs:
            result.add(pair.pad_top)
            result.add(pair.pad_bottom)
        return result

    def _colors_pairs_by_nodes(self) -> dict[str, tuple[int, int, int]]:
        """Assign stable pair colors to both pad nodes in every TOP/BOTTOM pair."""
        result: dict[str, tuple[int, int, int]] = {}
        for index, pair in enumerate(self.state.pairs):
            color = _color_pairs(index)
            result[pair.pad_top] = color
            result[pair.pad_bottom] = color
        return result

    def _counterpart(self, node: str | None) -> str | None:
        """Find the paired node on the opposite side for a given pad node."""
        if not node:
            return None
        for pair in self.state.pairs:
            if pair.pad_top == node:
                return pair.pad_bottom
            if pair.pad_bottom == node:
                return pair.pad_top
        return None

    def _expand_with_pads_paired(self, pads: list[Pad]) -> list[Pad]:
        """Expand a pad list with paired counterparts so edits stay synchronized across board sides."""
        pads_by_node = self._pads_by_node()
        result: dict[str, Pad] = {}
        for pad in pads:
            result[pad.node] = pad
            second_node = self._counterpart(pad.node)
            if second_node and second_node in pads_by_node:
                result[second_node] = pads_by_node[second_node]
        return list(result.values())

    def _nearest_pad(self, pads: list[Pad], x: float, y: float, side: str, without_limit: bool = False) -> Pad | None:
        """Find the nearest clickable pad within the zoom-adjusted hit radius."""
        if not pads:
            return None
        pad = min(pads, key=lambda p: math.hypot(p.x - x, p.y - y))
        distance = math.hypot(pad.x - x, pad.y - y)
        scale = float(self.canvas_info[side].get("scale", 1.0))
        limit = max(13.0 / max(scale, 0.1), pad.radius * 3.0)
        if without_limit or distance <= limit:
            return pad
        return None

    def _point_image(self, event: tk.Event, side: str) -> tuple[float, float]:
        """Convert canvas event coordinates into image pixel coordinates for the requested side."""
        canvas = self.canvas_info[side].get("canvas")
        scale = float(self.canvas_info[side].get("scale", 1.0))
        if isinstance(canvas, tk.Canvas):
            return canvas.canvasx(event.x) / max(scale, 0.01), canvas.canvasy(event.y) / max(scale, 0.01)
        return event.x / max(scale, 0.01), event.y / max(scale, 0.01)

    def _mouse_move(self, event: tk.Event, side: str) -> None:
        """Update OCR rectangle previews while the pointer moves in OCR selection modes."""
        mode = self.mode.get()
        if mode not in {"ocr_region", "ocr_manual_region"}:
            return
        x, y = self._point_image(event, side)
        self._update_ocr_region_preview(side, x, y)

    def _update_ocr_region_preview(self, side: str, x: float, y: float) -> None:
        """Updates the live OCR rectangle using the saved first corner and current pointer position."""
        if not self.ocr_region_start or self.ocr_region_start[0] != side:
            return
        _start_side, x0, y0, _manual = self.ocr_region_start
        self._set_ocr_region_preview(side, x0, y0, x, y)

    def _set_ocr_region_preview(self, side: str, x0: float, y0: float, x1: float, y1: float) -> None:
        """Clamp an OCR rectangle to image bounds and redraw the affected side."""
        width, height = self.images[side].size
        ax = min(max(0.0, x0), float(width - 1))
        bx = min(max(0.0, x1), float(width - 1))
        ay = min(max(0.0, y0), float(height - 1))
        by = min(max(0.0, y1), float(height - 1))
        x_min, x_max = sorted((ax, bx))
        y_min, y_max = sorted((ay, by))
        self.ocr_region_preview = {side: [x_min, y_min, max(1.0, x_max - x_min), max(1.0, y_max - y_min)]}
        self._redraw_side(side)

    def _mouse_wheel(self, event: tk.Event, side: str, direction: int | None = None) -> str:
        """Use native wheel events for Ctrl-zoom or normal canvas scrolling."""

        wheel_direction = direction if direction is not None else (1 if event.delta > 0 else -1)
        if event.state & 0x0004:
            self._change_zoom(side, 1.15 if wheel_direction > 0 else 0.87)
        else:
            canvas = self.canvas_info[side].get("canvas")
            if isinstance(canvas, tk.Canvas):
                canvas.yview_scroll(-1 if wheel_direction > 0 else 1, "units")
        return "break"

    def _change_zoom(self, side: str | None, multiplier: float) -> None:
        """Apply a bounded zoom multiplier to one or both board canvases."""
        side = (side,) if side else ("TOP", "BOTTOM")
        for s in side:
            self.zoom[s] = min(6.0, max(0.35, self.zoom[s] * multiplier))
        self._redraw_all()

    def _align_zoom(self) -> None:
        """Reset both canvases to the same default zoom level."""
        self.zoom = {"TOP": 1.0, "BOTTOM": 1.0}
        self._redraw_all()

    def _typical_radius(self, side: str) -> float:
        """Estimate a reasonable manual pad radius from existing pads on the same side."""
        pads = self._pads_side(side)
        if not pads:
            return 6.0
        radii = sorted(p.radius for p in pads)
        return max(3.0, radii[len(radii) // 2])

    def _next_id(self, side: str) -> str:
        """Generate the next unused manual pad identifier on the requested side."""
        existing_refs = {_pad_identifier(pad) for pad in self._pads_side(side)}
        index = 1
        while f"M{index:04d}" in existing_refs:
            index += 1
        return f"M{index:04d}"

    def _sorted_selected_pads(self) -> list[Pad]:
        """Orders selected pads by click order with a stable fallback for any missing order entries."""
        pads_by_node = self._pads_by_node()
        order = [w for w in self.selected_nodes_order if w in self.selected_nodes]
        missing_nodes = sorted(self.selected_nodes.difference(order))
        pads = [pads_by_node[w] for w in [*order, *missing_nodes] if w in pads_by_node]
        return sorted(pads, key=lambda p: (p.side, p.y, p.x))

    def _default_ref(self) -> str:
        """Generate the next component reference prefix based on the current selected pad count."""
        prefix = "R" if len(self.selected_nodes) == 2 else "U"
        existing_refs = {str(k.get("ref", "")) for k in self.state.manual_components}
        index = 1
        while f"{prefix}{index}" in existing_refs:
            index += 1
        return f"{prefix}{index}"
