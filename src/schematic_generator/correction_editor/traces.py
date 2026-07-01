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


class CorrectionEditorTracesMixin:
    def _click(self, event: tk.Event, side: str) -> None:
        """Dispatch a canvas click to the operation selected by the current tool mode."""
        x, y = self._point_image(event, side)
        mode = self.mode.get()
        if mode == "add":
            self._add_pad(side, x, y)
        elif mode == "delete":
            self._remove_pad(side, x, y)
        elif mode == "rename_pad":
            self._name_pad(side, x, y)
        elif mode == "pair":
            self._pairing_click(side, x, y)
        elif mode == "unpair":
            self._disconnect_pair(side, x, y)
        elif mode == "trace_bridge":
            self._bridge_click(side, x, y)
        elif mode == "trace_cut":
            self._preview_crossings(side, x, y)
        elif mode in {"trace_add", "trace_remove", "trace_ignore"}:
            self._paint_trace(side, x, y, mode)
        elif mode in {"ocr_region", "ocr_manual_region"}:
            self._click_ocr_region(side, x, y, manual=mode == "ocr_manual_region")
        elif mode == "component":
            self._toggle_pad_element(side, x, y)
        else:
            pad = self._nearest_pad(self._pads_side(side), x, y, side)
            self.selected_node = pad.node if pad else None
            self.drag_node = pad.node if pad else None
            self.drag_start_pad = (pad.x, pad.y) if pad else None
            self.drag_snapshot_pushed = False
            if pad:
                self.selected_trace = None
                self.type_pad.set(pad.type or "pad")
                if pad.node not in self.selected_nodes:
                    self.selected_nodes_order = [pad.node]
            else:
                self.selected_trace = (side, x, y) if self._trace_at(side, x, y) else None
                if self.selected_trace:
                    self.status.set(f"Trace selected on {side}. Press Delete to remove it.")
            self._refresh_status()
            self._redraw_all()

    def _drag(self, event: tk.Event, side: str) -> None:
        """Tracks pointer dragging for pad movement, trace painting, and OCR rectangle preview."""
        x, y = self._point_image(event, side)
        mode = self.mode.get()
        if mode in {"trace_add", "trace_remove", "trace_ignore"}:
            self._paint_trace(side, x, y, mode)
            return
        if mode in {"ocr_region", "ocr_manual_region"}:
            self._update_ocr_region_preview(side, x, y)
            return
        if mode != "move" or not self.drag_node:
            return
        pad = self._pads_by_node().get(self.drag_node)
        if not pad:
            return
        if not self.drag_snapshot_pushed:
            self._push_undo()
            self.drag_snapshot_pushed = True
        width, height = self.images[pad.side].size
        pad.x = min(max(0.0, x), float(width - 1))
        pad.y = min(max(0.0, y), float(height - 1))
        self.selected_node = pad.node
        self._redraw_all()

    def _end_drag(self, _event: tk.Event, side: str) -> None:
        """Commit trace painting, finish OCR rectangle selection, or record a completed pad movement when the mouse is released."""
        mode = self.mode.get()
        if mode in {"ocr_region", "ocr_manual_region"} and self.ocr_region_start and self.ocr_region_start[0] == side:
            x, y = self._point_image(_event, side)
            _start_side, x0, y0, _manual = self.ocr_region_start
            if abs(x - x0) >= 4 and abs(y - y0) >= 4:
                self._finish_ocr_region_selection(side, x, y)
            return
        if mode in {"trace_add", "trace_remove", "trace_ignore"} and self.trace_dirty:
            self._save_mask_side(side)
            type_corrections = {
                "trace_add": "mask_add",
                "trace_remove": "mask_remove",
                "trace_cut": "mask_cut",
                "trace_ignore": "mask_ignore",
            }.get(mode, "edit_trace_mask")
            add_correction(
                self.state,
                type_corrections,
                f"Edited trace mask {side}.",
                {"side": side, "mode": mode, "brush": int(self.brush_size.get())},
            )
            self.trace_dirty = False
            self._save_corrections_and_redraw()
            return

        if mode != "move" or not self.drag_node or not self.drag_start_pad:
            return
        pad = self._pads_by_node().get(self.drag_node)
        start_x, start_y = self.drag_start_pad
        self.drag_node = None
        self.drag_start_pad = None
        self.drag_snapshot_pushed = False
        if not pad:
            return
        if math.hypot(pad.x - start_x, pad.y - start_y) < 1.0:
            return
        add_correction(
            self.state,
            "move_pad",
            f"Moved pad {pad.node}.",
            {"pad": pad.node, "from": [round(start_x, 2), round(start_y, 2)], "to": [round(pad.x, 2), round(pad.y, 2)]},
        )
        self._save_corrections_and_redraw()

    def _paint_trace(self, side: str, x: float, y: float, mode: str) -> None:
        """Paint into the editable trace mask using the active brush and mark the mask as dirty."""
        if not self.trace_dirty:
            self._push_undo()
        r = max(1, int(self.brush_size.get()))
        draw = ImageDraw.Draw(self.mask[side])
        color = 255 if mode == "trace_add" else 0
        if mode == "trace_cut":
            r *= 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        self.trace_dirty = True
        self._redraw_side(side)

    def _preview_crossings(self, side: str, x: float, y: float) -> None:
        """Simulate the netlist before and after a trace cut and ask for confirmation with nearby net context."""
        r = max(2, int(self.brush_size.get()) * 2)
        before = self._simulate_nets(self.mask)
        mask_by = {s: self.mask[s].copy() for s in ("TOP", "BOTTOM")}
        draw = ImageDraw.Draw(mask_by[side])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=0)
        by = self._simulate_nets(mask_by)
        before_names = self._nets_near_point(before, side, x, y)
        by_names = self._nets_near_point(by, side, x, y)
        message = (
            f"Cut {side} at point ({x:.1f}, {y:.1f}), radius {r}px.\n\n"
            f"Net count before: {len(before['nets'])}\n"
            f"Net count after: {len(by['nets'])}\n"
            f"Nets near point before: {', '.join(before_names) or 'none'}\n"
            f"Nets near point after: {', '.join(by_names) or 'none'}\n\n"
            "Apply cut?"
        )
        if not messagebox.askyesno("Cut preview", message, parent=self):
            return
        self._push_undo()
        self.mask = mask_by
        self._save_mask_side(side)
        add_correction(
            self.state,
            "mask_cut",
            f"Cut trace {side}.",
            {
                "side": side,
                "point": [round(x, 2), round(y, 2)],
                "radius": r,
                "net_count_before": len(before["nets"]),
                "net_count_after": len(by["nets"]),
                "near_before": before_names,
                "near_after": by_names,
            },
        )
        self._save_corrections_and_redraw()

    def _simulate_nets(self, mask: dict[str, Image.Image]) -> dict[str, Any]:
        """Constructs a temporary netlist from copied pads and a candidate mask without mutating editor state."""
        top_pads = copy.deepcopy(self.state.top_pads)
        bottom_pads = copy.deepcopy(self.state.bottom_pads)
        pairs = copy.deepcopy(self.state.pairs)
        nets = build_nets(
            top_pads,
            bottom_pads,
            np.array(mask["TOP"], dtype=np.uint8),
            np.array(mask["BOTTOM"], dtype=np.uint8),
            pairs,
            None,
            None,
            None,
        )
        return {"nets": nets, "pads": [*top_pads, *bottom_pads]}

    def _nets_near_point(self, result: dict[str, Any], side: str, x: float, y: float) -> list[str]:
        """Find net names whose pads are close enough to explain a proposed trace edit."""
        pads = [pad for pad in result["pads"] if pad.side == side and pad.net]
        if not pads:
            return []
        nearby = [pad for pad in pads if math.hypot(pad.x - x, pad.y - y) <= max(80.0, pad.radius * 8.0)]
        if not nearby:
            nearby = sorted(pads, key=lambda pad: math.hypot(pad.x - x, pad.y - y))[:3]
        return sorted({pad.net for pad in nearby if pad.net})

    def _bridge_click(self, side: str, x: float, y: float) -> None:
        """Collect two endpoints and draw a conductive bridge into the trace mask."""
        if not self.trace_start or self.trace_start[0] != side:
            self.trace_start = (side, x, y)
            self.status.set(f"Bridge {side}: point A selected, click point B.")
            return
        _, x0, y0 = self.trace_start
        self.trace_start = None
        self._push_undo()
        r = max(2, int(self.brush_size.get()))
        draw = ImageDraw.Draw(self.mask[side])
        draw.line((x0, y0, x, y), fill=255, width=max(2, r * 2))
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=255)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
        self._save_mask_side(side)
        add_correction(
            self.state,
            "mask_bridge",
            f"Added trace bridge {side}.",
            {"side": side, "from": [round(x0, 2), round(y0, 2)], "to": [round(x, 2), round(y, 2)], "brush": r},
        )
        self._save_corrections_and_redraw()

    def _save_mask_side(self, side: str) -> None:
        """Persist one side of the editable trace mask to the corrections work folder."""
        mask = np.array(self.mask[side], dtype=np.uint8)
        save_mask_by_corrections(self.state.work_folder, side, mask)

    def _trace_at(self, side: str, x: float, y: float) -> bool:
        """Return whether a board coordinate sits on the editable trace mask."""
        mask = np.array(self.mask[side], dtype=np.uint8)
        height, width = mask.shape[:2]
        xi = min(width - 1, max(0, int(round(x))))
        yi = min(height - 1, max(0, int(round(y))))
        radius = 2
        y0 = max(0, yi - radius)
        y1 = min(height, yi + radius + 1)
        x0 = max(0, xi - radius)
        x1 = min(width, xi + radius + 1)
        return bool(np.any(mask[y0:y1, x0:x1] > 127))

    def _delete_selected_trace(self) -> bool:
        """Delete the connected trace component under the selected trace point."""
        if not self.selected_trace:
            return False
        side, x, y = self.selected_trace
        mask = np.array(self.mask[side], dtype=np.uint8)
        height, width = mask.shape[:2]
        xi = min(width - 1, max(0, int(round(x))))
        yi = min(height - 1, max(0, int(round(y))))
        binary = (mask > 127).astype(np.uint8)
        count, labels = cv2.connectedComponents(binary, 8)
        if count <= 1:
            self.status.set("No trace is selected.")
            self.selected_trace = None
            return False
        label = int(labels[yi, xi])
        if label <= 0:
            radius = 2
            y0 = max(0, yi - radius)
            y1 = min(height, yi + radius + 1)
            x0 = max(0, xi - radius)
            x1 = min(width, xi + radius + 1)
            nearby = labels[y0:y1, x0:x1]
            positive = nearby[nearby > 0]
            label = int(positive[0]) if positive.size else 0
        if label <= 0:
            self.status.set("No trace is selected.")
            self.selected_trace = None
            return False
        pixel_count = int(np.count_nonzero(labels == label))
        if not messagebox.askyesno(
            "Delete trace",
            f"Delete the selected connected trace on {side} ({pixel_count} pixels)?",
            parent=self,
        ):
            return True
        self._push_undo()
        mask[labels == label] = 0
        self.mask[side] = Image.fromarray(mask).convert("L")
        self._save_mask_side(side)
        add_correction(
            self.state,
            "mask_remove",
            f"Deleted selected trace {side}.",
            {"side": side, "point": [round(x, 2), round(y, 2)], "pixels": pixel_count},
        )
        self.selected_trace = None
        self._save_corrections_and_redraw()
        return True
